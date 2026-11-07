"""A Hugging Face model run on this computer (MedGemma). No dataset image leaves the machine: the weights are downloaded once
and then read from the local cache. The model is gated, so the download needs a Hugging Face access token (``HF_TOKEN`` in
``.env``) of an account that has accepted the model's terms.

Needs the packages of ``requirements.txt`` and, for a GPU, PyTorch's CUDA build installed first from pytorch.org. Decoding is greedy
(``do_sample=False``), the temperature-0 setting of the base paper. A model that does not fit in the GPU memory is split
between the GPU and the RAM (``device_map="auto"`` within ``gpu_memory_gib`` and ``cpu_memory_gib``): slower, but every number is
still computed in the model's own precision ``dtype``.
"""

from __future__ import annotations

import io
import re
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

from . import FatalError

DTYPES = ("bfloat16", "float16", "float32")
COMMIT = re.compile(r"[0-9a-f]{40}")


def _resolved_commit(model_id: str, revision: str) -> str | None:
    """The commit the model files come from: the revision itself when it is a commit hash, else read from the download cache
    (``.../snapshots/<commit>/config.json``). None when it cannot be determined."""
    if COMMIT.fullmatch(revision):
        return revision
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    path = try_to_load_from_cache(model_id, "config.json", revision=revision)
    commit = Path(path).parent.name if isinstance(path, str) else ""
    return commit if COMMIT.fullmatch(commit) else None


def _number(settings: dict, key: str) -> float | None:
    value = settings.get(key)
    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0):
        raise ValueError(f"[model] {key} must be a number above 0 (got {value!r})")
    return value


def _explain_load(exc: Exception, model_id: str) -> str:
    """Why the model could not be loaded, in plain words, with what to do about the common causes."""
    text = str(exc)
    low = text.lower()
    if "gated" in low or "restricted" in low or re.search(r"\b(401|403)\b", low):
        return (f"Hugging Face did not give access to {model_id}. Open the model page while logged in and accept its terms, create "
                "an access token (Settings, Access Tokens, type Read), and put the line HF_TOKEN=<your token> in the .env file "
                f"next to README.md. Hugging Face said: {text.splitlines()[0] if text else type(exc).__name__}")
    if "out of memory" in low:
        return (f"{model_id} does not fit in the memory it was given. Lower gpu_memory_gib or raise cpu_memory_gib in the config "
                f"(a changed setting is a new config file). Details: {text.splitlines()[0]}")
    return f"{model_id} could not be loaded ({type(exc).__name__}: {text.splitlines()[0] if text else 'no details'})"


class HFLocalModel:
    packages = ("torch", "transformers", "accelerate", "huggingface-hub", "pillow")  # recorded in every run card
    loaded_note = "model files found and loaded"  # replaced, once loaded, by a note that names the commit
    probe_note = "it runs on this computer, so it uses no quota; the first answer can take a while"

    def __init__(self, settings: dict):
        missing = [key for key in ("id", "dtype", "seed", "max_new_tokens") if key not in settings]
        if missing:
            raise ValueError(f"[model] is missing {', '.join(missing)} (the hf_local adapter needs id, dtype, seed and max_new_tokens)")
        self.model_id, self.dtype_name, self.seed = settings["id"], settings["dtype"], settings["seed"]
        self.max_new_tokens, self.revision = settings["max_new_tokens"], settings.get("revision", "main")
        if not isinstance(self.model_id, str) or not self.model_id or not isinstance(self.revision, str) or not self.revision:
            raise ValueError("[model] id and revision must be non-empty text")
        if self.dtype_name not in DTYPES:
            raise ValueError(f"[model] dtype must be one of {list(DTYPES)} (got {self.dtype_name!r})")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError(f"[model] seed must be a whole number (got {self.seed!r})")
        if isinstance(self.max_new_tokens, bool) or not isinstance(self.max_new_tokens, int) or self.max_new_tokens < 1:
            raise ValueError(f"[model] max_new_tokens must be a whole number of at least 1 (got {self.max_new_tokens!r})")
        gpu, cpu = _number(settings, "gpu_memory_gib"), _number(settings, "cpu_memory_gib")
        if (gpu is None) != (cpu is None):
            raise ValueError("[model] set gpu_memory_gib and cpu_memory_gib together, or neither")
        try:
            import torch
            import transformers
        except ImportError as exc:
            raise ValueError("PyTorch and transformers are not installed: first install PyTorch from pytorch.org (the CUDA build "
                             "for a GPU), then run pip install -r requirements.txt") from exc
        self.torch = torch
        self.cuda = bool(torch.cuda.is_available())
        kwargs = {"dtype": getattr(torch, self.dtype_name), "device_map": "auto" if self.cuda else "cpu"}
        self.max_memory = {0: f"{gpu}GiB", "cpu": f"{cpu}GiB"} if self.cuda and gpu is not None else None
        if self.max_memory:
            kwargs["max_memory"] = self.max_memory
        if settings.get("offload_folder"):
            kwargs["offload_folder"] = settings["offload_folder"]
        print(f"Loading {self.model_id} (revision {self.revision}); the first time about 9 GB are downloaded, which can take a long "
              f"time. {'GPU found.' if self.cuda else 'NO GPU found: this will be very slow.'}", flush=True)
        try:
            self.model = transformers.AutoModelForImageTextToText.from_pretrained(self.model_id, revision=self.revision, **kwargs)
            self.processor = transformers.AutoProcessor.from_pretrained(self.model_id, revision=self.revision)
        except ImportError as exc:  # for example accelerate, which device_map="auto" needs
            raise ValueError(f"A package that {self.model_id} needs is missing ({exc}). Run pip install -r requirements.txt") from exc
        except Exception as exc:  # a refused or failed download arrives as different error types in different versions
            raise ValueError(_explain_load(exc, self.model_id)) from exc
        self.model.eval()
        self.transformers_version = getattr(transformers, "__version__", None)
        self.commit = _resolved_commit(self.model_id, self.revision)
        if self.commit is None:
            print("WARNING: the commit of the model files could not be determined, so the run cannot record exactly which files "
                  "it used. Set revision in the config to a commit hash.", file=sys.stderr)
        self.loaded_note = f"{self.model_id} loaded at commit {self.commit or 'UNKNOWN'} (the run itself uses no API key)"
        eos = getattr(self.model.generation_config, "eos_token_id", None)
        self.eos = set(eos) if isinstance(eos, (list, tuple)) else ({eos} if eos is not None else set())

    def describe(self) -> dict:
        placed = getattr(self.model, "hf_device_map", None)
        return {"provider": "hf_local", "model_id": self.model_id, "revision_requested": self.revision,
                "revision_loaded": self.commit, "sdk": "transformers", "sdk_version": self.transformers_version,
                "torch_version": getattr(self.torch, "__version__", None), "cuda": self.cuda,
                "layers_per_device": dict(Counter(str(device) for device in placed.values())) if placed else None,
                "generation_config": {"do_sample": False, "max_new_tokens": self.max_new_tokens, "seed": self.seed,
                                      "dtype": self.dtype_name, "max_memory": self.max_memory}}

    def predict(self, image: bytes, mime_type: str, prompt: str) -> dict:
        try:
            picture = Image.open(io.BytesIO(image)).convert("RGB")
        except Exception as exc:  # a damaged image is not worth a retry
            raise FatalError(f"the image could not be opened ({type(exc).__name__}: {exc})") from exc
        messages = [{"role": "user", "content": [{"type": "image", "image": picture}, {"type": "text", "text": prompt}]}]
        torch = self.torch
        try:
            inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                                        return_tensors="pt")
            inputs = inputs.to(self.model.device, dtype=getattr(torch, self.dtype_name))
            n_in = inputs["input_ids"].shape[-1]
            torch.manual_seed(self.seed)
            with torch.inference_mode():
                generated = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise FatalError("The GPU ran out of memory while answering. Lower gpu_memory_gib in a new config file, or close "
                                 "other programs that use the graphics card.") from exc
            raise FatalError(f"generation failed ({type(exc).__name__}: {exc})") from exc
        new_ids = generated[0][n_in:]
        ids = [int(i) for i in new_ids]
        text = self.processor.decode(new_ids, skip_special_tokens=True).strip()
        if ids and ids[-1] in self.eos:
            finish = "STOP"
        elif len(ids) >= self.max_new_tokens:
            finish = "MAX_TOKENS"  # cut off at the limit
        else:
            finish = "OTHER"
        return {"text": text or None, "finish_reason": finish, "model_version": f"{self.model_id}@{self.commit or self.revision}",
                "response_id": None,
                "usage": {"prompt_tokens": int(n_in), "output_tokens": len(ids), "total_tokens": int(n_in) + len(ids)}}

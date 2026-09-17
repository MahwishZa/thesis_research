"""Model-backed generator: one Hugging Face causal LM, shared by every arm.

**One instance, both arms.** `RunConfig` stamps a single model name and
generation config onto every result record, and `assert_generator_parity`
refuses a run whose arms hold different generator objects. This class is built
to be constructed once and passed to both.

**Greedy decoding is the default and is not a tuning knob.** Sampling would
make the experiment a distribution comparison requiring many runs per question
to separate the intervention from decoding variance, which the hardware budget
does not allow. With `do_sample=False`, one run per arm is the whole
measurement and a repeat run reproduces it. Anything that would reintroduce
variance - temperature, top-p, top-k, a seed that matters - is therefore
absent rather than merely defaulted.

``torch`` and ``transformers`` are imported inside methods so this module
stays importable where they are not installed; the tests never load a model.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from .evidence import Evidence
from .generator import GenerationResult, Generator


class GeneratorError(RuntimeError):
    """Raised when a generator cannot be configured or run reproducibly."""


@dataclass(frozen=True)
class GenerationConfig:
    """Decoding settings, identical for every arm and recorded per run.

    ``max_new_tokens`` caps answer length. It is a control, not a treatment:
    an arm allowed longer answers has more opportunity to make an unsupported
    claim, so the cap is shared like the context budget.
    """

    max_new_tokens: int = 256
    #: Greedy. Kept as an explicit field so a run that changes it has to say so
    #: in the recorded config rather than changing behaviour invisibly.
    do_sample: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise GeneratorError("max_new_tokens must be positive")
        if not self.do_sample and (self.temperature is not None
                                   or self.top_p is not None):
            raise GeneratorError(
                "temperature/top_p are set but do_sample is False. Greedy "
                "decoding ignores them, so recording them would misdescribe "
                "the run."
            )
        if self.do_sample:
            raise GeneratorError(
                "sampling is not supported for the primary comparison: it "
                "makes one run per arm an estimate rather than the "
                "measurement, and separating the intervention from decoding "
                "variance would need repeated runs per question. Use greedy "
                "decoding."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }


@dataclass(frozen=True)
class ModelSpec:
    """Exactly which weights ran, pinned hard enough to reproduce.

    ``revision`` is required and must not be ``main``: a branch name resolves
    to different weights over time, so a result recorded against one could not
    be reproduced later. A commit sha can.
    """

    model_id: str
    revision: str
    quantization: Optional[str] = None
    dtype: str = "bfloat16"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.revision or self.revision == "main":
            raise GeneratorError(
                f"{self.model_id}: revision must be a pinned commit sha, not "
                f"{self.revision!r}. A branch name resolves to different "
                "weights over time and the run would not be reproducible."
            )
        if self.quantization not in (None, "nf4", "int8"):
            raise GeneratorError(
                f"unsupported quantization: {self.quantization!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "quantization": self.quantization,
            "dtype": self.dtype,
            "metadata": dict(self.metadata),
        }

    def fingerprint(self) -> str:
        """Short stable id for the exact weights-and-precision combination."""
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class HuggingFaceGenerator(Generator):
    """A causal LM answering from a prompt the arm already built.

    The arms build their own context string and pass it as ``prompt``; this
    class does not assemble evidence itself. That keeps prompt construction
    where prompt parity is already enforced, rather than splitting it across
    two layers.
    """

    def __init__(
        self,
        spec: ModelSpec,
        config: Optional[GenerationConfig] = None,
        *,
        device_map: str = "auto",
    ) -> None:
        self.spec = spec
        self.config = config or GenerationConfig()
        self.device_map = device_map
        self._tokenizer = None
        self._model = None

    # -- loading ---------------------------------------------------------

    def _quantization_config(self):
        if self.spec.quantization is None:
            return None
        try:
            import torch
            from transformers import BitsAndBytesConfig
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise GeneratorError(
                "quantized loading needs transformers and bitsandbytes"
            ) from exc
        if self.spec.quantization == "int8":
            return BitsAndBytesConfig(load_in_8bit=True)
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=getattr(torch, self.spec.dtype),
        )

    def load(self):
        """Load the model. Called once; every arm then shares this object."""
        if self._model is not None:
            return self._model
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise GeneratorError(
                f"{self.spec.model_id} needs torch and transformers installed"
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.spec.model_id, revision=self.spec.revision)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        kwargs: dict[str, Any] = {
            "revision": self.spec.revision,
            "device_map": self.device_map,
        }
        quantization = self._quantization_config()
        if quantization is not None:
            kwargs["quantization_config"] = quantization
        else:
            import torch
            kwargs["torch_dtype"] = getattr(torch, self.spec.dtype)

        self._model = AutoModelForCausalLM.from_pretrained(
            self.spec.model_id, **kwargs)
        self._model.eval()
        return self._model

    # -- generation ------------------------------------------------------

    def generate(
        self,
        question: str,
        evidence: Sequence[Evidence] = (),
        *,
        prompt: Optional[str] = None,
    ) -> GenerationResult:
        import torch

        model = self.load()
        tokenizer = self._tokenizer
        text = prompt if prompt is not None else question

        # Instruction-tuned checkpoints expect their chat template; skipping
        # it changes the answer distribution of every arm equally but makes
        # the run a worse instance of the model being named.
        if getattr(tokenizer, "chat_template", None):
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False, add_generation_prompt=True,
            )

        encoded = tokenizer(text, return_tensors="pt")
        encoded = {k: v.to(model.device) for k, v in encoded.items()}
        prompt_tokens = int(encoded["input_ids"].shape[1])

        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = output[0][prompt_tokens:]
        answer = tokenizer.decode(generated, skip_special_tokens=True).strip()

        return GenerationResult(
            text=answer,
            metadata={
                "model_id": self.spec.model_id,
                "revision": self.spec.revision,
                "quantization": self.spec.quantization,
                "prompt_tokens": prompt_tokens,
                "generated_tokens": int(generated.shape[0]),
                "truncated": int(generated.shape[0]) >= self.config.max_new_tokens,
            },
        )


#: RAG²'s own generator. The revision is left unset deliberately: pinning it
#: requires reading the sha off the Hub, and a placeholder here would be a
#: fabricated provenance record. `docs/generator_contract.md` says where to
#: get it and where to record it.
RAG2_GENERATOR_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

#: Declared fallback if the gated licence cannot be obtained (ledger D-38).
FALLBACK_GENERATOR_ID = "Qwen/Qwen2.5-7B-Instruct"

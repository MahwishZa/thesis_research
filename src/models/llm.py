"""The research models and their clients: Claude through Anthropic, GPT-4o, Gemini and Llama through OpenRouter, and MedGemma 1.5 4B-it on this
computer (transformers; no Ollama, no server, no key, no cost). The same settings serve every system (LLM-only, standard RAG, ...)."""
import base64
import io
import json
import os
import re
import time
import urllib.error
import urllib.request

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS = {
    # Claude rejects temperature, top_p and top_k, so none is sent (the provider's default sampling is used).
    "claude-sonnet-5-5": {"api": "anthropic", "id": "claude-sonnet-5-5", "max_tokens": 1024,
                          "thinking": {"type": "between_tools"}, "effort": "low"},
    # A dated snapshot: the plain name "gpt-4o" moves to newer versions.
    "gpt-4o": {"api": "openrouter", "id": "openai/gpt-4o-2024-08-06", "route": ["openai"], "temperature": 0.0, "seed": 0,
               "max_tokens": 1024, "image_detail": "high"},
    # Hidden reasoning counts toward max_tokens: with 2048 one image question (of 10) was cut off before its answer letter
    # (finish: length, 2,044 tokens out; validation trial 2026-11-24), hence the larger limit. Only the tokens used are paid for.
    "gemini-3.8-flash": {"api": "openrouter", "id": "google/gemini-3.8-flash", "route": ["google-ai-studio"],
                         "reasoning": {"effort": "minimal"}, "temperature": 0.0, "seed": 0, "max_tokens": 8192},
    "llama-4-maverick": {"api": "openrouter", "id": "meta-llama/llama-4-maverick", "route": ["parasail/fp8"],
                         "temperature": 0.0, "seed": 0, "max_tokens": 1024},
    # MedGemma 1.5 4B-it, the successor of MedGemma 1 (its card: better medical text reasoning, a modest gain on 2D images, same 4B size). The
    # same model and the same revision ran on this laptop in the earlier version of the project (2026-11-07; 16-bit weights split GPU 3 GiB / CPU 10 GiB).
    # Runs on the researcher's laptop (researcher, 2026-11-26: no budget, no Ollama; RTX 2050 with 4 GB). It reads text and pictures, so one
    # backbone serves all three views. The 16-bit weights (about 8.6 GB) do not fit the card: accelerate keeps 3 GiB on the card and the rest
    # in the computer's memory (the setting that ran this model on this laptop in the earlier version of the project). No quantization.
    # Greedy decoding, so a rerun gives the same answers on the same hardware. The revision is pinned: a changed model would be another
    # model. The weights are gated: accept the terms on Hugging Face and have HF_TOKEN in .env. 512 tokens hold reasoning, answer and evidence.
    "medgemma-1.5-4b-it": {"api": "local", "id": "google/medgemma-1.5-4b-it", "revision": "91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b",
                       "free": True, "dtype": "bfloat16", "max_memory": {"gpu": "3GiB", "cpu": "10GiB"}, "do_sample": False,
                       "max_tokens": 512},
    # A TRIAL, not a choice (researcher, 2026-11-27: "first check Qwen3.5-2B"): a 2B model with a vision encoder, Apache 2.0, 4.55 GB in 16 bits, so
    # most of it fits the card. Gated Delta Net layers: without the packages flash-linear-attention and causal-conv1d, transformers uses a slow
    # plain-PyTorch path (◻ not known whether they build on Windows). Non-thinking mode is its default for 2B; kept greedy like the other model.
    "qwen3.5-2b": {"api": "local", "id": "Qwen/Qwen3.5-2B", "revision": "15852e8c16360a2fea060d615a32b45270f8a8fc",
                   "free": True, "dtype": "bfloat16", "max_memory": {"gpu": "3.4GiB", "cpu": "10GiB"}, "do_sample": False,
                   "max_tokens": 512, "chat_template_kwargs": {"enable_thinking": False}},
}
KEYS = {"anthropic": ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY"), "openrouter": ("OPENROUTER_API_KEY",)}
RETRY_CODES = {408, 409, 429, 500, 502, 503, 504, 529}
MODERATION = re.compile(r"moderation|flagged", re.I)


class ApiError(RuntimeError):
    """An error that retrying did not fix (wrong key, no credit, unknown model ...): the run stops and can be resumed."""


class Refused(RuntimeError):
    """The host's content moderation refused this one question; the run saves it as an invalid answer and goes on."""


# ---- keys -----------------------------------------------------------------------------------------------------------------
def api_key(api):
    for name in KEYS[api]:
        if os.environ.get(name, "").strip():
            return os.environ[name].strip()
    raise SystemExit(f"no API key: put the line {KEYS[api][-1]}=<your key> in the .env file next to README.md")


def image_b64(data, path):
    with open(os.path.join(data, path), "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


# ---- Claude (Anthropic) ---------------------------------------------------------------------------------------------------
def anthropic_params(cfg, system, text, image):
    content = []
    if image:
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image}})
    content.append({"type": "text", "text": text})
    return {"model": cfg["id"], "max_tokens": cfg["max_tokens"], "system": system,
            "messages": [{"role": "user", "content": content}], "thinking": cfg["thinking"],
            "output_config": {"effort": cfg["effort"]}}


def anthropic_reply(r):
    """The answer of a Claude response; a refusal keeps its category."""
    text = "".join(b.text for b in r.content if getattr(b, "type", None) == "text")
    finish = r.stop_reason
    details = getattr(r, "stop_details", None)
    if finish == "refusal" and details is not None:
        finish = f"refusal: {getattr(details, 'category', None)}"
    return {"text": text, "finish_reason": finish, "served_by": r.model,
            "tokens_in": r.usage.input_tokens, "tokens_out": r.usage.output_tokens}


class AnthropicClient:
    def __init__(self, cfg, key):
        import anthropic  # pip install anthropic (see requirements.txt)
        self.cfg, self.sdk = cfg, anthropic
        self.client = anthropic.Anthropic(api_key=key, max_retries=6, timeout=180.0)  # the SDK retries 408/409/429/5xx

    def ask(self, system, text, image):
        try:
            r = self.client.messages.create(**anthropic_params(self.cfg, system, text, image))
        except self.sdk.APIStatusError as e:
            raise ApiError(f"HTTP {e.status_code}: {e.message}") from None
        except self.sdk.APIConnectionError as e:
            raise ApiError(f"cannot reach the Anthropic API: {e}") from None
        return anthropic_reply(r)


# ---- GPT-4o, Gemini, Llama (OpenRouter) -----------------------------------------------------------------------------------
def openrouter_request(cfg, system, text, image):
    content = [{"type": "text", "text": text}]
    if image:
        picture = {"url": f"data:image/jpeg;base64,{image}"}
        if cfg.get("image_detail"):
            picture["detail"] = cfg["image_detail"]
        content.append({"type": "image_url", "image_url": picture})
    body = {"model": cfg["id"], "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
            "max_tokens": cfg["max_tokens"], "temperature": cfg["temperature"], "seed": cfg["seed"],
            "provider": {"order": cfg["route"], "allow_fallbacks": False}}    # only the pinned host may answer
    if cfg.get("reasoning"):
        body["reasoning"] = cfg["reasoning"]
    return OPENROUTER_URL, body


def openrouter_reply(r):
    if r.get("error") and not r.get("choices"):
        detail = json.dumps(r["error"])[:500]
        raise (Refused if MODERATION.search(detail) else ApiError)(f"error in the answer: {detail}")
    choice = (r.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    usage = r.get("usage") or {}
    return {"text": msg.get("content") or "", "finish_reason": "refusal" if msg.get("refusal") else choice.get("finish_reason"),
            "served_by": f"{r.get('model')}@{r.get('provider')}", "tokens_in": usage.get("prompt_tokens"),
            "tokens_out": usage.get("completion_tokens")}


def post_json(url, head, body, retries=6, timeout=180, log=None):
    """POST with retries for busy or failing servers (honouring Retry-After). A refusal by the host's content moderation raises
    Refused; any other error raises ApiError at once."""
    data = json.dumps(body).encode("utf-8")
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, data=data, headers=head, method="POST")
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:500]
            if e.code == 403 and MODERATION.search(detail):
                raise Refused(f"refused by the host's content moderation (HTTP 403: {detail})") from None
            if e.code not in RETRY_CODES or attempt == retries:
                raise ApiError(f"HTTP {e.code}: {detail}") from None
            wait = float(e.headers.get("Retry-After") or min(60, 2 ** (attempt + 1)))
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == retries:
                raise ApiError(f"network error: {e}") from None
            wait = min(60, 2 ** (attempt + 1))
        if log:
            log.warning(f"retry {attempt + 1}/{retries} in {wait:.0f} s")
        time.sleep(wait)


class OpenRouterClient:
    def __init__(self, cfg, key, log=None):
        self.cfg, self.key, self.log = cfg, key, log

    def ask(self, system, text, image):
        url, body = openrouter_request(self.cfg, system, text, image)
        head = {"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"}
        return openrouter_reply(post_json(url, head, body, log=self.log))


def require_free_or_allowed(name, allow_paid):
    """A model that costs credit runs only when --allow-paid is given (the researcher has no budget left, 2026-11-26)."""
    if not MODELS[name].get("free") and not allow_paid:
        raise SystemExit(f"{name} costs credit. Nothing was sent. Add --allow-paid only if you mean to pay (a model entry marked free needs no flag).")


# ---- a model on this computer (transformers) --------------------------------------------------------------------------------------
def load_local(cfg):
    """(processor, model) of a local model at its pinned revision. accelerate keeps cfg["max_memory"]["gpu"] on the graphics card and puts the
    rest in the computer's memory (up to cfg["max_memory"]["cpu"]), from where each block is moved to the card when it is needed.
    The weights (about 8.6 GB) are downloaded from Hugging Face on first use. Stops, before loading anything, when torch has no graphics card."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    if not torch.cuda.is_available():
        # on the processor alone these models take minutes per question (found on 2026-11-27: torch 2.15.0+cpu, 16-bit maths at 1 GFLOPS), so nothing is loaded
        raise SystemExit(f"torch {torch.__version__} cannot use a graphics card (a '+cpu' version is the CPU-only build), and {cfg['id']} on the processor alone takes "
                         "minutes per question. Install the CUDA build first (see requirements.txt), check with: "
                         "python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\". Nothing was loaded.")
    processor = AutoProcessor.from_pretrained(cfg["id"], revision=cfg["revision"])
    kwargs = {"revision": cfg["revision"], "dtype": getattr(torch, cfg["dtype"]), "device_map": "auto",
              "max_memory": {0: cfg["max_memory"]["gpu"], "cpu": cfg["max_memory"]["cpu"]}}
    return processor, AutoModelForImageTextToText.from_pretrained(cfg["id"], **kwargs).eval()


def package_version(name):
    from importlib import metadata
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "unknown"


def decode_image(data):
    """The picture of a base64 JPEG string, as the RGB image the processor takes."""
    from PIL import Image
    return Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB")


class LocalClient:
    """The same ask(system, text, image) as the other clients, answered by a model on this computer. Greedy decoding."""

    def __init__(self, cfg, load=load_local):
        self.cfg = cfg
        self.processor, self.model = load(cfg)
        # the revision and the two libraries are written into every answer record: a rerun can be compared with it
        self.served_by = (f"{cfg['id']}@{cfg['revision']} (transformers {package_version('transformers')}, torch {package_version('torch')})")

    def build_inputs(self, system, text, image):
        """The model's input tensors for one question: the chat turns (picture first), tokenized, on the model's device."""
        content = ([{"type": "image", "image": decode_image(image)}] if image else []) + [{"type": "text", "text": text}]
        messages = [{"role": "system", "content": [{"type": "text", "text": system}]}, {"role": "user", "content": content}]
        return self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
                                                  **self.cfg.get("chat_template_kwargs", {})).to(self.model.device, dtype=self.model.dtype)

    def generate(self, inputs, max_new_tokens):
        """The new tokens (a list) for prepared inputs; greedy."""
        n_in = inputs["input_ids"].shape[-1]
        try:
            # temperature, top_p and top_k are cleared so that the model's own sampling defaults raise no warning: the decoding is greedy
            out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=self.cfg["do_sample"],
                                      temperature=None, top_p=None, top_k=None)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():                    # the run stops and can be resumed; lowering max_memory gpu is the fix
                raise ApiError(f"the graphics card ran out of memory ({e}); lower max_memory gpu of {self.cfg['id']} in MODELS and run the same command again") from None
            raise
        return out[0][n_in:]

    def ask(self, system, text, image):
        inputs = self.build_inputs(system, text, image)
        new = self.generate(inputs, self.cfg["max_tokens"])
        return {"text": self.processor.decode(new, skip_special_tokens=True).strip(),
                "finish_reason": "length" if len(new) >= self.cfg["max_tokens"] else "stop", "served_by": self.served_by,
                "tokens_in": int(inputs["input_ids"].shape[-1]), "tokens_out": len(new)}


def make_client(name, log=None):
    cfg = MODELS[name]
    if cfg["api"] == "local":
        return LocalClient(cfg)
    key = api_key(cfg["api"])
    return AnthropicClient(cfg, key) if cfg["api"] == "anthropic" else OpenRouterClient(cfg, key, log)

"""The four research models and their clients: Claude through Anthropic, GPT-4o, Gemini and Llama through OpenRouter.
The same settings serve every system (LLM-only, standard RAG, ...)."""
import base64
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
    # The free backbone of the RAG systems (researcher, 2026-11-26; the budget is spent). Open weights (Apache 2.0), takes images, one provider
    # (Google AI Studio; OpenRouter page, 2026-11-26). Thinking is left at the endpoint's default until the probe shows what it returns; hidden
    # reasoning counts toward max_tokens, hence the large limit. "free": no credit is used, so it needs no --allow-paid.
    "gemma-4-31b": {"api": "openrouter", "id": "google/gemma-4-31b-it:free", "route": ["google-ai-studio"], "temperature": 0.0, "seed": 0,
                    "max_tokens": 8192, "free": True},
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
        raise SystemExit(f"{name} costs credit. Nothing was sent. Use the free model (gemma-4-31b), or add --allow-paid if you mean to pay.")


def make_client(name, log=None):
    cfg = MODELS[name]
    key = api_key(cfg["api"])
    return AnthropicClient(cfg, key) if cfg["api"] == "anthropic" else OpenRouterClient(cfg, key, log)

"""The model adapters: one class per API, each answering one prompt, with or without images, in the same form."""

from __future__ import annotations

import base64
import os
import re
from pathlib import Path
from urllib.parse import urlparse

# ==============================================================================================================================
# Model adapters
#
# The LLM adapters: one class per API, each answering one prompt, with or without images, in the same form.
#
#     anthropic          Claude through the Anthropic API
#     openai_compatible  GPT-4o, Gemini and Llama at OpenRouter, and any host that speaks the same chat API (for example OpenAI
#                        or Hugging Face)
#
# Every adapter has ``describe()`` (what is asked: provider, model id, settings) and ``predict(images, prompt)``, where ``images`` is
# a list of (bytes, media type) pairs, empty for a text case (the case is inside the prompt). The
# answer is a dict with ``text`` (or None), ``finish_reason`` ("STOP" for a normal end, "MAX_TOKENS" when the answer was cut off at
# the output limit; any other value passes through), ``model_version`` (as the API reports it), ``response_id`` and ``usage``
# (token counts). A failure raises TransientError (worth retrying), QuotaExhausted (credit or a per-day quota is used up),
# Refused (the host's content moderation refuses this one request; the next may be fine) or FatalError (retrying cannot fix it; the
# message says what to do). An adapter also names the ``packages`` whose versions the run
# card records and says whether its model id is ``pinned`` to one version; ``loaded_note`` and ``probe_note`` tell the setup check
# how to describe it.
#
# Every image or case is sent to the provider's servers, so the dataset's terms must allow it. The SDKs' own retries are off,
# because a run retries failures itself, waits longer each time and counts them. API keys are read from .env (git-ignored), never
# shown.
# ==============================================================================================================================


TIMEOUT_SECONDS = 120  # the longest wait for one answer; a stalled connection must never wait forever
REFUSED = "REFUSED"  # the finish reason saved for a question the host refused (an invalid answer)
IMAGE_TYPES = ("image/jpeg", "image/png", "image/gif", "image/webp")  # what the Anthropic and OpenAI-style APIs accept


class TransientError(Exception):
    """A failure worth retrying later (rate limit, server overload, network). The image is left unanswered, not scored."""


class QuotaExhausted(Exception):
    """Credit or a per-day quota is used up, so retrying within minutes cannot help. Answers already saved are kept."""


class Refused(Exception):
    """The host refused this one request because of what it holds (its content moderation flagged a photo), so the same request
    would be refused again. Other requests are not affected: the run saves the question as an invalid answer and goes on."""

    def __init__(self, message: str):
        super().__init__(message)
        self.attempt = 1  # set by the run: how many requests it took


class FatalError(Exception):
    """A failure that retrying cannot fix (wrong key, unavailable model, refused setting). The message says what to do."""

    def __init__(self, message: str, code: int | None = None):
        super().__init__(message)
        self.code = code


def load_env(path: str | Path = ".env") -> None:
    """Puts the NAME=value lines of a local .env file (git-ignored; it holds the API keys) into the environment.

    A variable that is already set keeps its value, so a key set in the terminal wins. Blank lines, # comments, lines
    without "=" and empty values are skipped. A missing file is not an error.
    """
    path = Path(path)
    if not path.is_file():
        return
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path} must be saved as plain UTF-8 text; the editor saved it in another encoding") from exc
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = (part.strip() for part in line.split("=", 1))
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if name and value:
            os.environ.setdefault(name, value)


def load(provider: str, settings: dict):
    """The adapter named by ``provider`` in a config's [model] section, set up with that section's ``settings``."""
    adapters = {"anthropic": AnthropicModel, "openai_compatible": OpenAICompatibleModel}
    if provider not in adapters:
        raise ValueError(f"unknown model provider: {provider!r} (adapters: {', '.join(adapters)})")
    return adapters[provider](settings)


def _require(settings: dict, keys: tuple[str, ...], adapter: str) -> None:
    missing = [key for key in keys if key not in settings]
    if missing:
        raise ValueError(f"[model] is missing {', '.join(missing)} (the {adapter} adapter needs {', '.join(keys)})")
    if not isinstance(settings["id"], str) or not settings["id"]:
        raise ValueError("[model] id must be non-empty text")


def _whole(settings: dict, key: str) -> int:
    value = settings[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"[model] {key} must be a whole number of at least 1 (got {value!r})")
    return value


# ---------------------------------------------------------------------------------------------------------------------------------
# Claude (Anthropic)
# The key is ANTHROPIC_API_KEY or, if that is not set, CLAUDE_API_KEY. The current Claude models reject temperature, top_p and
# top_k, so none is sent and the provider's default sampling is used; the run card says so. No refusal fallback is requested:
# every answer comes from the model named in the config, and a refusal is recorded as finish reason REFUSAL.

ANTHROPIC_KEYS = ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY")
ANTHROPIC_THINKING = ("between_tools", "adaptive")  # between_tools = no reasoning for a request without tools; "disabled" is refused
ANTHROPIC_EFFORT = ("low", "medium", "high", "xhigh", "max")
ANTHROPIC_FINISH = {"end_turn": "STOP", "stop_sequence": "STOP", "max_tokens": "MAX_TOKENS", "refusal": "REFUSAL"}


class AnthropicModel:
    packages = ("anthropic", "pillow")
    probe_note = "it is one request to Anthropic and costs a fraction of a cent of your API credit"

    def __init__(self, settings: dict):
        _require(settings, ("id", "max_tokens", "thinking", "effort"), "anthropic")
        self.model_id, self.max_tokens = settings["id"], _whole(settings, "max_tokens")
        self.thinking, self.effort = settings["thinking"], settings["effort"]
        if self.thinking not in ANTHROPIC_THINKING:
            raise ValueError(f"[model] thinking must be one of {list(ANTHROPIC_THINKING)} (got {self.thinking!r})")
        if self.effort not in ANTHROPIC_EFFORT:
            raise ValueError(f"[model] effort must be one of {list(ANTHROPIC_EFFORT)} (got {self.effort!r})")
        self.pinned = not self.model_id.endswith("-latest")  # a Claude model id without a date is a fixed version; "-latest" moves
        try:
            import anthropic
        except ImportError as exc:
            raise ValueError("The anthropic package is not installed: run pip install -r requirements.txt") from exc
        key = next((os.environ[name].strip() for name in ANTHROPIC_KEYS if os.environ.get(name, "").strip()), None)
        if key is None:
            raise ValueError("no Anthropic API key found: put the line CLAUDE_API_KEY=<your key> (or ANTHROPIC_API_KEY=<your key>) "
                             "in the .env file next to README.md")
        self.sdk = anthropic
        self.client = anthropic.Anthropic(api_key=key, max_retries=0, timeout=TIMEOUT_SECONDS)
        self.loaded_note = f"{self.model_id}, API key found (not shown); each answer records the model version Anthropic reports"

    def describe(self) -> dict:
        return {"provider": "anthropic", "model_id": self.model_id, "sdk": "anthropic",
                "sdk_version": getattr(self.sdk, "__version__", None),
                "generation_config": {"max_tokens": self.max_tokens, "thinking": self.thinking, "effort": self.effort,
                                      "sampling": "provider default (temperature, top_p and top_k are not sent)"}}

    def _failure(self, exc) -> Exception:
        """The run's own kind of error for an API refusal: retry later, stop until credit is added, or fix a setting."""
        status, kind, code = exc.status_code, getattr(exc, "type", None), getattr(exc, "error_code", None)
        detail = getattr(exc, "api_message", None) or str(exc)
        if kind == "billing_error" or code == "enforced_spend_limit_reached":
            return QuotaExhausted(f"Anthropic cannot charge for more requests ({detail}). Add credit or raise the spend limit in the "
                                  "Anthropic console, then run the same command again.")
        if status in (408, 409, 429) or status >= 500:
            return TransientError(f"HTTP {status} {kind or ''}: {detail}".replace("  ", " "))
        if status == 401:
            return FatalError(f"Anthropic refused the API key (HTTP 401: {detail}). Check the key in .env.", status)
        if status == 403:
            return FatalError(f"Anthropic refused the request (HTTP 403: {detail}). The key may lack permission, or the account's country "
                              "or organisation may not be allowed to use the API.", status)
        if status == 404:
            return FatalError(f"Anthropic does not know the model {self.model_id!r} (HTTP 404: {detail}). Check the id in the config.", status)
        return FatalError(f"Anthropic refused the request (HTTP {status} {kind or ''}: {detail}). Fix the setting the message names.",
                          status)

    def predict(self, images: list[tuple[bytes, str]], prompt: str) -> dict:
        """One answer to the prompt, with the images (none, one or several) before it, in their order."""
        content = []
        for image, mime_type in images:
            if mime_type not in IMAGE_TYPES:
                raise FatalError(f"Claude accepts JPEG, PNG, GIF and WebP images, not {mime_type}")
            content.append({"type": "image", "source": {"type": "base64", "media_type": mime_type,
                                                        "data": base64.standard_b64encode(image).decode("ascii")}})
        content.append({"type": "text", "text": prompt})
        try:
            response = self.client.messages.create(model=self.model_id, max_tokens=self.max_tokens,
                                                   thinking={"type": self.thinking}, output_config={"effort": self.effort},
                                                   messages=[{"role": "user", "content": content}])
        except self.sdk.APIStatusError as exc:
            raise self._failure(exc) from exc
        except self.sdk.APIConnectionError as exc:  # also a timeout
            raise TransientError(f"cannot reach the Anthropic API ({type(exc).__name__}: {exc})") from exc
        except self.sdk.APIError as exc:
            raise FatalError(f"the Anthropic SDK failed ({type(exc).__name__}: {exc})") from exc
        text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text").strip() or None
        usage, stop = response.usage, response.stop_reason
        return {"text": text, "finish_reason": ANTHROPIC_FINISH.get(stop, str(stop).upper()), "model_version": response.model,
                "response_id": response.id,
                "usage": {"prompt_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
                          "total_tokens": usage.input_tokens + usage.output_tokens}}


# ---------------------------------------------------------------------------------------------------------------------------------
# OpenAI-style chat API: GPT-4o at OpenAI, and any host that speaks the same API (for example Llama at OpenRouter or Hugging Face)
# The host and the key come from the config: base_url (absent = OpenAI) and api_key_env, the name of the variable in .env that
# holds the key (default OPENAI_API_KEY). Decoding is as the base paper's: temperature 0 and a fixed seed; the host's seed is
# best effort, so a repeated run can still differ slightly. A host may serve one model from several back ends; extra_body passes
# a host-specific choice through (for OpenRouter, provider = {order = ["<host>"], allow_fallbacks = false} keeps every answer on
# one back end), and the back end's name is recorded with every answer when the host reports it.

OPENAI_DEFAULT_KEY = "OPENAI_API_KEY"
OPENAI_DETAIL = ("low", "high", "auto")  # how closely OpenAI looks at the image (more detail = more tokens)
OPENAI_FINISH = {"stop": "STOP", "length": "MAX_TOKENS", "content_filter": "CONTENT_FILTER"}
DATED = re.compile(r"\d{4}-\d{2}-\d{2}$")  # a model id ending in a date is a fixed snapshot; one without may move
MODERATION = re.compile(r"moderation|flagged", re.IGNORECASE)  # a refusal of this request's content, not of the key
DAILY = re.compile(r"per[- ]day|daily", re.IGNORECASE)  # waiting minutes cannot help with a per-day limit


def _provider_detail(exc) -> str:
    """What OpenRouter adds to a bare "Provider returned error": the provider's name and its own message (error.metadata), so that a
    refusal says why. Empty when the host gives nothing more."""
    body = getattr(exc, "body", None)
    inner = body.get("error", body) if isinstance(body, dict) else None
    meta = inner.get("metadata") if isinstance(inner, dict) else None
    if not isinstance(meta, dict):
        return ""
    parts = [str(meta[key])[:500] for key in ("provider_name", "raw") if meta.get(key)]
    return f" [{': '.join(parts)}]" if parts else ""


class OpenAICompatibleModel:
    packages = ("openai", "pillow")
    probe_note = "it is one request: a fraction of a cent of credit, or one of a free model's daily requests"

    def __init__(self, settings: dict):
        _require(settings, ("id", "temperature", "seed", "max_tokens"), "openai_compatible")
        self.model_id = settings["id"]
        self.temperature = settings["temperature"]
        if isinstance(self.temperature, bool) or not isinstance(self.temperature, (int, float)) or not 0 <= self.temperature <= 2:
            raise ValueError(f"[model] temperature must be a number from 0 to 2 (got {self.temperature!r})")
        self.seed = settings["seed"]
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError(f"[model] seed must be a whole number (got {self.seed!r})")
        self.max_tokens = _whole(settings, "max_tokens")
        self.base_url = settings.get("base_url")
        if self.base_url is not None and (not isinstance(self.base_url, str) or not self.base_url.startswith(("http://", "https://"))):
            raise ValueError(f"[model] base_url must be a web address starting with https:// (got {self.base_url!r})")
        self.key_name = settings.get("api_key_env", OPENAI_DEFAULT_KEY)
        if not isinstance(self.key_name, str) or not self.key_name:
            raise ValueError("[model] api_key_env must be the name of a variable in .env")
        self.image_detail = settings.get("image_detail")
        if self.image_detail is not None and self.image_detail not in OPENAI_DETAIL:
            raise ValueError(f"[model] image_detail must be one of {list(OPENAI_DETAIL)} (got {self.image_detail!r})")
        self.extra_body = settings.get("extra_body")
        if self.extra_body is not None and not isinstance(self.extra_body, dict):
            raise ValueError("[model] extra_body must be a table")
        self.pinned = settings.get("pinned", bool(DATED.search(self.model_id)))  # a test run needs a fixed version
        if not isinstance(self.pinned, bool):
            raise ValueError("[model] pinned must be true or false")
        try:
            import openai
        except ImportError as exc:
            raise ValueError("The openai package is not installed: run pip install -r requirements.txt") from exc
        key = os.environ.get(self.key_name, "").strip()
        if not key:
            raise ValueError(f"no API key found: put the line {self.key_name}=<your key> in the .env file next to README.md")
        self.sdk = openai
        self.client = openai.OpenAI(api_key=key, base_url=self.base_url, max_retries=0, timeout=TIMEOUT_SECONDS)
        host = urlparse(self.base_url).netloc if self.base_url else "api.openai.com"
        self.loaded_note = f"{self.model_id} at {host}, API key {self.key_name} found (not shown); each answer records the model name the host reports"

    def describe(self) -> dict:
        return {"provider": "openai_compatible", "model_id": self.model_id, "base_url": str(self.client.base_url), "sdk": "openai",
                "sdk_version": getattr(self.sdk, "__version__", None),
                "generation_config": {"temperature": self.temperature, "seed": self.seed, "max_tokens": self.max_tokens,
                                      "image_detail": self.image_detail, "extra_body": self.extra_body}}

    def _failure(self, exc) -> Exception:
        """The run's own kind of error for a refusal by the host: retry later, stop until credit or quota returns, or fix a setting."""
        status, code = exc.status_code, getattr(exc, "code", None)
        detail = (getattr(exc, "message", None) or str(exc)) + _provider_detail(exc)
        if code == "insufficient_quota" or status == 402 or (status == 429 and DAILY.search(detail)):
            return QuotaExhausted(f"The host cannot serve more requests ({detail}). Add credit, wait for the daily limit to reset, "
                                  "or use another model; then run the same command again.")
        if status in (408, 409, 429) or status >= 500:
            return TransientError(f"HTTP {status}: {detail}")
        if status == 401:
            return FatalError(f"The host refused the API key (HTTP 401: {detail}). Check {self.key_name} in .env.", status)
        if status == 403 and MODERATION.search(detail):
            return Refused(f"refused by the host's content moderation (HTTP 403: {detail[:300]})")
        if status == 403:
            return FatalError(f"The host refused the request (HTTP 403: {detail}). The key may lack permission, or the account's "
                              "country or organisation may not be allowed to use this model.", status)
        if status == 404:
            return FatalError(f"The host does not know the model {self.model_id!r} (HTTP 404: {detail}). Check the id and base_url "
                              "in the config.", status)
        return FatalError(f"The host refused the request (HTTP {status}: {detail}). Fix the setting the message names.", status)

    def predict(self, images: list[tuple[bytes, str]], prompt: str) -> dict:
        """One answer to the prompt, with the images (none, one or several) before it, in their order."""
        content = []
        for image, mime_type in images:
            if mime_type not in IMAGE_TYPES:
                raise FatalError(f"This adapter accepts JPEG, PNG, GIF and WebP images, not {mime_type}")
            picture = {"url": f"data:{mime_type};base64,{base64.standard_b64encode(image).decode('ascii')}"}
            if self.image_detail:
                picture["detail"] = self.image_detail
            content.append({"type": "image_url", "image_url": picture})
        content.append({"type": "text", "text": prompt})
        request = {"model": self.model_id, "temperature": self.temperature, "seed": self.seed, "max_tokens": self.max_tokens,
                   "messages": [{"role": "user", "content": content}]}
        if self.extra_body:
            request["extra_body"] = self.extra_body
        try:
            response = self.client.chat.completions.create(**request)
        except self.sdk.APIStatusError as exc:
            raise self._failure(exc) from exc
        except self.sdk.APIConnectionError as exc:  # also a timeout
            raise TransientError(f"cannot reach the host ({type(exc).__name__}: {exc})") from exc
        except self.sdk.APIError as exc:
            raise FatalError(f"the OpenAI SDK failed ({type(exc).__name__}: {exc})") from exc
        choice = response.choices[0] if response.choices else None
        message = choice.message if choice else None
        text = ((message.content or "").strip() or None) if message else None
        finish = OPENAI_FINISH.get(choice.finish_reason, str(choice.finish_reason).upper()) if choice else "NO_CHOICE"
        if message and getattr(message, "refusal", None) and not text:
            finish = "REFUSAL"
        backend = (getattr(response, "model_extra", None) or {}).get("provider")  # OpenRouter says which back end answered
        usage = response.usage
        return {"text": text, "finish_reason": finish, "model_version": f"{response.model}@{backend}" if backend else response.model,
                "response_id": response.id,
                "usage": {"prompt_tokens": getattr(usage, "prompt_tokens", None), "output_tokens": getattr(usage, "completion_tokens", None),
                          "total_tokens": getattr(usage, "total_tokens", None)}}

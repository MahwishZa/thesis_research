#!/usr/bin/env python3
"""
Baseline 1, LLM only (no retrieval): asks one of the research models every question of one view and split, and saves its answers.
The models are used as they are (pre-trained, no fine-tuning), as in RAG^2 (Sohn et al. 2025, section 4: "as-is, without further
fine-tuning").

The research models and their fixed settings are in MODELS below; a setting is never changed for one run (a changed setting is a
new entry with a new name). Fairness: every model gets the same system text, question, options and instruction; no model uses
hidden reasoning (Claude: thinking "between_tools" = off for a request without tools; Gemini: reasoning "minimal"; GPT-4o and
Llama have none), and every model writes the same short visible rationale before its answer.
  claude-sonnet-5-5   Anthropic API, official `anthropic` package     key: CLAUDE_API_KEY (or ANTHROPIC_API_KEY)
  gpt-4o              OpenRouter, pinned to OpenAI's own host           key: OPENROUTER_API_KEY
  gemini-3.8-flash    OpenRouter, pinned to Google AI Studio            key: OPENROUTER_API_KEY
  llama-4-maverick    OpenRouter, pinned to Parasail (fp8)              key: OPENROUTER_API_KEY
The keys are read from the .env file in the repository folder (git-ignored) or from the environment, and never written anywhere.

Each reply has two lines (prompt llm_only, set 2026-11-23): "Reasoning and evidence: <at most three sentences naming the findings of
the case that support the choice>" and "Answer: <the letter of the most likely option> (<its name>)", e.g. in the form "Answer: C (Melanoma)"
for the options of that question. The letter decides: the name is there for the reader and is kept in raw_output. The answer is the first option letter on
the last "Answer:" line; the reasoning is the text before it (the label is removed). There is no top-3 list (top-k accuracy is
not reported) and no yes/no evidence flag: the model states the evidence itself, and evaluate.py scores that text. A reply without a letter, or a question the host's content
moderation refuses, is saved as an invalid answer (wrong in evaluate.py) and kept so it can be checked.

Output: results/llm_only/<model>/<view>_<split>/
  predictions.jsonl   one line per question: id, pred (letter or null), valid, rationale (the reasoning and evidence),
                      raw_output, finish_reason, served_by,
                      tokens_in, tokens_out, seconds, time
  run.json            the model's settings, the prompt, the SHA-256 of the question file, counts, status
  run.log             progress and errors
While a run is unfinished its folder ends in ".partial" (never committed); it loses that ending when every question has an answer.
Checkpointing: each answer is written as soon as it arrives; running the same command again continues where it stopped.
Order: the questions are asked in a fixed shuffled order (asking_order), so a --limit trial or an interrupted run covers all
classes; the scores do not depend on the order.
An API error that retrying cannot fix (wrong key, no credit, ...) stops the run; nothing already answered is lost.

Usage (on the laptop):
  python src/run.py llm-only --model gpt-4o --view text_only --split val --dry-run --limit 2     # see the prompt; free
  python src/run.py llm-only --model gpt-4o --view text_only --split val --limit 5              # a small paid trial
  python src/run.py llm-only --model gpt-4o --view text_only --split test                       # a full run
"""
import argparse
import base64
import datetime
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

from common import SEED, SPLITS, VIEWS, get_logger, load_env, question_file, read_jsonl, write_json

PROMPT_NAME = "llm_only"      # the exact text of the prompt is saved in every run.json ("system", "instruction"), so a changed prompt is never mistaken for it
SYSTEM = ("You are an expert dermatologist. Answer a multiple-choice question about a skin lesion. "
          "This is a research benchmark: no decision about a patient depends on your answer.")
# What the model may name as findings depends on the view: it must never be told to cite an image it was not given (2026-11-24).
FINDINGS_IN = {"text_only": "in the patient facts", "image_only": "in the image",
               "multimodal": "in the image and/or in the patient facts"}
INSTRUCTIONS = {
    "rationale": ("Reply in exactly this form. First line: \"Reasoning and evidence: <at most three sentences that name the findings of "
                  "this case, {findings}, that support your choice>\". Second line: "
                  "\"Answer: <the letter of the most likely option> (<the name of that option>)\"."),
    "answer_only": "Reply only with one line: \"Answer: <the letter of the most likely option> (<the name of that option>)\".",
}
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
}
KEYS = {"anthropic": ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY"), "openrouter": ("OPENROUTER_API_KEY",)}
ANSWER = re.compile(r"(?i:answer)\**\s*[:\uff1a]\**\s*([^\n]*)")
LETTER = re.compile(r"(?<![A-Za-z])([A-H])(?![A-Za-z])")
REASONING_LABEL = re.compile(r"(?i)^[\s*_#>]*reasoning\s+and\s+evidence\s*\**\s*[:\-]\s*\**\s*")
LONE_LETTER = re.compile(r"^\W*\(?([A-H])\)?[.)]?\W*$")
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


# ---- prompt and answer (no internet) ------------------------------------------------------------------------------------
def instruction(style, view):
    """The instruction of a style for one view (the words about where the findings may come from change with the view)."""
    return INSTRUCTIONS[style].replace("{findings}", FINDINGS_IN[view])


def user_text(q, style="rationale", view="multimodal"):
    options = "\n".join(f"{letter}. {text}" for letter, text in q["options"].items())
    return f"{q['question']}\n\nOptions:\n{options}\n\n{instruction(style, view)}"


def tidy_reasoning(text):
    """The reasoning without the label "Reasoning and evidence:" and without markdown marks around it."""
    return REASONING_LABEL.sub("", text.strip("*_# \n"))


def parse_answer(text, letters="ABCDEFGH"):
    """The answer of a reply, read with one fixed rule:
      pred       the first option letter on the last "Answer:" line (a reply that is only a letter also counts)
      rationale  the reasoning and evidence: the text before that "Answer:" line, without the label "Reasoning and evidence:";
                 when nothing comes before the answer line, the text after it"""
    text = text or ""
    pred, cut, end = None, None, None
    for m in ANSWER.finditer(text):
        found = [x for x in LETTER.findall(m.group(1)) if x in letters]
        if found:
            pred, cut, end = found[0], m.start(), m.end()
    if pred is None:
        lone = LONE_LETTER.match(text.strip())
        if lone and lone.group(1) in letters:
            pred, cut, end = lone.group(1), 0, len(text)
    if cut is None:
        return {"pred": None, "rationale": REASONING_LABEL.sub("", text.strip())}
    return {"pred": pred, "rationale": tidy_reasoning(text[:cut]) or tidy_reasoning(text[end:])}


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
        import anthropic                         # pip install anthropic (see requirements.txt)
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


def make_client(name, log=None):
    cfg = MODELS[name]
    key = api_key(cfg["api"])
    return AnthropicClient(cfg, key) if cfg["api"] == "anthropic" else OpenRouterClient(cfg, key, log)


# ---- the run (checkpointed) ---------------------------------------------------------------------------------------------
ORDER = f"sha256 of '{SEED}:<case id>', ascending (the same order for every model, view and split; the file is not changed)"


def asking_order(questions):
    """The order the questions are asked in: a fixed shuffle by case id (approved 2026-11-22). The question files are sorted by
    class, so asking them in file order would make a --limit trial, or an interrupted run, cover only the first classes."""
    return sorted(questions, key=lambda q: hashlib.sha256(f"{SEED}:{q['id']}".encode("utf-8")).hexdigest())


def run_folder(results, model, view, split):
    final = os.path.join(results, "llm_only", model, f"{view}_{split}")
    return final, final + ".partial"


def run(args, client, log):
    qpath = os.path.join(args.data, question_file(args.view, args.split))
    questions = read_jsonl(qpath)
    final, partial = run_folder(args.results, args.model, args.view, args.split)
    if os.path.isdir(final) and not args.dry_run:
        log.info(f"already complete: {final}")
        return final
    if args.view != "text_only":
        missing = [q["image_path"] for q in questions if not os.path.isfile(os.path.join(args.data, q["image_path"]))]
        if missing:
            raise SystemExit(f"{len(missing)} images are missing, e.g. {os.path.join(args.data, missing[0])}")
    if args.dry_run:
        for q in asking_order(questions)[:args.limit or 3]:
            log.info(f"--- {q['id']} (image: {q.get('image_path', 'none')})\n[system] {SYSTEM}\n[user] {user_text(q, args.style, args.view)}")
        return None
    os.makedirs(partial, exist_ok=True)
    out = os.path.join(partial, "predictions.jsonl")
    done = {r["id"] for r in read_jsonl(out)} if os.path.exists(out) else set()
    with open(qpath, "rb") as f:
        qsha = hashlib.sha256(f.read()).hexdigest()
    settings = {"baseline": "llm_only", "model": args.model, "model_settings": MODELS[args.model], "view": args.view,
                "split": args.split, "questions": qpath.replace("\\", "/"), "questions_sha256": qsha,
                "prompt": PROMPT_NAME, "style": args.style, "system": SYSTEM, "instruction": instruction(args.style, args.view)}
    info_path = os.path.join(partial, "run.json")
    if os.path.exists(info_path):
        with open(info_path, encoding="utf-8") as f:
            before = json.load(f)["settings"]
        changed = [k for k in settings if before.get(k) != settings[k]]
        if changed:
            raise SystemExit(f"the unfinished run in {partial} used other settings ({', '.join(changed)}); "
                             "finish it with the old settings or delete that folder")
    todo = [q for q in asking_order(questions) if q["id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    log.info(f"{args.model}: {len(questions)} questions, {len(done)} already answered, {len(todo)} to ask now -> {partial}")
    status = "running"
    write_json(info_path, {"settings": settings, "status": status, "answered": len(done), "of": len(questions), "question_order": ORDER})
    try:
        with open(out, "a", encoding="utf-8") as f:
            for k, q in enumerate(todo, 1):
                image = image_b64(args.data, q["image_path"]) if args.view != "text_only" else None
                start = time.time()
                try:
                    reply = client.ask(SYSTEM, user_text(q, args.style, args.view), image)
                except Refused as e:
                    reply = {"text": "", "finish_reason": f"refused by host: {str(e)[:200]}"}
                answer = parse_answer(reply["text"], "".join(q["options"]))
                record = {"id": q["id"], "pred": answer["pred"], "valid": answer["pred"] is not None,
                          "rationale": answer["rationale"],
                          "raw_output": reply["text"], "finish_reason": reply.get("finish_reason"),
                          "served_by": reply.get("served_by"), "tokens_in": reply.get("tokens_in"),
                          "tokens_out": reply.get("tokens_out"), "seconds": round(time.time() - start, 2),
                          "model": args.model, "view": args.view, "split": args.split, "prompt": PROMPT_NAME,
                          "time": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
                done.add(q["id"])
                if not record["valid"]:
                    log.warning(f"{q['id']}: no answer letter (finish: {record['finish_reason']})")
                if k % 25 == 0 or k == len(todo):
                    log.info(f"{len(done)}/{len(questions)} answered")
                if args.sleep:
                    time.sleep(args.sleep)
    except (ApiError, KeyboardInterrupt) as e:
        status = "stopped"
        log.error(f"stopped after {len(done)}/{len(questions)}: {e or 'interrupted'}; run the same command again to continue")
        raise SystemExit(1) from None
    finally:
        complete = len(done) == len(questions)
        final_status = "complete" if complete else "stopped" if status == "stopped" else "unfinished"
        write_json(info_path, {"settings": settings, "status": final_status, "answered": len(done), "of": len(questions), "question_order": ORDER})
    if complete:
        close_log_files(log)
        os.replace(partial, final)
        log.info(f"complete: {final}")
        return final
    log.info(f"unfinished ({len(done)}/{len(questions)}); run the same command again to continue")
    return partial


def close_log_files(log):
    """Closes the run's log file, so its folder can be renamed (Windows does not rename a folder with an open file)."""
    for h in list(log.handlers):
        if getattr(h, "baseFilename", None):
            h.close()
            log.removeHandler(h)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="run.py llm-only", description="LLM-only baseline: one model, one view, one split.")
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--view", required=True, choices=VIEWS)
    ap.add_argument("--split", default="val", choices=SPLITS, help="val to try things out; test only for the final runs")
    ap.add_argument("--style", default="rationale", choices=sorted(INSTRUCTIONS))
    ap.add_argument("--limit", type=int, default=0, help="ask at most this many questions now (a trial)")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds to wait between questions")
    ap.add_argument("--dry-run", action="store_true", help="show the prompts of the first questions; no API call, no key needed")
    ap.add_argument("--env-file", default=".env", help="the file with the API keys")
    ap.add_argument("--data", default="data")
    ap.add_argument("--results", default="results")
    return ap.parse_args(argv)


def llm_only_main(argv=None):
    args = parse_args(argv)
    final, partial = run_folder(args.results, args.model, args.view, args.split)
    name = f"llm_only.{args.model}.{args.view}.{args.split}"
    log, client = get_logger(name), None
    if not args.dry_run and not os.path.isdir(final):
        unusable = load_env(args.env_file)
        if unusable:
            log.warning(f"{args.env_file}: lines {unusable} are not NAME=value (or have no value) and were skipped")
        client = make_client(args.model, log)                       # stops here when the key is missing
        log = get_logger(name, os.path.join(partial, "run.log"))
    return run(args, client, log)


COMMANDS = {"llm-only": llm_only_main}      # the other baselines (standard RAG, RAG^2, the proposed system) are added here


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] not in COMMANDS:
        print("usage: python src/run.py llm-only --model <model> --view <view> --split <split> [--limit N] [--dry-run]\n"
              "commands: " + ", ".join(COMMANDS))
        return 0 if argv in ([], ["-h"], ["--help"]) else 2
    return COMMANDS[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main())

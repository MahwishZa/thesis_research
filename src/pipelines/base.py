"""Baseline 1, LLM only (no retrieval), and baseline 2, standard RAG: asks one of the research models every question of one view and
split, and saves its answers.
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
not reported) and no yes/no evidence flag: the model states the evidence itself, and the evaluation (src/evaluation/) scores that text. A reply without a letter, or a question the host's content
moderation refuses, is saved as an invalid answer (wrong in the evaluation (src/evaluation/)) and kept so it can be checked.

Output: results/llm_only/<view>/<model>_<split>/
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

Standard RAG (python -m src.main standard-rag, added 2026-11-24): the same models, prompt wording and answer rule, plus retrieval:
  1. query      the patient facts of the question (an image-only question has none, so its query is a fixed sentence and every
                image-only question gets the same passages). Nothing is written by a model: the query is made of the question's own
                words. The true class, the options and the corpus tags are never in the query.
  2. retrieval  the MedCPT query encoder and the index built by `python -m src.main index`; the k = 3 passages with the highest cosine
                similarity of the whole corpus (no filter by disease).
  3. prompt     the question, the options, the numbered passages (title and text; tags are not shown) and the instruction: the two lines
                above and a third line, "Evidence: [<number>] "<one sentence copied word for word from that passage>"" or
                "Evidence: none". The sentence is checked against the passage by the evaluation (src/evaluation/) (a quote is never trusted).
  Output: results/standard_rag/<view>/<model>_<split>/ with the same files as above and, in every line, query, contexts (rank, score,
  chunk_id, title, text, tags: the tags are for the evaluation only) and evidence.
  evidence is null for an explicit "Evidence: none"; a missing or unreadable line gives {"chunk_id": null, "quote": "", "unparsed": true}.

Usage (on the laptop):
  python -m src.main llm-only --model gpt-4o --view text_only --split val --dry-run --limit 2     # see the prompt; free
  python -m src.main llm-only --model gpt-4o --view text_only --split val --limit 5              # a small paid trial
  python -m src.main llm-only --model gpt-4o --view text_only --split test                       # a full run
  python -m src.main standard-rag --model gpt-4o --view text_only --split val --dry-run --limit 2 # prompts with passages; no API call"""
import datetime
import hashlib
import json
import os
import time

from src.data.loaders import question_file
from src.models.llm import MODELS, ApiError, Refused, image_b64
from src.pipelines.prompts import (
    PROMPT_NAME,
    RAG_PROMPT_NAME,
    SYSTEM,
    instruction,
    parse_answer,
    parse_evidence,
    user_text,
)
from src.utils.io import read_jsonl, write_json
from src.utils.reproducibility import SEED, text_sha256


def add_tokens(a, b):
    return None if a is None and b is None else (a or 0) + (b or 0)


def rag_system(mode):
    """The results folder of a standard-RAG run: the approved retrieval (hybrid-rerank, researcher's decision 2026-11-24) is the
    standard RAG; any other mode is a trial with its own folder, so it never skips or overwrites a standard-RAG run."""
    return "standard_rag" if mode == "hybrid-rerank" else "standard_rag_" + mode.replace("-", "_")


# ---- the run (checkpointed) ---------------------------------------------------------------------------------------------
ORDER = f"sha256 of '{SEED}:<case id>', ascending (the same order for every model, view and split; the file is not changed)"


def asking_order(questions):
    """The order the questions are asked in: a fixed shuffle by case id (approved 2026-11-22). The question files are sorted by
    class, so asking them in file order would make a --limit trial, or an interrupted run, cover only the first classes."""
    return sorted(questions, key=lambda q: hashlib.sha256(f"{SEED}:{q['id']}".encode("utf-8")).hexdigest())


def run_folder(results, model, view, split, system="llm_only"):
    final = os.path.join(results, system, view, f"{model}_{split}")
    return final, final + ".partial"


def run(args, client, log, rag=None):
    """One run: every question of one view and split, asked of one model. `rag` (a StandardRag) adds retrieval and evidence."""
    system = rag_system(rag.mode) if rag else "llm_only"
    style = getattr(args, "style", "rationale")
    qpath = os.path.join(args.data, question_file(args.view, args.split))
    questions = read_jsonl(qpath)
    final, partial = run_folder(args.results, args.model, args.view, args.split, system)
    if os.path.isdir(final) and not args.dry_run:
        log.info(f"already complete: {final}")
        return final
    if args.view != "text_only":
        missing = [q["image_path"] for q in questions if not os.path.isfile(os.path.join(args.data, q["image_path"]))]
        if missing:
            raise SystemExit(f"{len(missing)} images are missing, e.g. {os.path.join(args.data, missing[0])}")
    if rag:
        rag.ready(args.view)

    def build(q, image):
        if rag is None:
            return user_text(q, style, args.view), {}, None, None
        return rag.prepare(q, args.view, client, image)

    if args.dry_run:
        for q in asking_order(questions)[:args.limit or 3]:
            text, extra, _, _ = build(q, None)
            log.info(f"--- {q['id']} (image: {q.get('image_path', 'none')})\n[system] {SYSTEM}\n[query] {extra.get('query') or 'the image ' + extra.get('query_image', '-')}\n"
                     f"[user] {text}")
        return None
    os.makedirs(partial, exist_ok=True)
    out = os.path.join(partial, "predictions.jsonl")
    done = {r["id"] for r in read_jsonl(out)} if os.path.exists(out) else set()
    qsha = text_sha256(qpath)
    settings = {"baseline": system, "model": args.model, "model_settings": MODELS[args.model], "view": args.view,
                "split": args.split, "questions": qpath.replace("\\", "/"), "questions_sha256": qsha,
                "prompt": RAG_PROMPT_NAME if rag else PROMPT_NAME, "style": style, "system": SYSTEM,
                "instruction": rag.instruction(args.view) if rag else instruction(style, args.view)}
    if rag:
        settings["retrieval"] = rag.settings(args.view)
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
        with open(out, "a", encoding="utf-8", newline="\n") as f:      # LF on Windows too
            for k, q in enumerate(todo, 1):
                image = image_b64(args.data, q["image_path"]) if args.view != "text_only" else None
                start = time.time()
                text, extra, extra_in, extra_out = build(q, image)
                try:
                    reply = client.ask(SYSTEM, text, image)
                except Refused as e:
                    reply = {"text": "", "finish_reason": f"refused by host: {str(e)[:200]}"}
                answer = parse_answer(reply["text"], "".join(q["options"]))
                record = {"id": q["id"], "pred": answer["pred"], "valid": answer["pred"] is not None,
                          "rationale": answer["rationale"],
                          "raw_output": reply["text"], "finish_reason": reply.get("finish_reason"),
                          "served_by": reply.get("served_by"), "tokens_in": add_tokens(reply.get("tokens_in"), extra_in),
                          "tokens_out": add_tokens(reply.get("tokens_out"), extra_out), "seconds": round(time.time() - start, 2),
                          "model": args.model, "view": args.view, "split": args.split,
                          "prompt": RAG_PROMPT_NAME if rag else PROMPT_NAME,
                          "time": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
                if rag:
                    record.update(extra)
                    record["evidence"] = parse_evidence(reply["text"], extra["contexts"])
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

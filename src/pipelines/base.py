"""What every system shares: the prompts and the answer format, and the run of one system on one view and split (resumable,
checkpointed)."""
import datetime
import hashlib
import json
import os
import re
import time

from src.data.loaders import question_file
from src.models.llm import MODELS, ApiError, Refused, image_b64
from src.utils.io import read_jsonl, write_json
from src.utils.reproducibility import SEED, text_sha256

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
ANSWER = re.compile(r"(?i:answer)\**\s*[:\uff1a]\**\s*([^\n]*)")
LETTER = re.compile(r"(?<![A-Za-z])([A-H])(?![A-Za-z])")
REASONING_LABEL = re.compile(r"(?i)^[\s*_#>]*reasoning\s+and\s+evidence\s*\**\s*[:\-]\s*\**\s*")
LONE_LETTER = re.compile(r"^\W*\(?([A-H])\)?[.)]?\W*$")


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


# ---- standard RAG: query, retrieval, prompt and evidence (no internet; the encoder needs torch) ---------------------------
RAG_PROMPT_NAME = "standard_rag"
RAG_INSTRUCTION = ("Reply in exactly this form. First line: \"Reasoning and evidence: <at most three sentences that name the findings of "
                   "this case, {findings}, that support your choice; mention a passage as {numbers} only if it supports your "
                   "choice>\". Second line: \"Answer: <the letter of the most likely option> (<the name of that option>)\". Third line: "
                   "\"Evidence: [<the number of one passage>] <one sentence copied word for word from that passage that supports your "
                   "answer, between quotation marks>\" or \"Evidence: none\" if no passage supports your answer.")
PASSAGES_HEADER = "Retrieved passages (from medical sources; they may not be about this case):"
EVIDENCE_LINE = re.compile(r"(?im)^\W*evidence\s*\**\s*[:\-]\**\s*(.*)$")
EVIDENCE_CITATION = re.compile(r"\[\s*(\d+)\s*\]\s*[\"\u201c\u2018']?(.*?)[\"\u201d\u2019']?\s*$", re.S)
UNPARSED = {"chunk_id": None, "quote": "", "unparsed": True}


def rag_instruction(view, k=3):
    """The instruction of the RAG prompt for one view and k passages."""
    numbers = "[1]" if k == 1 else ", ".join(f"[{i}]" for i in range(1, k)) + f" or [{k}]"
    return RAG_INSTRUCTION.replace("{findings}", FINDINGS_IN[view]).replace("{numbers}", numbers)


def rag_text(q, view, contexts):
    options = "\n".join(f"{letter}. {text}" for letter, text in q["options"].items())
    passages = "\n\n".join(f"[{n}] {c['title']}\n{c['text']}" if c.get("title") else f"[{n}]\n{c['text']}"
                            for n, c in enumerate(contexts, 1))
    return f"{q['question']}\n\nOptions:\n{options}\n\n{PASSAGES_HEADER}\n{passages}\n\n{rag_instruction(view, len(contexts))}"


def parse_evidence(text, contexts):
    """The evidence line of a reply: None for an explicit "Evidence: none"; {"chunk_id", "quote"} for "Evidence: [2] "...""
    (chunk_id is the id of passage 2, or None when the number is outside the list, so the quote is not found); UNPARSED when
    there is no usable line. Whether the quote is really in the passage is checked by the evaluation (src/metrics.py), never here."""
    lines = EVIDENCE_LINE.findall(text or "")
    if not lines:
        return dict(UNPARSED)
    last = lines[-1].strip().strip("*_ ")
    if re.match(r"(?i)none\b", last):
        return None
    m = EVIDENCE_CITATION.match(last)
    if not m or not m.group(2).strip():
        return dict(UNPARSED)
    n = int(m.group(1))
    return {"chunk_id": contexts[n - 1]["chunk_id"] if 1 <= n <= len(contexts) else None, "quote": m.group(2).strip()}


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
        os.replace(partial, final)
        log.info(f"complete: {final}")
        return final
    log.info(f"unfinished ({len(done)}/{len(questions)}); run the same command again to continue")
    return partial


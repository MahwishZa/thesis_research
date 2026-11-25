"""The prompts and the answer format: the LLM-only prompt, the RAG prompt with numbered passages, and the parsing of the
answer letter and the evidence line."""
import re

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
    there is no usable line. Whether the quote is really in the passage is checked by the evaluation (src/evaluation/), never here."""
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

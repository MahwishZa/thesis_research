"""Per-paper stance (stage 2): what does ONE study say about the question's claim?

Reading five abstracts at once is noisy (the same five papers in another order change 14% of
verdicts). Here the model reads one paper at a time and gives one letter: the study supports
the claim, contradicts it, or says nothing clear. The three probabilities come from the first
output token's distribution over the three letters; ``synthesis.py`` turns them into features.

Input to the model per paper: the question (read as a claim), the paper's title and its RESULTS
and CONCLUSIONS sections (an abstract without labelled sections contributes its last three
sentences), at most 200 words. Two wordings with different letter orders exist (A, B); the pilot
runs both, the full runs use the one that scored higher in the hand check.

    python -m experiments.medchange.stance --split dev --pilot --model-path models\\<file>.gguf
    python -m experiments.medchange.stance --split dev --model-path models\\<file>.gguf

Backends: ``llama`` (default; the same GGUF model as the answers; first-token log-probabilities,
or ``--hard-labels`` for the plain letter) and ``flan`` (Flan-T5-large, the one declared fallback).
Output: ``data/stance_<split>.jsonl`` (``stance_pilot.jsonl`` for the pilot) plus a configuration
record beside it; resumable; a resume under a different model or setting is refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

from .generate_answers import check_config, config_path, file_sha256, load_jsonl

HERE = Path(__file__).resolve().parent
TOP_K = 8
MAX_SNIPPET_WORDS = 200
SEED = 42
CLASSES = ("supports", "contradicts", "neither")   # canonical order of every probability vector
LETTERS = ("A", "B", "C")
MIN_LETTER_MASS = 0.5            # first-token probability that must sit on the three letters
FLAN_MODEL = "google/flan-t5-large"
FLAN_MAX_TOKENS = 512

SYSTEM = ("You are a careful medical evidence reader. You judge what a single study says and "
          "you do not use outside knowledge.")

WORDINGS = {
    "A": {
        "letters": {"A": "supports", "B": "contradicts", "C": "neither"},
        "template": (
            "Question, read as a claim to be checked: {question}\n\n"
            "Below are the title and the key findings of ONE study. Decide what this study alone "
            "says about the claim.\n"
            "A = the study's findings support the claim (a benefit or effect consistent with it)\n"
            "B = the study's findings contradict the claim (no benefit, harm, or an opposite effect)\n"
            "C = the study says nothing clear about the claim (unrelated, inconclusive or mixed)\n\n"
            "Study: {study}\n\n"
            "Answer with a single letter: A, B or C."),
    },
    "B": {
        "letters": {"A": "neither", "B": "supports", "C": "contradicts"},
        "template": (
            "Claim to check (taken from this question): {question}\n\n"
            "Read the study excerpt below, then say how it bears on the claim.\n"
            "A = unclear or unrelated: the excerpt does not settle the claim either way\n"
            "B = in favour: the excerpt reports results consistent with the claim being true or "
            "the treatment working\n"
            "C = against: the excerpt reports no benefit, harm, or the opposite of the claim\n\n"
            "Excerpt: {study}\n\n"
            "Reply with one letter only (A, B or C)."),
    },
}


# --------------------------------------------------------------------------------------
# What the model reads: title + RESULTS + CONCLUSIONS
# --------------------------------------------------------------------------------------

_LABEL = re.compile(r"(?:(?<=\s)|^)([A-Z][A-Z0-9 ,'’/&()-]{2,60}?):\s")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
KEY_RESULTS = ("RESULT", "FINDING")
KEY_CONCLUSIONS = ("CONCLUSION", "INTERPRETATION")
#: A capitalised word followed by a colon only counts as a section label if it contains one of
#: these words; "BMI:" or "HR:" inside running text must not cut a section.
BOUNDARY_WORDS = KEY_RESULTS + KEY_CONCLUSIONS + (
    "BACKGROUND", "OBJECTIVE", "AIM", "PURPOSE", "METHOD", "DESIGN", "SETTING", "PARTICIPANT",
    "PATIENT", "SUBJECT", "INTERVENTION", "MEASUREMENT", "OUTCOME", "INTRODUCTION", "CONTEXT",
    "DISCUSSION", "SEARCH", "SELECTION", "DATA", "STUDY", "MATERIAL", "REVIEW", "ELIGIBILITY",
    "LIMITATION", "FUNDING", "REGISTRATION", "TRIAL", "IMPLICATION", "SIGNIFICANCE", "RATIONALE",
    "EXPOSURE", "POPULATION", "SAMPLE", "CRITERIA", "SOURCE", "STRATEGY", "RELEVANCE")


def split_sections(abstract: str) -> list[tuple[str, str]]:
    """[(LABEL, text)] for an abstract whose sections are marked ``LABEL: text`` (PubMed's
    structured abstracts, as stored in the frozen pools); [] when it has no recognised labels."""
    marks = []
    for m in _LABEL.finditer(abstract or ""):
        label = m.group(1).strip()
        if any(word in label for word in BOUNDARY_WORDS):
            marks.append((m.start(1), m.end(), label))
    out = []
    for i, (_, end, label) in enumerate(marks):
        stop = marks[i + 1][0] if i + 1 < len(marks) else len(abstract)
        out.append((label, abstract[end:stop].strip()))
    return out


def key_text(abstract: str) -> tuple[str, str]:
    """(results text, conclusions text); an unlabelled abstract gives ("", its last three sentences)."""
    sections = split_sections(abstract)
    results = [t for label, t in sections if any(w in label for w in KEY_RESULTS) and t]
    conclusions = [t for label, t in sections if any(w in label for w in KEY_CONCLUSIONS) and t]
    if results or conclusions:
        return " ".join(results), " ".join(conclusions)
    sentences = [x for x in _SENTENCE.split((abstract or "").strip()) if x]
    return "", " ".join(sentences[-3:])


def study_snippet(candidate: dict, max_words: int = MAX_SNIPPET_WORDS) -> str:
    """Title, then results, then conclusions, at most ``max_words`` words; the conclusions keep at
    least half of what is left after the title."""
    title = (candidate.get("title") or "").strip()
    results, conclusions = key_text(candidate.get("abstract") or "")
    budget = max(20, max_words - len(title.split()))
    r_words, c_words = results.split(), conclusions.split()
    c_keep = min(len(c_words), max(budget // 2, budget - len(r_words)))
    r_keep = min(len(r_words), budget - c_keep)
    parts = [title.rstrip(".")] if title else []
    if r_keep:
        parts.append("Results: " + " ".join(r_words[:r_keep]))
    if c_keep:
        parts.append("Conclusions: " + " ".join(c_words[:c_keep]))
    if not parts:
        return "No abstract available."
    return ". ".join(p.rstrip(".") for p in parts) + "."


def build_messages(wording: str, question: str, study: str) -> tuple[str, str]:
    """(system, user) for one paper; the study text comes last so earlier tokens are shared."""
    return SYSTEM, WORDINGS[wording]["template"].format(question=question.strip(), study=study)


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------
# Turning a model's answer into class probabilities
# --------------------------------------------------------------------------------------

def letter_probs_from_top_logprobs(top: Sequence[dict]) -> tuple[Optional[dict], float]:
    """({letter: probability, renormalised over A/B/C}, mass) from the first token's top
    log-probabilities; (None, 0.0) when none of the three letters is among them. ``mass`` is the
    probability the model put on the letters before renormalising."""
    found: dict[str, float] = {}
    for entry in top:
        token = str(entry["token"]).strip().upper()
        if token in LETTERS:
            found[token] = max(found.get(token, 0.0), math.exp(float(entry["logprob"])))
    mass = sum(found.values())
    if not found or mass <= 0:
        return None, 0.0
    return {letter: found.get(letter, 0.0) / mass for letter in LETTERS}, mass


def canonical_probs(wording: str, letter_probs: dict) -> tuple[float, float, float]:
    """Letter probabilities -> (supports, contradicts, neither) under the wording's letter order."""
    out = [0.0, 0.0, 0.0]
    for letter, cls in WORDINGS[wording]["letters"].items():
        out[CLASSES.index(cls)] += letter_probs.get(letter, 0.0)
    return tuple(out)


def argmax_class(probs: Optional[Sequence[float]]) -> str:
    """Most probable class; ties and invalid outputs resolve to "neither" (no signal)."""
    if probs is None:
        return "neither"
    best = max(probs)
    for cls in ("neither", "supports", "contradicts"):
        if probs[CLASSES.index(cls)] == best:
            return cls
    return "neither"                                            # pragma: no cover


class LlamaStance:
    """The GGUF model of the answers, one output token, log-probabilities of the three letters.

    Log-probabilities need ``logits_all=True`` in llama-cpp-python (≈ 0.5 GB extra memory at a
    1,536-token context); ``hard_labels`` skips that and reads the plain letter instead."""

    min_mass = MIN_LETTER_MASS

    def __init__(self, model_path, n_ctx: int = 1536, n_threads: Optional[int] = None,
                 n_gpu_layers: int = 0, hard_labels: bool = False):
        from llama_cpp import Llama
        self.hard = hard_labels
        self.n_ctx = n_ctx
        self.model_sha256 = file_sha256(model_path)
        self.name = f"llama-{'hard' if hard_labels else 'logprobs'}:{Path(model_path).name}"
        self._llm = Llama(model_path=str(model_path), n_ctx=n_ctx, n_threads=n_threads,
                          n_gpu_layers=n_gpu_layers, logits_all=not hard_labels, verbose=False,
                          seed=SEED)

    def score(self, build: Callable[[str], tuple[str, str]], study: str):
        system, user = build(study)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if self.hard:
            reply = self._llm.create_chat_completion(messages=messages, max_tokens=1, temperature=0.0)
            match = re.search(r"[ABC]", reply["choices"][0]["message"]["content"].upper())
            if not match:
                return None, 0.0
            return {l: float(l == match.group(0)) for l in LETTERS}, 1.0
        reply = self._llm.create_chat_completion(messages=messages, max_tokens=1, temperature=0.0,
                                                 logprobs=True, top_logprobs=20)
        content = reply["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        return letter_probs_from_top_logprobs(content)


def fit_study(build: Callable[[str], tuple[str, str]], study: str,
              count_tokens: Callable[[str], int], limit: int, floor_words: int = 20) -> str:
    """Shorten ``study`` (from the end) until the whole prompt fits ``limit`` tokens, so that a
    truncating tokenizer never cuts the instruction that follows the study."""
    words = study.split()
    while True:
        system, user = build(" ".join(words))
        if count_tokens(f"{system}\n\n{user}") <= limit or len(words) <= floor_words:
            return " ".join(words)
        words = words[:max(floor_words, int(len(words) * 0.85))]


class FlanT5Stance:
    """The declared fallback: unmodified Flan-T5-large, first decoder step over the three letters."""

    min_mass = 0.1
    model_sha256 = None

    def __init__(self, device: Optional[str] = None):
        self.device = device
        self.name = f"flan:{FLAN_MODEL}"
        self.n_ctx = FLAN_MAX_TOKENS
        self._tok = self._model = None

    def _ensure(self):
        if self._model is None:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(FLAN_MODEL)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(FLAN_MODEL).eval()
            if self.device:
                self._model.to(self.device)
            ids = [self._tok.encode(l, add_special_tokens=False) for l in LETTERS]
            if any(len(i) != 1 for i in ids):
                raise RuntimeError(f"'A', 'B' and 'C' must each be one token, got {ids}")
            self._ids = [i[0] for i in ids]
        return self._tok, self._model

    def score(self, build: Callable[[str], tuple[str, str]], study: str):
        import torch
        tok, model = self._ensure()
        study = fit_study(build, study, lambda t: len(tok(t)["input_ids"]), FLAN_MAX_TOKENS - 2)
        system, user = build(study)
        enc = tok(f"{system}\n\n{user}", truncation=True, max_length=FLAN_MAX_TOKENS,
                  return_tensors="pt")
        if self.device:
            enc = {k: v.to(self.device) for k, v in enc.items()}
        start = model.config.decoder_start_token_id
        with torch.inference_mode():
            dec = torch.full((1, 1), start, device=enc["input_ids"].device)
            logits = model(**enc, decoder_input_ids=dec).logits[0, 0, :]
            everything = torch.softmax(logits, dim=-1)
        picked = [float(everything[i]) for i in self._ids]
        mass = sum(picked)
        if mass <= 0:
            return None, 0.0
        return {l: p / mass for l, p in zip(LETTERS, picked)}, mass


# --------------------------------------------------------------------------------------
# Running it over a split (resumable)
# --------------------------------------------------------------------------------------

def top_candidates(pool: dict, k: int = TOP_K) -> list[dict]:
    """The first ``k`` candidates in cross-encoder order."""
    return sorted(pool["candidates"], key=lambda c: (c["rank"], c["pmid"]))[:k]


def derangement(ids: Sequence[str], seed: int) -> dict[str, str]:
    """{id: another id}, nobody mapped to itself (a seeded cyclic shift of a shuffle)."""
    ordered = sorted(ids)
    if len(ordered) < 2:
        raise ValueError("an irrelevant-paper control needs at least two items")
    random.Random(seed).shuffle(ordered)
    return {ordered[i]: ordered[(i + 1) % len(ordered)] for i in range(len(ordered))}


def pilot_items(ids: Sequence[str], n: int, seed: int = 20261003) -> list[str]:
    """A seeded sample of ``n`` item ids (no label is looked at)."""
    ordered = sorted(ids)
    random.Random(seed).shuffle(ordered)
    return sorted(ordered[:n])


def record_key(rec: dict) -> tuple:
    return (rec["item_id"], rec["pmid"], rec["wording"], bool(rec.get("control")), rec.get("source_item"))


def score_paper(scorer, wording: str, question: str, candidate: dict, clock=time.time) -> dict:
    """One output record (without item fields)."""
    study = study_snippet(candidate)
    started = clock()
    letter_probs, mass = scorer.score(lambda s: build_messages(wording, question, s), study)
    seconds = clock() - started
    valid = letter_probs is not None and mass >= scorer.min_mass
    probs = list(canonical_probs(wording, letter_probs)) if valid else None
    return {"pmid": candidate["pmid"], "rank": candidate["rank"], "wording": wording,
            "backend": scorer.name, "probs": probs, "letter_mass": round(mass, 4),
            "argmax": argmax_class(probs), "seconds": round(seconds, 2),
            "snippet_sha256": _text_sha256(study)[:16]}


def run(items: Sequence[dict], pools: dict, scorer, wording: str, out: Path, *, top_k: int = TOP_K,
        control: Optional[dict] = None, clock=time.time,
        on_progress: Callable[[int, int], None] = lambda a, b: None) -> int:
    """Score the first ``top_k`` candidates of every item's pool, appending one JSON line per paper
    and skipping finished ones. With ``control`` ({item_id: other_item_id}) the papers come from the
    other item's pool but are judged against this item's question (irrelevant-paper control)."""
    done = {record_key(r) for r in load_jsonl(out)}
    work = []
    for item in items:
        src = control[item["item_id"]] if control else item["item_id"]
        if src not in pools:
            continue
        for cand in top_candidates(pools[src], top_k):
            key = (item["item_id"], cand["pmid"], wording, bool(control), src if control else None)
            if key not in done:
                work.append((item, cand, src))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8", newline="\n") as handle:
        for k, (item, cand, src) in enumerate(work, 1):
            rec = score_paper(scorer, wording, item["question"], cand, clock)
            rec.update(item_id=item["item_id"], control=bool(control), source_item=src if control else None)
            handle.write(json.dumps(rec) + "\n")
            handle.flush()
            on_progress(k, len(work))
    return len(work)


RELEVANT = ("backend", "model_sha256", "top_k", "max_snippet_words", "system_sha256",
            "wording_sha256_A", "wording_sha256_B", "n_ctx")


def stance_config(scorer, top_k: int) -> dict:
    return {"backend": scorer.name, "model_sha256": scorer.model_sha256, "top_k": top_k,
            "max_snippet_words": MAX_SNIPPET_WORDS, "system_sha256": _text_sha256(SYSTEM),
            "wording_sha256_A": _text_sha256(WORDINGS["A"]["template"]),
            "wording_sha256_B": _text_sha256(WORDINGS["B"]["template"]),
            "n_ctx": scorer.n_ctx, "temperature": 0.0, "seed": SEED}


def load_choice(data_dir: Path) -> Optional[dict]:
    path = data_dir / "stance_choice.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--split", default="dev", choices=("dev", "confirm"))
    ap.add_argument("--backend", default="llama", choices=("llama", "flan"))
    ap.add_argument("--model-path", default=None, help="GGUF file (llama backend)")
    ap.add_argument("--wording", default="auto", choices=("auto", "A", "B"),
                    help="auto = the wording chosen by the pilot hand check (stance_choice.json)")
    ap.add_argument("--top", type=int, default=TOP_K)
    ap.add_argument("--pilot", action="store_true",
                    help="40 seeded dev items, both wordings, plus the irrelevant-paper control")
    ap.add_argument("--pilot-n", type=int, default=40)
    ap.add_argument("--hard-labels", action="store_true", help="plain letter instead of log-probabilities")
    ap.add_argument("--n-ctx", type=int, default=1536)
    ap.add_argument("--n-threads", type=int, default=None)
    ap.add_argument("--device", default=None, help="flan backend: cpu or cuda")
    ap.add_argument("--limit", type=int, default=None, help="first N items only (timing test)")
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    data = Path(args.data_dir)

    if args.pilot and args.split != "dev":
        print("the pilot runs on the dev split only", file=sys.stderr)
        return 2
    if args.backend == "llama" and not args.model_path:
        print("--model-path is required for the llama backend", file=sys.stderr)
        return 2
    if args.backend == "llama" and not Path(args.model_path).is_file():
        print(f"model file not found: {args.model_path}", file=sys.stderr)
        return 2
    wordings = ["A", "B"] if args.pilot else [args.wording]
    if wordings == ["auto"]:
        choice = load_choice(data)
        if not choice:
            print("no wording chosen yet: run the pilot and its hand check first "
                  "(stance_check score), or pass --wording A or B", file=sys.stderr)
            return 2
        wordings = [choice["wording"]]

    bench = [r for r in load_jsonl(data / "benchmark.jsonl")
             if r["split"] == args.split and not r["likely_label_noise"]]
    pools = {r["item_id"]: r for r in load_jsonl(data / f"frozen_{args.split}.jsonl")}
    items = sorted((r for r in bench if r["item_id"] in pools), key=lambda r: r["item_id"])
    if args.pilot:
        keep = set(pilot_items([r["item_id"] for r in items], args.pilot_n))
        items = [r for r in items if r["item_id"] in keep]
    items = items[:args.limit]
    if not items:
        print("no items with frozen pools: run freeze_candidates first", file=sys.stderr)
        return 2

    scorer = (LlamaStance(args.model_path, args.n_ctx, args.n_threads, 0, args.hard_labels)
              if args.backend == "llama" else FlanT5Stance(args.device))
    out = Path(args.out or data / ("stance_pilot.jsonl" if args.pilot else f"stance_{args.split}.jsonl"))
    differs = check_config(out, stance_config(scorer, args.top), RELEVANT)
    if differs:
        print(f"refusing to extend {out.name}: its recorded stance configuration differs in "
              f"{', '.join(differs)} (see {config_path(out).name}). Use a different --out.", file=sys.stderr)
        return 2

    started = time.time()
    progress = (lambda k, t: print(f"  {k}/{t} ({time.time() - started:.0f}s)", flush=True)
                if k % 10 == 0 or k == t else None)
    total = 0
    for wording in wordings:
        total += run(items, pools, scorer, wording, out, top_k=args.top, on_progress=progress)
    if args.pilot:
        control = derangement([r["item_id"] for r in items], seed=20261004)
        total += run(items, pools, scorer, wordings[0], out, top_k=args.top, control=control,
                     on_progress=progress)
    print(f"scored {total} papers -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Adapted RAG² baseline and evidence-criteria verification: the pure logic (no model, no network).

The realigned study (docs/experimentation.md) compares, on the same as-of candidate records:

* ``R2``     adapted RAG²: a rationale as the dense query, retrieval balanced over evidence types, a
             zero-shot LLM filter, and the standard answer prompt of ``prompts.py``. The baseline.
* ``R2-RQ``, ``R2-BR``, ``R2-NF``  R2 without the rationale query / the balancing / the filter (ablations).
* ``R2C``    R2's evidence read once with the evidence criteria and design/date labels (criteria control).
* ``R2V``    proposed: R2's answer as a draft, checked against the evidence with the criteria.
* ``R2V-ND`` R2V without the dates and without the currency criterion (temporal ablation).

Everything that decides what a system reads or writes lives here, so ``SETTINGS`` and the prompt texts
are hashed into every output (``design_record``) and frozen before the confirmatory run.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Optional, Sequence

import numpy as np

from .freeze_candidates import eligible
from .prompts import SYSTEM as ANSWER_SYSTEM
from .prompts import TEMPLATE as ANSWER_TEMPLATE
from .synthesis import LABELS, study_type

STRATA = ("SR/MA", "RCT", "other")
QUOTA = 8                    # candidates per evidence type taken by the rationale's dense score
RERANK_KEEP = 8              # cross-encoder top-k handed to the filter
BUDGET = 5                   # admitted abstracts, as for B1
FILTER_THRESHOLD = 0.5       # P(yes) needed to pass the filter
MIN_YES_NO_MASS = 0.5        # first-token probability that must sit on "yes"/"no" for a valid judgement
RATIONALE_MAX_NEW_TOKENS = 128
ANSWER_MAX_NEW_TOKENS = 160
ANSWER_N_CTX = 6144
FILTER_N_CTX = 1536

LISTS = ("R2", "R2-RQ", "R2-BR")
LIST_OF = {"R2": "R2", "R2-NF": "R2", "R2-RQ": "R2-RQ", "R2-BR": "R2-BR",
           "R2C": "R2", "R2V": "R2", "R2V-ND": "R2"}
ARMS = ("R2", "R2-NF", "R2-RQ", "R2-BR", "R2C", "R2V", "R2V-ND")
#: Arms whose answer is written with the evidence criteria (all of them reuse R2's admitted set).
CRITERIA_ARMS = ("R2C", "R2V", "R2V-ND")

SETTINGS = {"strata": list(STRATA), "quota_per_stratum": QUOTA, "rerank_keep": RERANK_KEEP,
            "budget": BUDGET, "filter_threshold": FILTER_THRESHOLD, "min_yes_no_mass": MIN_YES_NO_MASS,
            "rationale_max_new_tokens": RATIONALE_MAX_NEW_TOKENS,
            "answer_max_new_tokens": ANSWER_MAX_NEW_TOKENS, "answer_n_ctx": ANSWER_N_CTX,
            "filter_n_ctx": FILTER_N_CTX, "filter_snippet": "title + results + conclusions, <= 200 words",
            "dense_query": "rationale (question if the rationale is empty)",
            "rerank_query": "question", "no_evidence": "standard closed-book answer",
            "invalid_verification": "draft verdict kept"}

# --------------------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------------------

RATIONALE_SYSTEM = "You are a medical expert."
RATIONALE_TEMPLATE = (
    "The following is a question about medical evidence. Solve it in a step-by-step fashion, starting by "
    "summarizing the available information: the population, the intervention or exposure, the comparison "
    "and the outcome. Then explain what is known and end with your tentative answer (supported, refuted, or "
    "not enough information).\n\nHere is the question: {question}")

FILTER_SYSTEM = "You are a careful medical evidence reader."
FILTER_TEMPLATE = (
    "Question: {question}\n\nDocument: {study}\n\n"
    "Does this document contain information that helps answer the question? Answer Yes or No.")

JUDGE_SYSTEM = "You are a careful medical evidence reader."
JUDGE_TEMPLATE = (
    "Question: {question}\n\nStudy: {study}\n\n"
    "Does this study directly test the intervention or exposure named in the question, in the question's "
    "population and for its outcome? Answer Yes or No.")

REVIEW_SYSTEM = "You are a careful medical evidence reviewer. You judge only from the studies you are given."

CRITERIA = (
    "Evidence criteria:\n"
    "1. Directness: a study counts only if it tests the intervention or exposure named in the question, in the "
    "question's population and for its outcome. Studies of other interventions, other populations, or only "
    "laboratory or surrogate measures are indirect and do not count.\n"
    "2. Design: randomized controlled trials and systematic reviews or meta-analyses carry the most weight; "
    "other designs count less.\n"
    "3. SUPPORTED: the direct studies show that the intervention works or the claim is true, at least partially, "
    "even if the certainty is low.\n"
    "4. REFUTED: the direct studies show no benefit, an effect similar to placebo or to the comparison, or harm.\n"
    "5. NOT ENOUGH INFORMATION: only when there are no direct studies; not because the certainty is low or the "
    "results are mixed.\n"
    "6. If direct studies disagree, decide by the weight of the direct randomized evidence{currency}.")
CURRENCY = ", and let newer evidence take precedence over older evidence it may have superseded"
CRITERIA_DATED = CRITERIA.format(currency=CURRENCY)
CRITERIA_UNDATED = CRITERIA.format(currency="")

FORMAT = ("Reply in exactly this format:\n"
          "DIRECT STUDIES: <the numbers of the direct studies, or NONE>\n"
          "FINDINGS: <one sentence on what the direct studies found>\n"
          "FINAL VERDICT: <SUPPORTED, REFUTED or NOT ENOUGH INFORMATION>")
CRIT_TASK = "Answer the question by applying the evidence criteria to the studies above.\n\n" + FORMAT
VERIFY_TASK = (
    "Below is a draft answer to the question. Check it against the studies above using the evidence criteria. "
    "The draft may rely on indirect studies, overlook studies that found no benefit, or use knowledge that is "
    "not in the studies; keep its verdict only if the criteria support it.\n\nDraft answer:\n{draft}\n\n" + FORMAT)
#: The criteria prompts share everything up to the task, so consecutive calls reuse the evaluated prefix.
REVIEW_TEMPLATE = "Question: {question}\n\nStudies:\n{studies}\n\n{criteria}\n\n{task}"

DESIGN_LABEL = {"SR/MA": "systematic review or meta-analysis", "RCT": "randomized or controlled trial",
                "other": "other study design"}

PROMPT_TEXTS = {"answer_system": ANSWER_SYSTEM, "answer_template": ANSWER_TEMPLATE,
                "rationale_system": RATIONALE_SYSTEM, "rationale_template": RATIONALE_TEMPLATE,
                "filter_system": FILTER_SYSTEM, "filter_template": FILTER_TEMPLATE,
                "review_system": REVIEW_SYSTEM, "criteria_dated": CRITERIA_DATED,
                "criteria_undated": CRITERIA_UNDATED, "crit_task": CRIT_TASK, "verify_task": VERIFY_TASK,
                "review_template": REVIEW_TEMPLATE}


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def settings_hash() -> str:
    return sha256(json.dumps(SETTINGS, sort_keys=True))[:12]


def prompt_hashes() -> dict:
    return {name: sha256(text)[:16] for name, text in sorted(PROMPT_TEXTS.items())}


def design_record(model_sha256: str, encoders: Optional[dict] = None) -> dict:
    """Everything that fixes the realigned systems; frozen (committed) before the confirmatory run."""
    from experiments.shared.retrieval.encoders import ARTICLE_ENCODER, CROSS_ENCODER, QUERY_ENCODER
    return {"settings": SETTINGS, "settings_sha256": settings_hash(), "prompts": prompt_hashes(),
            "generator_model_sha256": model_sha256,
            "encoders": encoders or {"query": QUERY_ENCODER, "article": ARTICLE_ENCODER,
                                     "reranker": CROSS_ENCODER}}


def build_rationale_prompt(question: str) -> tuple[str, str]:
    return RATIONALE_SYSTEM, RATIONALE_TEMPLATE.format(question=question.strip())


def build_filter_messages(question: str, study: str) -> tuple[str, str]:
    return FILTER_SYSTEM, FILTER_TEMPLATE.format(question=question.strip(), study=study)


def build_judge_messages(question: str, study: str) -> tuple[str, str]:
    return JUDGE_SYSTEM, JUDGE_TEMPLATE.format(question=question.strip(), study=study)


# --------------------------------------------------------------------------------------
# Retrieval: rationale query + balanced retrieval + cross-encoder re-ranking
# --------------------------------------------------------------------------------------

def balanced(order: Sequence[int], strata: Sequence[str], quota: int = QUOTA) -> list[int]:
    """The first ``quota`` indices of every evidence type in ``order``, returned in ``order``."""
    taken = {s: 0 for s in STRATA}
    out = []
    for i in order:
        s = strata[i]
        if taken[s] < quota:
            taken[s] += 1
            out.append(i)
    return out


def _ranked(scores: Sequence[float], records: Sequence[dict]) -> list[int]:
    return sorted(range(len(records)), key=lambda i: (-float(scores[i]), records[i]["pmid"]))


def lists_hash(lists: dict) -> str:
    payload = {v: [[c["pmid"], c["rank"], c["lower"], c["upper"]] for c in cands]
               for v, cands in sorted(lists.items())}
    return sha256(json.dumps(payload, separators=(",", ":")))


def build_lists(item: dict, records: Sequence[dict], abstracts: dict, rationale: str, *, query_encoder,
                article_encoder, reranker, quota: int = QUOTA, keep: int = RERANK_KEEP) -> dict:
    """The ranked candidate lists of R2 and of its two retrieval ablations for one item.

    ``R2``: dense ranking by the rationale, ``quota`` per evidence type, cross-encoder re-ranking against the
    question, top ``keep``. ``R2-RQ``: the same with the question as the dense query. ``R2-BR``: the
    rationale's dense ranking without balancing, the same number of candidates re-ranked.
    """
    cutoff = item["newest"]["date"]
    recs = [r for r in records if r["upper"] and r["upper"] <= cutoff and eligible(r, abstracts)]
    base = {"item_id": item["item_id"], "cutoff": cutoff, "n_eligible": len(recs),
            "rationale_used": bool(rationale.strip())}
    if not recs:
        lists = {v: [] for v in LISTS}
        return dict(base, lists=lists, lists_hash=lists_hash(lists))
    texts = [f'{abstracts[r["pmid"]]["title"]}. {abstracts[r["pmid"]]["abstract"]}' for r in recs]
    vectors = np.asarray(article_encoder.encode(texts))
    queries = np.asarray(query_encoder.encode([item["question"], rationale.strip() or item["question"]]))
    d_question, d_rationale = vectors @ queries[0], vectors @ queries[1]
    strata = [study_type(r.get("pubtypes")) for r in recs]
    by_rationale, by_question = _ranked(d_rationale, recs), _ranked(d_question, recs)
    chosen = balanced(by_rationale, strata, quota)
    selection = {"R2": chosen, "R2-RQ": balanced(by_question, strata, quota),
                 "R2-BR": by_rationale[:len(chosen)]}
    union = sorted(set().union(*selection.values()))
    ce = dict(zip(union, (float(s) for s in reranker.score(item["question"], [texts[i] for i in union]))))
    lists = {}
    for variant, idx in selection.items():
        ranked = sorted(idx, key=lambda i: (-ce[i], recs[i]["pmid"]))[:keep]
        lists[variant] = [{
            "pmid": recs[i]["pmid"], "rank": rank, "rerank_score": ce[i],
            "dense_rationale": float(d_rationale[i]), "dense_question": float(d_question[i]),
            "stratum": strata[i], "lower": recs[i]["lower"], "upper": recs[i]["upper"],
            "pubtypes": recs[i].get("pubtypes", []), "journal": recs[i].get("journal", ""),
            "title": abstracts[recs[i]["pmid"]]["title"], "abstract": abstracts[recs[i]["pmid"]]["abstract"]}
            for rank, i in enumerate(ranked, 1)]
    return dict(base, lists=lists, lists_hash=lists_hash(lists))


def strip_text(record: dict) -> dict:
    """A shareable copy of a lists record: identifiers, ranks, scores and dates, no publisher text."""
    out = {k: v for k, v in record.items() if k != "lists"}
    out["lists"] = {v: [{k: c[k] for k in c if k not in ("title", "abstract")} for c in cands]
                    for v, cands in record["lists"].items()}
    return out


# --------------------------------------------------------------------------------------
# Filter: P(yes) from the first token, admission
# --------------------------------------------------------------------------------------

def yes_probability(top: Sequence[dict]) -> tuple[Optional[float], float]:
    """(P(yes) renormalised over yes/no, yes+no mass) from a first token's top log-probabilities;
    (None, 0.0) when neither word is among them."""
    import math
    found = {"yes": 0.0, "no": 0.0}
    for entry in top:
        token = str(entry["token"]).strip().lower()
        if token in found:
            found[token] = max(found[token], math.exp(float(entry["logprob"])))
    mass = found["yes"] + found["no"]
    if mass <= 0:
        return None, 0.0
    return found["yes"] / mass, mass


def admit(arm: str, lists: dict, scores: dict, budget: int = BUDGET,
          threshold: float = FILTER_THRESHOLD) -> list[dict]:
    """The abstracts ``arm`` reads: its list in cross-encoder order, filtered (except R2-NF), at most
    ``budget``. ``scores`` maps pmid -> filter record; a missing judgement is an error, an invalid one keeps
    the paper."""
    cands = lists[LIST_OF[arm]]
    if arm == "R2-NF":
        return list(cands[:budget])
    out = []
    for c in cands:
        s = scores.get(c["pmid"])
        if s is None:
            raise KeyError(f"no filter judgement for {c['pmid']}")
        if not s["valid"] or s["p_yes"] >= threshold:
            out.append(c)
        if len(out) == budget:
            break
    return out


# --------------------------------------------------------------------------------------
# Criteria prompts and verification
# --------------------------------------------------------------------------------------

def publication_year(c: dict) -> str:
    return (c.get("upper") or c.get("lower") or "")[:4] or "year unknown"


def format_studies(passages: Sequence[dict], dated: bool) -> str:
    rows = []
    for i, p in enumerate(passages, 1):
        label = DESIGN_LABEL[p.get("stratum") or study_type(p.get("pubtypes"))]
        meta = f"{label}, {publication_year(p)}" if dated else label
        rows.append(f"[{i}] ({meta}) {p['title']}. {p['abstract']}")
    return "\n".join(rows)


def build_criteria_prompt(question: str, passages: Sequence[dict], *, draft: Optional[str] = None,
                          dated: bool = True) -> tuple[str, str]:
    """(system, user) for R2C (``draft`` None) or the verification arms (R2V dated, R2V-ND undated)."""
    task = CRIT_TASK if draft is None else VERIFY_TASK.format(draft=draft.strip())
    user = REVIEW_TEMPLATE.format(question=question.strip(), studies=format_studies(passages, dated),
                                  criteria=CRITERIA_DATED if dated else CRITERIA_UNDATED, task=task)
    return REVIEW_SYSTEM, user


_FINAL = re.compile(r"FINAL\s+VERDICT\W*(NOT\s+ENOUGH\s+INFORMATION|SUPPORTED|REFUTED)", re.IGNORECASE)
_ECHO = "SUPPORTED, REFUTED or NOT ENOUGH INFORMATION"


def parse_final_verdict(text: str) -> Optional[str]:
    """The last ``FINAL VERDICT`` line; None when absent or when the format line was copied verbatim."""
    if not text or _ECHO.lower() in text.lower():
        return None
    found = _FINAL.findall(text)
    return re.sub(r"\s+", " ", found[-1].upper()) if found else None


def direct_numbers(text: str) -> Optional[list[int]]:
    """Study numbers on the ``DIRECT STUDIES`` line ([] for NONE); None when the line is missing."""
    m = re.search(r"DIRECT\s+STUDIES\W*([^\n]*)", text or "", re.IGNORECASE)
    if not m:
        return None
    return sorted({int(x) for x in re.findall(r"\d+", m.group(1))})


def cited_numbers(text: str) -> list[int]:
    """Numbers cited as ``[n]``, ``[n, m]`` or ``[n-m]`` in a standard answer."""
    out = set()
    for group in re.findall(r"\[([\d\s,;–-]+)\]", text or ""):
        for a, b in re.findall(r"(\d+)\s*[–-]\s*(\d+)", group):
            out.update(range(int(a), int(b) + 1))
        out.update(int(x) for x in re.findall(r"\d+", group))
    return sorted(out)


def unsupported_decisive(verdict: Optional[str], text: str, n_admitted: int) -> Optional[bool]:
    """True when a SUPPORTED/REFUTED verdict cites none of the admitted studies; None without evidence."""
    if n_admitted == 0:
        return None
    if verdict not in ("SUPPORTED", "REFUTED"):
        return False
    numbers = set(cited_numbers(text)) | set(direct_numbers(text) or [])
    return not any(1 <= n <= n_admitted for n in numbers)


_YEAR = re.compile(r"(?<!\d)(19[5-9]\d|20[0-4]\d)(?!\d)")


def anachronistic_years(text: str, cutoff: str) -> list[int]:
    """Years mentioned in ``text`` that are later than the question date's year (no admitted study, all
    published before the question date, can support them). Approximate: a four-digit count is read as a year."""
    limit = int(cutoff[:4])
    return sorted({int(y) for y in _YEAR.findall(text or "") if int(y) > limit})


def label_index(label: Optional[str]) -> Optional[int]:
    return LABELS.index(label) if label in LABELS else None

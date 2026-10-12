"""Measurements on the answers of the realigned study: per arm the verdict accuracy, the unsupported-answer indicators, what the
criteria arms changed and what each arm retrieved, and the paired comparisons with the reading of the primary one.

Shared by ``analyze_rag2`` (the held-out and development splits), ``analyze_fresh`` (the pre-registered test) and ``ad_subgroup``;
the tables and the command line are in ``analyze_rag2``. The primary comparison is R2V - R2 over all items (exact McNemar, paired
bootstrap 95% interval), read as *confirmed* (p < .05 and the interval above 0), *positive but not confirmed* (the difference is
above 0 otherwise) or *no evidence* (the difference is not above 0) (``docs/protocol.md`` §1, ``docs/evaluation.md`` §5). Secondary
comparisons are Holm-corrected among themselves, the ablations likewise.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional, Sequence

from . import rag2 as R
from .abstracts import study_type
from .arms import point_date
from .files import load_jsonl
from .scoring import ALPHA, DIGITS, holm_family, paired, stable_difference, summarize

ARM_ORDER = ("B0", "B1", "R2", "R2-RQ", "R2-BR", "R2-NF", "R2C", "R2V", "R2V-ND")
EVIDENCE_ARMS = ("B1",) + tuple(a for a in ARM_ORDER if a.startswith("R2"))
PRIMARY = ("R2V", "R2", ("changed", "unchanged"))
SECONDARY = (("R2", "B1", ("changed", "unchanged")), ("R2V", "B1", ("changed", "unchanged")),
             ("R2C", "R2", ("changed", "unchanged")), ("R2V", "R2C", ("changed", "unchanged")),
             ("R2V", "R2V-ND", ("changed", "unchanged")), ("R2V", "R2", ("changed",)))
ABLATIONS = (("R2", "R2-RQ", ("changed", "unchanged")), ("R2", "R2-BR", ("changed", "unchanged")),
             ("R2", "R2-NF", ("changed", "unchanged")))
BOTH = ("changed", "unchanged")


def load_split(data: Path, split: str, answers_dir: Optional[Path] = None) -> tuple[dict, dict]:
    """(items, answers) of one split: the usable items of ``data/benchmark.jsonl`` and, as {(item id, arm): record}, the B0/B1 answers
    (``answers_<split>.jsonl``) and the answers of the R2 family (``rag2_answers_<split>.jsonl``) from ``answers_dir`` (default: ``data``)."""
    folder = data if answers_dir is None else answers_dir
    items = {r["item_id"]: r for r in load_jsonl(data / "benchmark.jsonl")
             if r["split"] == split and not r["likely_label_noise"]}
    answers = {}
    for name in (f"answers_{split}.jsonl", f"rag2_answers_{split}.jsonl"):
        for r in load_jsonl(folder / name):
            if r["item_id"] in items and r["arm"] in ARM_ORDER:
                answers[(r["item_id"], r["arm"])] = r
    return items, answers


def present_arms(answers: dict) -> list[str]:
    arms = {arm for _, arm in answers}
    return [a for a in ARM_ORDER if a in arms]


# --------------------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------------------

def hallucination_rows(items: dict, answers: dict, arms: Sequence[str]) -> dict:
    """Anachronism rate (all answers) and unsupported decisive verdicts (answers that had evidence)."""
    out = {}
    for arm in arms:
        ids = [i for i in items if (i, arm) in answers]
        if not ids:
            continue
        ana = [bool(R.anachronistic_years(answers[(i, arm)]["text"], items[i]["newest"]["date"])) for i in ids]
        flags = [R.unsupported_decisive(answers[(i, arm)]["verdict"], answers[(i, arm)]["text"],
                                        len(answers[(i, arm)].get("admitted", []))) for i in ids]
        flags = [f for f in flags if f is not None]
        out[arm] = {"n": len(ids), "anachronism_rate": round(sum(ana) / len(ids), DIGITS),
                    "unsupported_decisive_rate": round(sum(flags) / len(flags), DIGITS) if flags else None,
                    "answers_with_evidence": len(flags)}
    return out


def truncation_rows(answers: dict, arms: Sequence[str]) -> dict:
    """Answers written after the abstracts were shortened because the prompt did not fit the context window."""
    out = {}
    for arm in arms:
        n = sum(1 for (_, a), r in answers.items() if a == arm and r.get("context_truncated"))
        if n:
            out[arm] = n
    return out


def verifier_rows(items: dict, answers: dict, arms: Sequence[str]) -> dict:
    """For the criteria arms: valid outputs, how often the verdict differs from R2's, and whether a change
    fixed or broke the answer."""
    out = {}
    for arm in arms:
        if arm not in R.CRITERIA_ARMS:
            continue
        ids = [i for i in items if (i, arm) in answers and answers[(i, arm)].get("fallback") is None]
        if not ids:
            continue
        recs = [answers[(i, arm)] for i in ids]
        changed = [i for i, r in zip(ids, recs) if r["changed"]]
        gold = {i: items[i]["newest"]["label"] for i in ids}
        fixed = sum(answers[(i, arm)]["verdict"] == gold[i] and answers[(i, arm)]["draft_verdict"] != gold[i]
                    for i in changed)
        broke = sum(answers[(i, arm)]["verdict"] != gold[i] and answers[(i, arm)]["draft_verdict"] == gold[i]
                    for i in changed)
        transitions: dict = {}
        for i in changed:
            key = f"{answers[(i, arm)]['draft_verdict']} -> {answers[(i, arm)]['verdict']}"
            transitions[key] = transitions.get(key, 0) + 1
        out[arm] = {"n_with_evidence": len(ids), "valid_rate": round(sum(r["valid"] for r in recs) / len(ids), DIGITS),
                    "changed_rate": round(len(changed) / len(ids), DIGITS), "changes_fixed": fixed,
                    "changes_broke": broke, "transitions": dict(sorted(transitions.items()))}
    return out


# --------------------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------------------

def _admitted_meta(item_id: str, rec: dict, frozen: dict) -> list[dict]:
    if "admitted_stratum" in rec:
        return [{"pmid": p, "lower": lo, "upper": up, "stratum": s}
                for p, lo, up, s in zip(rec["admitted"], rec["admitted_lower"], rec["admitted_upper"],
                                        rec["admitted_stratum"])]
    pool = {c["pmid"]: c for c in frozen.get(item_id, {}).get("candidates", [])}
    out = []
    for p in rec.get("admitted", []):
        c = pool.get(p)
        if c is not None:
            out.append({"pmid": p, "lower": c["lower"], "upper": c["upper"], "stratum": study_type(c.get("pubtypes"))})
    return out


def retrieval_rows(items: dict, answers: dict, arms: Sequence[str], frozen: dict,
                   directness: Optional[dict] = None) -> dict:
    """Per evidence arm: how much was admitted, of which evidence types, how recent, how much from the update
    window, overlap with B1, and (if judged) the share judged directly on the question."""
    out = {}
    for arm in arms:
        if arm not in EVIDENCE_ARMS:
            continue
        ids = [i for i in items if (i, arm) in answers]
        if not ids:
            continue
        counts, zero, mix, shares, any_window, ages, jac, direct = [], 0, {s: 0 for s in R.STRATA}, [], 0, [], [], []
        for i in ids:
            it, rec = items[i], answers[(i, arm)]
            meta = _admitted_meta(i, rec, frozen)
            counts.append(len(rec.get("admitted", [])))
            zero += not rec.get("admitted")
            for m in meta:
                mix[m["stratum"]] += 1
            flags = [it["previous"]["date"] < m["lower"] and m["upper"] <= it["newest"]["date"] for m in meta]
            if meta:
                shares.append(sum(flags) / len(meta))
            any_window += any(flags)
            cutoff = date.fromisoformat(it["newest"]["date"])
            ages += [max(0, (cutoff - point_date(m)).days) / 365.25 for m in meta]
            if arm != "B1" and (i, "B1") in answers:
                a_, b_ = set(rec.get("admitted", [])), set(answers[(i, "B1")].get("admitted", []))
                if a_ | b_:
                    jac.append(len(a_ & b_) / len(a_ | b_))
            if directness:
                judged = [directness[(i, p)] for p in rec.get("admitted", [])
                          if (i, p) in directness and directness[(i, p)]["valid"]]
                if judged:
                    direct.append(sum(j["p_yes"] >= R.FILTER_THRESHOLD for j in judged) / len(judged))
        total = sum(mix.values())
        out[arm] = {"n": len(ids), "mean_admitted": round(sum(counts) / len(ids), 2),
                    "share_without_evidence": round(zero / len(ids), DIGITS),
                    "evidence_mix": {s: round(v / total, DIGITS) for s, v in mix.items()} if total else None,
                    "update_window_share": round(sum(shares) / len(shares), DIGITS) if shares else None,
                    "items_with_update_window_evidence": round(any_window / len(ids), DIGITS),
                    "mean_age_years": round(sum(ages) / len(ages), 2) if ages else None,
                    "jaccard_with_B1": round(sum(jac) / len(jac), DIGITS) if jac else None,
                    "directness_at_k": round(sum(direct) / len(direct), DIGITS) if direct else None}
    return out


# --------------------------------------------------------------------------------------
# Comparisons and the reading of the primary difference
# --------------------------------------------------------------------------------------

def compare(items: dict, answers: dict, specs) -> dict:
    out = {}
    for a, b, kinds in specs:
        r = paired(items, answers, a, b, kinds)
        if r:
            out[f"{a} vs {b}" + ("" if tuple(kinds) == BOTH else f" ({'+'.join(kinds)})")] = r
    return out


def primary_result(items: dict, answers: dict) -> Optional[dict]:
    r = paired(items, answers, *PRIMARY)
    if not r:
        return None
    confirmed = r["mcnemar_p"] < ALPHA and r["ci95"][0] > 0 and r["diff_a_minus_b"] > 0
    return dict(r, confirmed=bool(confirmed))


def primary_reading(primary: Optional[dict], split: str) -> str:
    """The reading of the primary accuracy difference R2V - R2 (docs/evaluation.md §5): *confirmed*, *positive but not
    confirmed* or *no evidence*."""
    if primary is None:
        return "not run"
    if split not in ("confirm", "ad"):
        return "development estimate only (exploratory; read on the held-out split)"
    if primary["confirmed"]:
        return "confirmed"
    if primary["diff_a_minus_b"] > 0:
        return "positive but not confirmed"
    return "no evidence"


def stable_ids_from_audit(path: Optional[Path]) -> Optional[list[str]]:
    if not path or not path.is_file():
        return None
    return sorted(r["item_id"] for r in load_jsonl(path) if r.get("which") == "newest" and r["gold"] == r["relabel"])


def case_study(items: dict, answers: dict, arms: Sequence[str]) -> dict:
    ids = sorted(i for i, it in items.items() if it.get("ad_related"))
    rows = {}
    for i in ids:
        rows[i] = {"kind": items[i]["kind"], "gold": items[i]["newest"]["label"],
                   "previous": items[i]["previous"]["label"],
                   "verdicts": {a: answers[(i, a)]["verdict"] for a in arms if (i, a) in answers}}
    return {"items": rows, "correct": {a: sum(answers[(i, a)]["verdict"] == items[i]["newest"]["label"]
                                              for i in ids if (i, a) in answers) for a in arms},
            "n": len(ids)}


def build_report(items: dict, answers: dict, split: str, frozen: Optional[dict] = None,
           directness: Optional[dict] = None, stable_ids: Optional[Sequence[str]] = None) -> dict:
    arms = present_arms(answers)
    primary = primary_result(items, answers)
    rep = {"split": split, "items": len(items), "arms": arms,
           "generation": summarize(items, answers, arms),
           "unsupported_answers": hallucination_rows(items, answers, arms),
           "context_truncated": truncation_rows(answers, arms),
           "verification": verifier_rows(items, answers, arms),
           "retrieval": retrieval_rows(items, answers, arms, frozen or {}, directness),
           "primary": primary, "reading": primary_reading(primary, split),
           "secondary": holm_family(compare(items, answers, SECONDARY)),
           "ablations": holm_family(compare(items, answers, ABLATIONS)),
           "case_study_alzheimers": case_study(items, answers, arms)}
    if stable_ids:
        rep["label_stable"] = {"n": len(stable_ids),
                               "R2V_minus_R2": stable_difference(items, answers, "R2V", "R2", stable_ids),
                               "R2_minus_B1": stable_difference(items, answers, "R2", "B1", stable_ids)}
    return rep

"""Synthetic data and fakes shared by the tests of the realigned study (no model, no MedCPT download, no network)."""

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from experiments.medchange import rag2 as R
from experiments.medchange.benchmark import load_groups
from experiments.medchange.encoders import CrossEncoderReranker, Encoder

S, F_, N = "SUPPORTED", "REFUTED", "NOT ENOUGH INFORMATION"
TYPES = {"SR/MA": ["Systematic Review"], "RCT": ["Randomized Controlled Trial"], "other": ["Journal Article"]}


def write(path: Path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def item(iid, kind="changed", gold=S, prev=F_, date="2015-06-01", split="dev", ad=False):
    return {"item_id": iid, "kind": kind, "split": split, "likely_label_noise": False, "ad_related": ad,
            "question": f"Does drug {iid} reduce pain in adults?",
            "newest": {"label": gold, "date": date, "pmid": f"N{iid}"},
            "previous": {"label": prev, "date": "2010-01-01", "pmid": f"O{iid}"}}


def records(n_per_type=10, upper="2012-12-31", lower="2012-01-01"):
    out = []
    for kind, pubtypes in TYPES.items():
        for k in range(n_per_type):
            out.append({"pmid": f"{kind}-{k}", "lower": lower, "upper": upper, "pubtypes": pubtypes, "journal": "J"})
    return out


def abstracts(recs):
    return {r["pmid"]: {"pmid": r["pmid"], "title": f"Trial of drug pain {r['pmid']}",
                        "abstract": ("RESULTS: pain fell. CONCLUSIONS: drug reduces pain. " * 8)} for r in recs}


def cand(pmid, rank, stratum="RCT", upper="2012-12-31", lower="2012-01-01"):
    return {"pmid": pmid, "rank": rank, "rerank_score": 1.0 / rank, "stratum": stratum, "lower": lower,
            "upper": upper, "pubtypes": TYPES[stratum], "journal": "J", "title": f"Title {pmid}",
            "abstract": f"RESULTS: findings {pmid}. CONCLUSIONS: conclusion {pmid}."}


def lists_for(n=8):
    return {v: [cand(f"{v}-{i}", i + 1) for i in range(n)] for v in R.LISTS}


def scores_for(lists, p=0.9, valid=True):
    return {c["pmid"]: {"pmid": c["pmid"], "p_yes": p, "mass": 0.99, "valid": valid}
            for cands in lists.values() for c in cands}


def fake_generate(system, user):
    if "Draft answer" in user:
        return "DIRECT STUDIES: 1\nFINDINGS: no benefit over placebo.\nFINAL VERDICT: REFUTED"
    if "Evidence criteria" in user:
        return "DIRECT STUDIES: NONE\nFINDINGS: no direct study.\nFINAL VERDICT: NOT ENOUGH INFORMATION"
    if "step-by-step" in user:
        return "Population: adults. Intervention: drug. Outcome: pain. Tentative answer: supported."
    return "VERDICT: SUPPORTED\nStudy [1] shows benefit."


class FakeScorer:
    name, model_sha256, n_ctx = "fake", "0", 1

    def __init__(self, invalid=()):
        self.invalid = set(invalid)
        self.calls = 0

    def score(self, question, study):
        self.calls += 1
        if any(p in study for p in self.invalid):
            return None, 0.0
        return 0.8, 0.95


def answer(iid, arm, verdict, admitted=(), text=None, **extra):
    rec = {"item_id": iid, "arm": arm, "verdict": verdict, "admitted": list(admitted),
           "text": text if text is not None else f"VERDICT: {verdict} [1]"}
    if arm.startswith("R2"):
        rec.update(admitted_lower=["2012-01-01"] * len(admitted), admitted_upper=["2012-12-31"] * len(admitted),
                   admitted_stratum=["RCT"] * len(admitted))
    rec.update(extra)
    return rec


def world(n=40, r2v_extra=6):
    """n items, gold SUPPORTED; R2 right on half; R2V right on half plus ``r2v_extra`` more."""
    items = {f"MC-{k:03d}": item(f"MC-{k:03d}", kind="changed" if k % 2 else "unchanged",
                                 split="confirm", ad=(k == 0)) for k in range(n)}
    answers = {}
    for k, iid in enumerate(sorted(items)):
        r2_ok = k < n // 2
        r2v_ok = r2_ok or k < n // 2 + r2v_extra
        for arm, ok in (("B0", k < n // 3), ("B1", r2_ok), ("R2", r2_ok), ("R2C", r2v_ok), ("R2V", r2v_ok)):
            v = S if ok else N
            extra = {}
            if arm in ("R2C", "R2V"):
                extra = dict(draft_verdict=S if r2_ok else N, valid=True, changed=(v != (S if r2_ok else N)),
                             fallback=None)
            answers[(iid, arm)] = answer(iid, arm, v, admitted=() if arm == "B0" else ("p1", "p2"), **extra)
    return items, answers


def row(i, label, date, concl="conclusion text", q="Does X help Y?"):
    return {"": str(i), "Label": label, "Question": q, "objectives": "",
            "conclusions": concl, "PMID": f"/{1000 + i}/",
            "DOI_Date": f"Cochrane Database Syst Rev. {date};1:CD00{1000 + i:04d}. doi: x"}


def fixture():
    # group 1: changed (newest row 0 SUPPORTED, older row 1 NEI)
    # group 2: unchanged (rows 2,3 REFUTED); group 3: changed but near-identical text
    medrev = {
        0: row(0, "SUPPORTED", "2020 Mar 5", "New trials show a clear benefit of X."),
        1: row(1, "NOT ENOUGH INFORMATION", "2010 Jan 2", "Too few studies to judge."),
        2: row(2, "REFUTED", "2019 May 1"), 3: row(3, "REFUTED", "2012"),
        4: row(4, "SUPPORTED", "2018 Jun 1", "Identical wording of the conclusion here."),
        5: row(5, "NOT ENOUGH INFORMATION", "2014 Jun 1", "Identical wording of the conclusion here."),
    }
    groups = load_groups([
        {"Group_ID": "1", "Study_ID": "0"}, {"Group_ID": "", "Study_ID": "1"},
        {"Group_ID": "2", "Study_ID": "2"}, {"Group_ID": "", "Study_ID": "3"},
        {"Group_ID": "3", "Study_ID": "4"}, {"Group_ID": "", "Study_ID": "5"},
    ])
    return medrev, groups


class HashingEncoder(Encoder):
    """Deterministic stand-in for tests. **Never a thesis result.**

    Produces a stable pseudo-random vector per text by hashing. It has no
    semantics whatsoever: it exists so the index, retrieval and freezing code
    can be exercised on a machine without torch. Any retrieval quality it
    appears to show is an artifact of hashing.
    """

    name = "hashing-stub"

    def __init__(self, dim: int = 32) -> None:
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], "big")
            rows.append(np.random.default_rng(seed).standard_normal(self.dim))
        return (np.asarray(rows, dtype=np.float32) if rows
                else np.zeros((0, self.dim), dtype=np.float32))


class LexicalOverlapReranker(CrossEncoderReranker):
    """Deterministic stand-in reranker for tests. **Never a thesis result.**

    Scores by token overlap with the query. Unlike the hashing encoder this
    is weakly meaningful, which is useful for asserting that reranking
    reorders a candidate list at all - but it is not MedCPT and must never
    stand in for it in a run that produces numbers.
    """

    name = "lexical-overlap-stub"

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        wanted = set(query.lower().split())
        return np.asarray(
            [len(wanted & set(p.lower().split())) / (len(wanted) or 1)
             for p in passages],
            dtype=np.float32,
        )

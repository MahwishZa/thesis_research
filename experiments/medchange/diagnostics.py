"""P0 diagnostics for stage 2: three cheap checks on the frozen dev pools, no generation.

    python -m experiments.medchange.diagnostics --split dev --out-dir experiments\\medchange\\results

1. **Helpfulness inputs.** ``helpfulness.py`` puts the question *after* the abstract and truncates at
   512 tokens from the end, so for a long abstract the question and the "Answer yes or no" line may
   have been cut off. Reports how many of the stage-1 helpfulness inputs exceed 512 tokens (needs the
   Flan-T5 tokenizer that ``helpfulness`` already downloaded; ``--no-tokenizer`` skips it).
2. **What the stance step would read.** Share of candidates whose abstract has labelled RESULTS or
   CONCLUSIONS sections, and the length of the text the model would see.
3. **Study types.** Systematic reviews / meta-analyses, randomized trials and other papers in the pools,
   in the top 8 and in B1's admitted set, and how B1's verdicts and the gold verdicts differ between
   questions whose B1 evidence contains a systematic review and those that do not (descriptive only).

Writes ``diagnostics_<split>.json`` and ``.md`` (no source text).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Callable, Optional

from .generate_answers import load_jsonl
from .helpfulness import MAX_TOKENS, MODEL, build_inputs
from .stance import KEY_CONCLUSIONS, KEY_RESULTS, TOP_K, split_sections, study_snippet, top_candidates
from .synthesis import study_type

HERE = Path(__file__).resolve().parent
MIN_SR_SHARE = 0.30          # plan: the study-type weight needs a review in the top 8 for >= 30% of items


def _share(k: int, n: int) -> Optional[float]:
    return round(k / n, 4) if n else None


def helpfulness_truncation(items: dict, pools: dict, answers: dict,
                           count_tokens: Callable[[str], int], limit: int = MAX_TOKENS) -> dict:
    """How many stage-1 helpfulness inputs exceed ``limit`` tokens (and would lose the question)."""
    total = over = 0
    lengths = []
    over_by_pmid: dict[tuple, bool] = {}
    for item_id, it in items.items():
        if item_id not in pools:
            continue
        texts = build_inputs(it["question"], pools[item_id]["candidates"])
        for cand, text in zip(pools[item_id]["candidates"], texts):
            n = count_tokens(text)
            lengths.append(n)
            total += 1
            over += n > limit
            over_by_pmid[(item_id, cand["pmid"])] = n > limit
    b2_total = b2_over = 0
    for (item_id, arm), rec in answers.items():
        if arm != "B2":
            continue
        for pmid in rec["admitted"]:
            if (item_id, pmid) in over_by_pmid:
                b2_total += 1
                b2_over += over_by_pmid[(item_id, pmid)]
    return {"limit_tokens": limit, "inputs": total, "over_limit": over, "share_over_limit": _share(over, total),
            "median_tokens": statistics.median(lengths) if lengths else None,
            "b2_admitted_papers": b2_total, "b2_admitted_over_limit": b2_over,
            "b2_admitted_share_over_limit": _share(b2_over, b2_total)}


def _has_key_sections(c: dict) -> bool:
    return any(any(w in label for w in KEY_RESULTS + KEY_CONCLUSIONS)
               for label, text in split_sections(c.get("abstract") or "") if text)


def snippet_stats(items: dict, pools: dict) -> dict:
    cands = [c for i in items if i in pools for c in pools[i]["candidates"]]
    labelled = sum(_has_key_sections(c) for c in cands)
    words = [len(study_snippet(c).split()) for c in cands]
    return {"candidates": len(cands), "with_results_or_conclusions": _share(labelled, len(cands)),
            "snippet_words_median": statistics.median(words) if words else None,
            "snippet_words_max": max(words) if words else None}


def composition(cands) -> dict:
    kinds = [study_type(c.get("pubtypes")) for c in cands]
    n = len(kinds)
    return {k: _share(kinds.count(k), n) for k in ("SR/MA", "RCT", "other")}


def study_type_stats(items: dict, pools: dict, answers: dict) -> dict:
    pool_c, top_c = [], []
    sr_top8 = sr_b1 = n = 0
    groups = {"with_sr_ma_in_b1": {"n": 0, "b1_verdicts": {}, "gold": {}},
              "without_sr_ma_in_b1": {"n": 0, "b1_verdicts": {}, "gold": {}}}
    for item_id, it in items.items():
        if item_id not in pools:
            continue
        n += 1
        cands = pools[item_id]["candidates"]
        pool_c += cands
        top = top_candidates(pools[item_id], TOP_K)
        top_c += top
        sr_top8 += any(study_type(c.get("pubtypes")) == "SR/MA" for c in top)
        b1 = answers.get((item_id, "B1"))
        if b1:
            by_pmid = {c["pmid"]: c for c in cands}
            has = any(study_type(by_pmid[p].get("pubtypes")) == "SR/MA" for p in b1["admitted"] if p in by_pmid)
            sr_b1 += has
            g = groups["with_sr_ma_in_b1" if has else "without_sr_ma_in_b1"]
            g["n"] += 1
            g["b1_verdicts"][b1["verdict"] or "none"] = g["b1_verdicts"].get(b1["verdict"] or "none", 0) + 1
            g["gold"][it["newest"]["label"]] = g["gold"].get(it["newest"]["label"], 0) + 1
    return {"items": n, "pool_composition": composition(pool_c), "top8_composition": composition(top_c),
            "items_with_sr_ma_in_top8": _share(sr_top8, n),
            "items_with_sr_ma_in_b1_admitted": _share(sr_b1, n),
            "study_type_weight_has_material": (sr_top8 / n >= MIN_SR_SHARE) if n else None,
            "b1_by_sr_ma": groups}


def run_diagnostics(items: dict, pools: dict, answers: dict,
                    count_tokens: Optional[Callable[[str], int]]) -> dict:
    return {"helpfulness_inputs": (helpfulness_truncation(items, pools, answers, count_tokens)
                                   if count_tokens else {"skipped": "no tokenizer"}),
            "stance_text": snippet_stats(items, pools),
            "study_types": study_type_stats(items, pools, answers)}


def to_markdown(split: str, rep: dict) -> str:
    h, s, t = rep["helpfulness_inputs"], rep["stance_text"], rep["study_types"]
    lines = [f"# P0 diagnostics, {split} pools", "", "## 1. Stage-1 helpfulness inputs (question after the abstract, cut at 512 tokens)", ""]
    if "skipped" in h:
        lines.append("Skipped (no tokenizer).")
    else:
        lines += [f"* {h['over_limit']} of {h['inputs']} inputs ({100 * h['share_over_limit']:.1f}%) exceed "
                  f"{h['limit_tokens']} tokens; median {h['median_tokens']}.",
                  f"* Among B2's admitted papers: {h['b2_admitted_over_limit']} of {h['b2_admitted_papers']} "
                  f"({100 * (h['b2_admitted_share_over_limit'] or 0):.1f}%) were over the limit.",
                  "* Any input over the limit lost the question and the \"Answer yes or no\" line, so its helpfulness score is not "
                  "a judgement of that question. Stage-1 conclusions about recency are unaffected (they compare arms sharing "
                  "the same scores); the claim that the helpfulness score is a weak selector is confounded by this."]
    lines += ["", "## 2. What the stance step would read", "",
              f"* {100 * (s['with_results_or_conclusions'] or 0):.1f}% of {s['candidates']} candidates have labelled RESULTS or CONCLUSIONS sections "
              f"(the rest contribute their last three sentences); median {s['snippet_words_median']} words, maximum {s['snippet_words_max']}.",
              "", "## 3. Study types", "",
              f"* Pool: {t['pool_composition']}; top 8: {t['top8_composition']}.",
              f"* Items with a systematic review or meta-analysis in the top 8: {100 * (t['items_with_sr_ma_in_top8'] or 0):.1f}%; "
              f"in B1's admitted five: {100 * (t['items_with_sr_ma_in_b1_admitted'] or 0):.1f}%.",
              f"* Study-type weight has material to act on (>= {int(100 * MIN_SR_SHARE)}% of items): "
              f"**{'yes' if t['study_type_weight_has_material'] else 'no'}**.", "",
              "| B1 evidence | items | B1 verdicts | gold verdicts |", "|---|---|---|---|"]
    for name, g in t["b1_by_sr_ma"].items():
        lines.append(f"| {name.replace('_', ' ')} | {g['n']} | {g['b1_verdicts']} | {g['gold']} |")
    return "\n".join(lines) + "\n"


def _tokenizer_counter() -> Optional[Callable[[str], int]]:
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MODEL)
    except Exception as exc:                                    # offline, not installed, ...
        print(f"tokenizer unavailable ({exc}); skipping the truncation check", file=sys.stderr)
        return None
    return lambda text: len(tok(text)["input_ids"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--split", default="dev", choices=("dev", "confirm"))
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--no-tokenizer", action="store_true")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    data = Path(args.data_dir)
    items = {r["item_id"]: r for r in load_jsonl(data / "benchmark.jsonl")
             if r["split"] == args.split and not r["likely_label_noise"]}
    pools = {r["item_id"]: r for r in load_jsonl(data / f"frozen_{args.split}.jsonl")}
    answers = {(r["item_id"], r["arm"]): r for r in load_jsonl(data / f"answers_{args.split}.jsonl")}
    if not pools:
        print("no frozen pools yet", file=sys.stderr)
        return 2
    counter = None if args.no_tokenizer else _tokenizer_counter()
    rep = run_diagnostics(items, pools, answers, counter)
    text = to_markdown(args.split, rep)
    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, body in ((f"diagnostics_{args.split}.json", json.dumps(rep, indent=2) + "\n"),
                           (f"diagnostics_{args.split}.md", text)):
            with open(out / name, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(body)
        print(f"wrote {out / f'diagnostics_{args.split}.md'}")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

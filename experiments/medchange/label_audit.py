"""Label audit: how reproducible are the gold labels? (automatic; no human labelling)

The gold verdicts are gpt-4o-mini's labels of each Cochrane review's authors' conclusions, produced
with the authors' public labelling rubric (``Code/generate_questions_labels.ipynb`` of the MedChange
release). Nobody here can certify them medically, so this script measures what can be measured: an
INDEPENDENT local model (a different family from the generator, e.g. Qwen2.5-7B-Instruct) re-labels
every item from the same text with the same rubric, and we report agreement, Cohen's kappa, per-class
agreement and whether a label change between versions is reproduced. The items on which the two
labelers agree on the newest version are "label-stable"; the realigned analysis also reports its result
on that subset (descriptive; the primary analysis keeps all items). Agreement is label
*reproducibility*, not medical truth: two language models can share a bias.

    python -m experiments.medchange.label_audit --split dev --medchange-dir ..\\MedChange --model-path models\\Qwen2.5-7B-Instruct-Q4_K_M.gguf

The confirmatory split is audited only after the realigned design record exists (so its labels are
not looked at before the design is frozen).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence

from .benchmark import LABELS, read_csv
from .generate_answers import check_config, config_path, file_sha256, llama_generator, llama_version, load_jsonl

HERE = Path(__file__).resolve().parent

SYSTEM = "You're a helpful assistant. Your task is to help with labelling in the medical and clinical domain."
# The label definitions are the authors' own (generate_questions_labels.ipynb), applied to a given question.
TEMPLATE = (
    "You will be given the objectives and the author's conclusions of a clinical systematic review, and a "
    "question answerable with yes/no/maybe that sums up its main medical objective.\n\n"
    "Give a label for the author's conclusions. The label tries to answer the question by looking at the "
    "conclusions. The label may be ONLY one of the following three: (1) SUPPORTED; (2) REFUTED; "
    "(3) NOT ENOUGH INFORMATION. Do not make up a new label. Please only select the third label if not enough "
    "studies were found, not if the certainty of the conclusion is low! Please strive to predict REFUTED or "
    "SUPPORTED even if the certainty of these conclusions by the authors is low or weak! Label SUPPORTED means "
    "the hypothesis is at least partially supported by the conclusions, label REFUTED means it is at least "
    "partially not supported or similar to placebo.\n\n"
    "Answer with exactly one line: LABEL: <label>\n\n"
    "QUESTION: {question}\nOBJECTIVES: {objectives}\nAUTHOR'S CONCLUSIONS: {conclusions}")

_LABEL = re.compile(r"LABEL\s*:\s*\**\s*(NOT\s+ENOUGH\s+INFORMATION|SUPPORTED|REFUTED)", re.IGNORECASE)


def build_prompt(question: str, objectives: str, conclusions: str) -> str:
    return TEMPLATE.format(question=question.strip(), objectives=(objectives or "").strip(),
                           conclusions=(conclusions or "").strip())


def parse_label(text: str) -> Optional[str]:
    m = _LABEL.search(text or "")
    return re.sub(r"\s+", " ", m.group(1).upper()) if m else None


def cohens_kappa(pairs: Sequence[tuple[str, str]]) -> Optional[float]:
    if not pairs:
        return None
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    expected = sum((sum(a == c for a, _ in pairs) / n) * (sum(b == c for _, b in pairs) / n) for c in LABELS)
    return None if expected == 1 else round((observed - expected) / (1 - expected), 4)


def tasks(items: Sequence[dict]) -> list[tuple[dict, str]]:
    """(item, "newest" | "previous"): the newest version always, the older one for changed items."""
    out = []
    for it in items:
        out.append((it, "newest"))
        if it["kind"] == "changed":
            out.append((it, "previous"))
    return out


def run(items: Sequence[dict], medrev: dict, generate: Callable[[str, str], str], out: Path,
        on_progress: Callable[[int, int], None] = lambda a, b: None) -> int:
    done = {(r["item_id"], r["which"]) for r in load_jsonl(out)}
    todo = [(it, w) for it, w in tasks(items) if (it["item_id"], w) not in done]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8", newline="\n") as handle:
        for k, (it, which) in enumerate(todo, 1):
            row = medrev[it[which]["row"]]
            prompt = build_prompt(it["question"], row.get("objectives", ""), row.get("conclusions", ""))
            raw = generate(SYSTEM, prompt)
            handle.write(json.dumps({"item_id": it["item_id"], "which": which, "gold": it[which]["label"],
                                     "relabel": parse_label(raw), "raw": (raw or "")[:60],
                                     "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()[:16]}) + "\n")
            handle.flush()
            on_progress(k, len(todo))
    return len(todo)


def summarize(rows: Sequence[dict], items: dict) -> dict:
    newest = [r for r in rows if r["which"] == "newest" and r["item_id"] in items]
    valid = [r for r in newest if r["relabel"]]
    pairs = [(r["gold"], r["relabel"]) for r in valid]
    per_class = {}
    for lab in LABELS:
        sel = [r for r in valid if r["gold"] == lab]
        per_class[lab] = {"n": len(sel), "agreement": round(sum(r["relabel"] == lab for r in sel) / len(sel), 4) if sel else None}
    changed = {}
    for r in rows:
        if r["item_id"] in items and items[r["item_id"]]["kind"] == "changed" and r["relabel"]:
            changed.setdefault(r["item_id"], {})[r["which"]] = r["relabel"]
    both = [v for v in changed.values() if len(v) == 2]
    stable = sorted(r["item_id"] for r in valid if r["relabel"] == r["gold"])
    return {"n_items": len(newest), "n_unparsed": len(newest) - len(valid),
            "agreement": round(sum(a == b for a, b in pairs) / len(pairs), 4) if pairs else None,
            "kappa": cohens_kappa(pairs), "per_gold_class": per_class,
            "confusion": {f"{g}->{p}": sum(1 for a, b in pairs if a == g and b == p) for g in LABELS for p in LABELS},
            "changed_items_both_versions_labelled": len(both),
            "label_change_reproduced": round(sum(v["newest"] != v["previous"] for v in both) / len(both), 4) if both else None,
            "n_stable": len(stable), "stable_item_ids": stable}


def to_markdown(split: str, rep: dict, model: str) -> str:
    pct = lambda x: "n/a" if x is None else f"{100 * x:.1f}%"
    lines = [f"# Label audit, {split} split", "",
             f"Independent labeler: {model}; the authors' rubric; {rep['n_items']} items "
             f"({rep['n_unparsed']} unparsed).", "",
             f"* Agreement with the gold label (newest version): **{pct(rep['agreement'])}**; Cohen's kappa "
             f"**{rep['kappa']}**.",
             f"* Label change between versions reproduced for {pct(rep['label_change_reproduced'])} of "
             f"{rep['changed_items_both_versions_labelled']} changed items.",
             f"* Label-stable items (both labelers agree): {rep['n_stable']}.", "",
             "| Gold class | items | agreement |", "|---|---|---|"]
    lines += [f"| {k} | {v['n']} | {pct(v['agreement'])} |" for k, v in rep["per_gold_class"].items()]
    lines += ["", "Agreement measures how reproducible the labels are with another model; it is not medical truth, "
              "and two models can share a bias."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--split", default="dev", choices=("dev", "confirm", "ad"))
    ap.add_argument("--medchange-dir", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--out-dir", default=str(HERE / "results"))
    ap.add_argument("--design-record", default=str(HERE / "results" / "rag2_design.json"))
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--n-threads", type=int, default=None)
    args = ap.parse_args(argv)
    if args.split in ("confirm", "ad") and not Path(args.design_record).is_file():
        print("the confirmatory labels are audited only after the design is frozen "
              f"({args.design_record} not found)", file=sys.stderr)
        return 2
    if not Path(args.model_path).is_file():
        print(f"model file not found: {args.model_path}", file=sys.stderr)
        return 2
    items = [r for r in load_jsonl(Path(args.data_dir) / "benchmark.jsonl")
             if r["split"] == args.split and not r["likely_label_noise"]]
    medrev = {int(r[""]): r for r in read_csv(Path(args.medchange_dir) / "Datasets" / "MedRevQA.csv")}
    out = Path(args.data_dir) / f"label_audit_{args.split}.jsonl"
    sha = file_sha256(args.model_path)
    differs = check_config(out, {"model_sha256": sha, "template_sha256": hashlib.sha256(TEMPLATE.encode()).hexdigest(),
                                 "system_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(), "n_ctx": args.n_ctx,
                                 "llama_cpp_python": llama_version()},
                           ("model_sha256", "template_sha256", "system_sha256", "n_ctx"))
    if differs:
        print(f"refusing to extend {out.name}: configuration differs in {', '.join(differs)} "
              f"(see {config_path(out).name})", file=sys.stderr)
        return 2
    gen = llama_generator(args.model_path, args.n_ctx, args.n_threads, 0, 12)
    run(items, medrev, gen, out, lambda k, t: print(f"  {k}/{t}", flush=True) if k % 25 == 0 or k == t else None)
    rep = summarize(load_jsonl(out), {i["item_id"]: i for i in items})
    rep.update(split=args.split, labeler=Path(args.model_path).name, labeler_sha256=sha)
    od = Path(args.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    (od / f"label_audit_{args.split}.json").write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8", newline="\n")
    text = to_markdown(args.split, rep, Path(args.model_path).name)
    (od / f"label_audit_{args.split}.md").write_text(text, encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

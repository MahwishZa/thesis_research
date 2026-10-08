"""Paper-style result tables and figures for the thesis (no new experiments are run).

Turns the committed results of one split into the kind of tables the base paper (RAG², Sohn et al.,
NAACL 2025) uses: systems against a benchmark (their Table 2), and one generator with different
evidence-admission methods (their Table 3), plus the figures that go with them. The benchmark here is
MedChange (verdict accuracy against the newest Cochrane verdict), not multiple-choice QA, so the numbers
are not comparable with theirs; only the layout is.

    python -m experiments.medchange.report --split confirm --medchange-dir ..\\MedChange

Reads ``benchmark.jsonl`` (gold labels), ``results/answers_<split>.jsonl`` (B0, B1 and the other stage-1 arms),
``results/rag2_answers_<split>.jsonl`` (R2, R2C, R2V, R2V-ND), the label audit and, for the closed-book rows, the
benchmark authors' released answers (``Code/GeneratedAnswers`` of their repository). Writes ``results/report/REPORT.md``,
``tables.tex``, ``report_data_<split>.json`` and two PNG figures (matplotlib, optional: ``pip install -e ".[report]"``).
Every number is computed from those files; nothing is typed in. Only the confirmatory split supports claims
(``docs/evaluation.md``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

from .benchmark import LABELS
from .generate_answers import load_jsonl
from .headroom import MODELS, parse_label
from .scoring import class_stats, correct_map, load_answers, paired, wilson

HERE = Path(__file__).resolve().parent
RELEASED = {"qwen25-7b": "Qwen2.5-7B", "mistral-24b": "Mistral-24B", "llama33-70b": "Llama-3.3-70B",
            "gpt4o-mini": "GPT-4o-mini", "deepsek-v3": "DeepSeek-V3"}      # file names of the release
STAGE1 = (("B0", "Llama-3-8B-Instruct (Q4_K_M), no retrieval"),
          ("B1", "+ MedCPT retrieval, top-5 (standard RAG)"),
          ("B2", "+ zero-shot helpfulness filter (untrained RAG²-style stand-in)"),
          ("B3", "+ recency re-ranking (TempRALM-style)"),
          ("P", "+ helpfulness + recency (stage-1 Temporal Filter)"),
          ("C1", "+ helpfulness + shuffled dates (control)"))
REALIGNED = (("R2", "adapted RAG²: rationale query + type-balanced retrieval + LLM filter (baseline)"),
             ("R2C", "R2's evidence read once with explicit evidence criteria (control)"),
             ("R2V", "R2's answer verified against the evidence criteria (proposed)"),
             ("R2V-ND", "R2V without dates and without the currency criterion (ablation)"))
REALIGNED_GROUP = "Adapted RAG² and evidence-criteria verification (this thesis, realigned study)"
KINDS = {"changed": ("changed",), "unchanged": ("unchanged",), "all": ("changed", "unchanged")}


# --------------------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------------------

def released_verdicts(items: dict, medchange_dir: Path) -> dict:
    """{(item_id, "R:<model>"): {"verdict": ...}} from the authors' released closed-book answers."""
    out = {}
    folder = Path(medchange_dir) / "Code" / "GeneratedAnswers"
    for model in MODELS:
        path = folder / f"{model}_answers.txt"
        if not path.is_file():
            continue
        preds = [parse_label(x) for x in path.read_text(encoding="utf-8", errors="replace").split("\n")]
        for item_id, it in items.items():
            row = it["newest"]["row"]
            if row < len(preds):
                out[(item_id, f"R:{model}")] = {"verdict": preds[row]}
    return out


def accuracy_cell(items: dict, answers: dict, arm: str, kinds: Sequence[str]) -> Optional[dict]:
    c = correct_map(items, answers, arm, kinds)
    if not c:
        return None
    k, n = sum(c.values()), len(c)
    return {"n": n, "correct": k, "accuracy": k / n, "ci": wilson(k, n)}


def system_rows(items: dict, answers: dict, arms_present: Sequence[str], released: Sequence[str]) -> list[dict]:
    """The Table-2-style rows: group, name and the accuracy cells for changed / unchanged / all items."""
    rows = []

    def add(group, name, arm):
        cells = {kind: accuracy_cell(items, answers, arm, kinds) for kind, kinds in KINDS.items()}
        if cells["all"]:
            rows.append({"group": group, "arm": arm, "name": name, **cells})

    for model in released:
        add("Closed-book LLMs, released answers (existing work)", RELEASED.get(model, model), f"R:{model}")
    for arm, name in STAGE1:
        if arm in arms_present:
            add("Llama-3-8B-Instruct, local (this thesis, stage 1)", f"{arm}: {name}", arm)
    for arm, name in REALIGNED:
        if arm in arms_present:
            add(REALIGNED_GROUP, f"{arm}: {name}", arm)
    return rows


def filtering_rows(items: dict, answers: dict, arms_present: Sequence[str], baseline: str = "B1") -> list[dict]:
    """The Table-3-style rows: one generator, different admission methods, paired against the baseline."""
    rows = []
    for arm, name in STAGE1 + REALIGNED:
        if arm not in arms_present or arm == "B0":
            continue
        cell = accuracy_cell(items, answers, arm, ("changed",))
        unch = accuracy_cell(items, answers, arm, ("unchanged",))
        diff = None if arm == baseline else paired(items, answers, arm, baseline, ("changed",))
        rows.append({"arm": arm, "name": name, "changed": cell, "unchanged": unch, "diff_vs_baseline": diff})
    return rows


def class_rows(items: dict, answers: dict, arms_present: Sequence[str]) -> list[dict]:
    rows = []
    for arm, name in (("B0", "no retrieval"),) + tuple(STAGE1[1:]) + REALIGNED:
        ids = [i for i, it in items.items() if it["kind"] == "changed" and (i, arm) in answers]
        if arm not in arms_present or not ids:
            continue
        s = class_stats(items, answers, arm, ("changed",))
        nei = sum(answers[(i, arm)]["verdict"] == "NOT ENOUGH INFORMATION" for i in ids) / len(ids) if ids else None
        rows.append({"arm": arm, "name": name, "recall": s["recall"], "macro_f1": s["macro_f1"], "nei_share": nei})
    return rows


def constant_baselines(items: dict) -> dict:
    out = {}
    for kind, kinds in KINDS.items():
        ids = [i for i, it in items.items() if it["kind"] in kinds]
        if ids:
            out[kind] = {lab: sum(items[i]["newest"]["label"] == lab for i in ids) / len(ids) for lab in LABELS}
    return out


# --------------------------------------------------------------------------------------
# Text
# --------------------------------------------------------------------------------------

def _pct(x: Optional[float], digits: int = 1) -> str:
    return "n/a" if x is None else f"{100 * x:.{digits}f}"


def _cell(c: Optional[dict], bold: bool = False) -> str:
    if not c:
        return "n/a"
    text = _pct(c["accuracy"])
    return f"**{text}**" if bold else text


def _best(rows: list[dict], kind: str) -> Optional[float]:
    vals = [r[kind]["accuracy"] for r in rows if r.get(kind)]
    return max(vals) if vals else None


def table_systems(rows: list[dict], n: dict) -> str:
    """Markdown in the layout of the base paper's Table 2 (best value per column in bold)."""
    best = {k: _best(rows, k) for k in KINDS}
    lines = [f"| System | Changed (n = {n['changed']}) | Unchanged (n = {n['unchanged']}) | All (n = {n['all']}) | 95% CI, all |",
             "|---|---|---|---|---|"]
    group = None
    for r in rows:
        if r["group"] != group:
            group = r["group"]
            lines.append(f"| *{group}* | | | | |")
        cells = [_cell(r[k], bool(r.get(k)) and r[k]["accuracy"] == best[k]) for k in KINDS]
        ci = r["all"]["ci"]
        lines.append(f"| {r['name']} | {cells[0]} | {cells[1]} | {cells[2]} | {_pct(ci[0])}–{_pct(ci[1])} |")
    return "\n".join(lines) + "\n"


def table_filtering(rows: list[dict]) -> str:
    """Markdown in the layout of the base paper's Table 3: one generator, different admission methods."""
    lines = ["| Method (top-5 passages) | Changed | Δ vs B1 (pp) | 95% CI (pp) | exact McNemar p | Unchanged |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        d = r["diff_vs_baseline"]
        diff = ["—", "—", "—"] if d is None else [f"{100 * d['diff_a_minus_b']:+.1f}",
                                                     f"{100 * d['ci95'][0]:+.1f} to {100 * d['ci95'][1]:+.1f}",
                                                     f"{d['mcnemar_p']:.4f}"]
        lines.append(f"| {r['arm']}: {r['name']} | {_cell(r['changed'])} | {diff[0]} | {diff[1]} | {diff[2]} | "
                     f"{_cell(r['unchanged'])} |")
    return "\n".join(lines) + "\n"


def table_classes(rows: list[dict]) -> str:
    lines = ["| Arm | recall SUPPORTED | recall REFUTED | recall NOT ENOUGH INFORMATION | macro-F1 (%) | answers NOT ENOUGH INFORMATION |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        rc = r["recall"]
        lines.append(f"| {r['arm']}: {r['name']} | {_pct(rc['SUPPORTED'], 0)}% | {_pct(rc['REFUTED'], 0)}% | "
                     f"{_pct(rc['NOT ENOUGH INFORMATION'], 0)}% | {_pct(r['macro_f1'])} | {_pct(r['nei_share'], 0)}% |")
    return "\n".join(lines) + "\n"




def to_latex(rows: list[dict], caption: str, label: str) -> str:
    """The systems table as a booktabs table for the thesis document."""
    best = {k: _best(rows, k) for k in KINDS}
    out = ["\\begin{table}[t]", "\\centering", "\\small", "\\begin{tabular}{lrrr}", "\\toprule",
           "System & Changed & Unchanged & All \\\\", "\\midrule"]
    group = None
    for r in rows:
        if r["group"] != group:
            group = r["group"]
            out.append(f"\\multicolumn{{4}}{{l}}{{\\emph{{{_tex(group)}}}}} \\\\")
        cells = []
        for k in KINDS:
            c = r[k]
            text = _pct(c["accuracy"]) if c else "n/a"
            cells.append(f"\\textbf{{{text}}}" if c and c["accuracy"] == best[k] else text)
        out.append(f"{_tex(r['name'])} & {' & '.join(cells)} \\\\")
    out += ["\\bottomrule", "\\end{tabular}", f"\\caption{{{_tex(caption)}}}", f"\\label{{{label}}}", "\\end{table}"]
    return "\n".join(out) + "\n"


def _tex(text: str) -> str:
    for a, b in (("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("²", "$^2$"), ("Δ", "$\\Delta$"), ("–", "--")):
        text = text.replace(a, b)
    return text


# --------------------------------------------------------------------------------------
# Figures (matplotlib is optional)
# --------------------------------------------------------------------------------------

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e5e4e0", "#fcfcfb"
VIOLET = "#7b5cc4"                                                           # fourth category (not part of the validated three)
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9993"      # validated categorical slots 1-3 (+ neutral)


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "text.color": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 9, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left"})
    return plt


SHORT = {"B0": "B0 no retrieval", "B1": "B1 + MedCPT top-5", "B2": "B2 + helpfulness filter", "B3": "B3 + recency re-rank",
         "P": "P helpfulness + recency", "C1": "C1 shuffled-date control",
         "R2": "R2 adapted RAG² (baseline)", "R2C": "R2C criteria only", "R2V": "R2V verification (proposed)",
         "R2V-ND": "R2V-ND no dates"}
GROUP_COLOURS = {"Closed-book": BLUE, "Llama-3-8B": ORANGE, "Adapted RAG²": VIOLET}


def _colour(group: str) -> str:
    return next((c for k, c in GROUP_COLOURS.items() if group.startswith(k)), GRAY)


def _label(row: dict) -> str:
    return SHORT.get(row["arm"], row["name"])


def _style(ax, axis="x"):
    getattr(ax, f"{axis}axis").grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def figure_systems(rows: list[dict], const: dict, path: Path, status: str) -> bool:
    plt = _plt()
    if plt is None or not rows:
        return False
    fig, ax = plt.subplots(figsize=(9.0, 0.30 * len(rows) + 2.4))
    ys = list(range(len(rows)))[::-1]
    for y, r in zip(ys, rows):
        c = r["changed"]
        if not c:
            continue
        ax.barh(y, 100 * c["accuracy"], height=0.56, color=_colour(r["group"]))
        ax.plot([100 * c["ci"][0], 100 * c["ci"][1]], [y, y], color=INK, lw=0.9, solid_capstyle="butt")
        ax.text(79, y, f"{100 * c['accuracy']:.1f}", va="center", ha="right", fontsize=8, color=INK)
    ref = const.get("changed", {}).get("SUPPORTED")
    if ref is not None:
        ax.axvline(100 * ref, color=MUTED, lw=1, ls=(0, (4, 3)))
        ax.text(100 * ref + 0.5, len(rows) - 0.45, "always answering SUPPORTED", fontsize=8, color=MUTED, va="bottom")
    ax.set_yticks(ys)
    ax.set_yticklabels([_label(r) for r in rows], fontsize=8)
    ax.set_xlim(0, 80)
    ax.set_ylim(-0.7, len(rows) - 0.1)
    ax.set_xlabel("Verdict accuracy on changed items (%); line = 95% Wilson interval")
    ax.set_title(f"Accuracy by system ({status})")
    _style(ax)
    groups = list(dict.fromkeys(r["group"] for r in rows))
    handles = [plt.Rectangle((0, 0), 1, 1, color=_colour(g)) for g in groups]
    fig.legend(handles, groups, loc="lower center", fontsize=7.5, frameon=False, ncol=1)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True


def figure_classes(rows: list[dict], path: Path, status: str) -> bool:
    plt = _plt()
    if plt is None or not rows:
        return False
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    width = 0.26
    for j, (lab, colour) in enumerate(zip(LABELS, (BLUE, ORANGE, AQUA))):
        xs = [i + (j - 1) * width for i in range(len(rows))]
        vals = [100 * (r["recall"][lab] or 0) for r in rows]
        ax.bar(xs, vals, width=width * 0.9, color=colour,
               label=lab.capitalize() if lab != "NOT ENOUGH INFORMATION" else "Not enough information")
        for x, v in zip(xs, vals):
            ax.text(x, v + 1, f"{v:.0f}", ha="center", fontsize=7, color=MUTED)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r["arm"] for r in rows])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Recall of the gold class (%), changed items")
    ax.set_title(f"What each system gets right ({status})")
    _style(ax, "y")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return True






# --------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------

def build(split: str, items: dict, answers: dict, released: Sequence[str], audit: Optional[dict]) -> dict:
    present = sorted({arm for (_, arm) in answers if not arm.startswith("R:")})
    n = {k: sum(it["kind"] in kinds for it in items.values()) for k, kinds in KINDS.items()}
    return {"split": split, "status": "confirmatory" if split == "confirm" else "exploratory", "n": n,
            "systems": system_rows(items, answers, present, released),
            "filtering": filtering_rows(items, answers, present), "classes": class_rows(items, answers, present),
            "constant": constant_baselines(items),
            "label_audit": {k: audit[k] for k in ("agreement", "kappa", "n_items", "n_stable", "label_change_reproduced",
                                                  "per_gold_class")} if audit else None}


def report_markdown(d: dict, figures: dict[str, bool]) -> str:
    status = d["status"]
    lines = [f"# Results tables and figures: {d['split']} split, {status}", "",
             "Generated by `python -m experiments.medchange.report` from the committed results; nothing here is typed in. "
             "Layout follows the base paper (RAG², Sohn et al., NAACL 2025): their Table 2 (systems × benchmark) and "
             "Table 3 (one generator, different filtering methods). **The numbers are not comparable with theirs**: this is "
             "verdict accuracy (SUPPORTED / REFUTED / NOT ENOUGH INFORMATION against the newest Cochrane review's label), "
             "not multiple-choice accuracy, on much smaller samples.", ""]
    if d["split"] == "dev":
        lines += ["> **Dev split, exploratory.** These 226 questions were used to check the pipeline and the design; "
                  "intervals are wide (about ±6 to ±8 points) and nothing here is a confirmatory finding. The confirmatory "
                  "split (528 questions) is run once, after the design is frozen (`docs/protocol.md`).", ""]
    n = d["n"]
    groups = {r["group"][:6] for r in d["systems"]}
    arms = {r["arm"] for r in d["systems"]}
    note = []
    if "Closed" in groups:
        note.append("Closed-book rows are the benchmark authors' own released answers (their prompt, no retrieval) scored on the "
                    "same questions.")
    note.append("Local rows: Meta-Llama-3-8B-Instruct Q4_K_M on CPU, one prompt, greedy decoding.")
    if arms & {"R2", "R2V"}:
        note.append("R2 is an adapted RAG², not a reproduction (docs/methodology.md). The requirement is read on R2V − R2 only "
                    "(RAG2_FINDINGS.md: the difference, its interval and the pre-declared reading); differences between other rows "
                    "of this table are descriptive, and none is confirmed. The local 8B systems score at or below several "
                    "closed-book models on these questions, which is context, not a controlled comparison.")
    note.append("Bold: best per column.")
    if arms & {"B2", "P", "C1"}:
        note.append("B2/P/C1 use an untrained zero-shot Flan-T5 helpfulness score, **not** RAG²'s trained filter (its checkpoint is "
                    "not distributed); for 18.5% of its inputs the question was cut off by truncation (`docs/log.md`, Phase 28).")
    lines += ["## Table 1. Verdict accuracy (%) of LLMs and RAG variants", "", table_systems(d["systems"], n), " ".join(note), ""]
    if figures.get("fig1"):
        lines += ["![Accuracy by system](fig1_accuracy_by_system.png)", ""]
    if len(d["filtering"]) > 1:
        lines += ["## Table 2. One generator, different evidence-admission methods", "", table_filtering(d["filtering"]),
                  "Differences are against B1 on changed items; p values are uncorrected exact McNemar tests. The pre-stated "
                  f"Holm-corrected comparisons of the realigned study are in `rag2_analysis_{d['split']}.md`.", ""]
    if d["classes"]:
        lines += ["## Table 3. Where the accuracy comes from (changed items)", "", table_classes(d["classes"]), ""]
        if figures.get("fig2"):
            lines += ["![Per-class recall](fig2_class_recall.png)", ""]
    if d.get("label_audit"):
        a = d["label_audit"]
        lines += ["## Label reproducibility", "", f"An independent model (Qwen2.5-7B-Instruct) re-labelled the gold labels with the authors' "
                  f"rubric: agreement {_pct(a['agreement'])}%, kappa {a['kappa']}, {a['n_stable']} of {a['n_items']} items label-stable; "
                  f"{_pct(a['label_change_reproduced'])}% of label changes between versions reproduced. This is reproducibility, not "
                  "medical truth, and it bounds what any accuracy here can mean.", ""]
    lines += ["## Relation to the base paper", "",
              "RAG² reports +6.9 points on MedQA for Llama-3-8B-Instruct (57.7 → 64.6) using a Flan-T5 filter trained on "
              "perplexity-based labels, rationale queries and balanced retrieval over a 564 GB index of four corpora, trained on one "
              "H100 GPU. None of that is reproducible here: the trained checkpoint is not distributed, a local retraining attempt "
              "(removed from the repository; `docs/log.md` Phases 13–18) learned only the class prior, and the thesis runs on a CPU laptop with a different task (as-of verdict "
              "accuracy). What is comparable is the experimental design, a fixed generator with different evidence-admission methods "
              "(Table 2) and the comparison with existing systems on identical inputs (Table 1). The closed-book rows are other "
              "models answering the same questions without retrieval, so they put the local 8B system in context; they are not a "
              "controlled comparison (different sizes, training and prompts). Recency-aware retrieval (TempRALM) corresponds to "
              "the stage-1 rows B3 and P.", ""]
    return "\n".join(lines)


def write_report(d: dict, out: Path, make_figures: bool = True) -> dict[str, bool]:
    out.mkdir(parents=True, exist_ok=True)
    status = d["status"]
    figures = {"fig1": False, "fig2": False}
    if make_figures:
        figures["fig1"] = figure_systems(d["systems"], d["constant"], out / "fig1_accuracy_by_system.png", status)
        figures["fig2"] = figure_classes(d["classes"], out / "fig2_class_recall.png", status)
    (out / "REPORT.md").write_text(report_markdown(d, figures), encoding="utf-8", newline="\n")
    (out / "tables.tex").write_text(
        to_latex(d["systems"], f"Verdict accuracy (\\%) on the MedChange {d['split']} split ({status}); not comparable with "
                 "multiple-choice benchmarks.", f"tab:systems-{d['split']}"), encoding="utf-8", newline="\n")
    (out / f"report_data_{d['split']}.json").write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8", newline="\n")
    return figures


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--split", default="dev", choices=("dev", "confirm"))
    ap.add_argument("--medchange-dir", default=None, help="clone of jvladika/MedChange (closed-book rows)")
    ap.add_argument("--data-dir", default=str(HERE / "data"))
    ap.add_argument("--results-dir", default=str(HERE / "results"))
    ap.add_argument("--out-dir", default=None, help="default: <results-dir>/report")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    data, results = Path(args.data_dir), Path(args.results_dir)
    items = {r["item_id"]: r for r in load_jsonl(data / "benchmark.jsonl")
             if r["split"] == args.split and not r["likely_label_noise"]}
    if not items:
        print("no benchmark items: run build_benchmark first", file=sys.stderr)
        return 2
    answers = load_answers(results, args.split)
    answers.update({(r["item_id"], r["arm"]): r for r in load_jsonl(results / f"rag2_answers_{args.split}.jsonl")
                    if r["arm"] in dict(REALIGNED)})
    if not answers:
        print(f"no answers for the {args.split} split in {results}", file=sys.stderr)
        return 2
    released: list[str] = []
    if args.medchange_dir:
        extra = released_verdicts(items, Path(args.medchange_dir))
        answers.update(extra)
        first = next(iter(items))
        released = [m for m in MODELS if (first, f"R:{m}") in extra]
    elif args.split == "dev":
        print("note: no --medchange-dir, so the closed-book rows are left out", file=sys.stderr)

    audit_path = results / f"label_audit_{args.split}.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else None
    d = build(args.split, items, answers, released, audit)
    figures = write_report(d, Path(args.out_dir) if args.out_dir else results / "report", not args.no_figures)
    print(f"wrote the report for the {args.split} split; figures: " + ", ".join(k for k, v in figures.items() if v))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""The result tables of docs/results.md, filled from the finished test runs in results/ (no model is asked, nothing is written unless
``--output`` is given):

    python -m src.evaluate tables [--output FILE]

Every score is computed from a run's saved answers (``evaluate.score_run``), never read from an old metrics.json, and every
comparison is ``evaluate.compare``. A model or dataset without a finished test run leaves its cells empty; ``n/a`` means the score
cannot be computed (for example the selective risk of a model that never marks its evidence "sufficient"). Computing every comparison
takes a minute or two.
"""

from __future__ import annotations

from pathlib import Path

from .datasets import DATASETS
from .evaluate import RESULTS, compare, score_run

APPROACH = "closed_book"
MODELS = (("claude-sonnet-5-5", "Claude Sonnet 5.5"), ("gpt-4o", "GPT-4o"), ("gemini-3.8-flash", "Gemini 3.8 Flash"),
          ("llama-4-maverick", "Llama 4 Maverick"))  # the order of every table; the first model is the reference of section 6.2
SEES = {"published_text": "the case text", "published_text_photos": "the case text and the article's photos",
        "published_photos": "the article's photos only", "clinic_record": "the clinic record only", "clinic_photo": "the phone photo only",
        "clinic_photo_record": "the phone photo and the record"}  # in the order of every table
PAIRS = (("published_text", "published_text_photos"), ("published_text_photos", "published_photos"), ("published_text", "published_photos"),
         ("clinic_record", "clinic_photo"), ("clinic_photo", "clinic_photo_record"), ("clinic_record", "clinic_photo_record"))  # A, B


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def span(ci) -> str:
    return f"{100 * ci[0]:.1f} to {100 * ci[1]:.1f}"


def signed(x: float) -> str:
    return f"{100 * x:+.1f}"


def p_value(p: float) -> str:
    return "<0.0001" if p < 0.0001 else f"{p:.4f}"


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header), *("| " + " | ".join(row) + " |" for row in rows)]


class Runs:
    """The finished test runs under ``results``, scored on first use."""

    def __init__(self, results: Path = RESULTS):
        self.results, self._scored, self._known = Path(results), {}, {}

    def folder(self, dataset: str, model: str) -> Path | None:
        """The finished test run of a model on a dataset, or None. A run still in progress (.partial) is not one; two finished test
        runs are an error (a run is made once per model, dataset and approach)."""
        folders = sorted(f for f in (self.results / dataset / APPROACH / model).glob("test_*") if f.is_dir() and not f.name.endswith(".partial"))
        if len(folders) > 1:
            raise ValueError(f"{dataset} / {model}: more than one finished test run ({', '.join(f.name for f in folders)}); keep one")
        return folders[0] if folders else None

    def scored(self, dataset: str, model: str):
        """(folder, run card, scores) of a finished run, or None."""
        if (dataset, model) not in self._scored:
            folder = self.folder(dataset, model)
            self._scored[dataset, model] = None
            if folder is not None:
                card, _, metrics = score_run(folder)
                if card["status"] != "complete" or card["split"] != "test":
                    raise ValueError(f"{folder.as_posix()} is not a finished test run")
                self._scored[dataset, model] = (folder, card, metrics)
        return self._scored[dataset, model]

    def questions(self, dataset: str) -> int | None:
        """How many test questions a dataset has, as far as any finished run shows."""
        for model, _ in MODELS:
            found = self.scored(dataset, model)
            if found:
                return found[2]["main"]["n"]
        return None


def _risk(selective: dict) -> str:
    covered = selective["covered"]
    return f"{pct(selective['selective_risk'])} ({selective['covered_wrong']} of {covered})" if covered else "n/a"


def overview(runs: Runs) -> list[str]:
    rows = []
    for dataset in SEES:
        cells = []
        for model, _ in MODELS:
            found = runs.scored(dataset, model)
            main = found[2]["main"] if found else None
            cells.append(f"{pct(main['balanced_accuracy'])} ({span(main['balanced_accuracy_ci95'])})" if main else "")
        rows.append([f"`{dataset}`", *cells])
    return table(["Dataset", *(name for _, name in MODELS)], rows)


def by_dataset(runs: Runs, dataset: str) -> list[str]:
    rows = []
    for model, name in MODELS:
        found = runs.scored(dataset, model)
        if not found:
            rows.append([name, *[""] * 8])
            continue
        folder, _, metrics = found
        main = metrics["main"]
        rows.append([name, pct(main["balanced_accuracy"]), span(main["balanced_accuracy_ci95"]), pct(main["macro_f1"]),
                     pct(main["top3_balanced_accuracy"]), f"{main['invalid']} ({pct(main['invalid_rate'])})",
                     pct(main["selective"]["coverage"]), _risk(main["selective"]), f"`{folder.name}`"])
    return table(["Model", "Balanced accuracy", "95% interval", "Macro-F1", "Top-3", "Invalid answers", "Coverage", "Selective risk", "Run"], rows)


def per_class(runs: Runs, dataset: str) -> list[str]:
    found = [runs.scored(dataset, model) for model, _ in MODELS]
    reference = next((f[2]["main"]["per_class"] for f in found if f), {})
    rows = [[label, str(info["support"]),
             *[f"{100 * f[2]['main']['per_class'][label]['recall']:.1f}" if f else "" for f in found]] for label, info in reference.items()]
    return table(["Class", "Cases", *(name for _, name in MODELS)], rows)


def _beside(runs: Runs, datasets: list[str], key: str) -> list[str]:
    rows = []
    for dataset in datasets:
        for model, name in MODELS:
            found = runs.scored(dataset, model)
            if not found:
                rows.append([f"`{dataset}`", name, "", "", "", ""])
            elif key not in found[2]:
                rows.append([f"`{dataset}`", name, "n/a", "n/a", "n/a", "n/a"])
            else:
                part = found[2][key]["metrics"]
                rows.append([f"`{dataset}`", name, str(part["n"]), pct(part["balanced_accuracy"]), span(part["balanced_accuracy_ci95"]),
                             pct(part["macro_f1"])])
    return table(["Dataset", "Model", "Questions", "Balanced accuracy", "95% interval", "Macro-F1"], rows)


def without_small_classes(runs: Runs) -> list[str]:
    return _beside(runs, [d for d in SEES if DATASETS[d].set_apart], "without_small_classes")


def complete_record(runs: Runs) -> list[str]:
    return _beside(runs, [d for d in SEES if DATASETS[d].missing_fact], "all_facts_recorded")


def _difference(a: Path, b: Path) -> list[str]:
    c = compare(a, b)
    low, high = c["balanced_accuracy_difference_ci95"]
    return [str(c["n"]), signed(c["balanced_accuracy_difference"]), f"{signed(low)} to {signed(high)}", signed(c["macro_f1_difference"]),
            f"{c['only_b']} / {c['only_a']}", p_value(c["mcnemar_p"])]


_DIFFERENCE = ["Balanced accuracy, B minus A", "95% interval", "Macro-F1, B minus A", "Right only in B / only in A", "McNemar p"]


def same_model(runs: Runs) -> list[str]:
    rows = []
    for model, name in MODELS:
        for a, b in PAIRS:
            first, second = runs.folder(a, model), runs.folder(b, model)
            if first and second:
                runs.scored(a, model), runs.scored(b, model)  # both are checked as finished runs of the current format
                rows.append([name, f"`{a}`", f"`{b}`", *_difference(first, second)])
            else:
                known = [n for n in (runs.questions(a), runs.questions(b)) if n]
                rows.append([name, f"`{a}`", f"`{b}`", str(min(known)) if known else "", *[""] * 5])
    return table(["Model", "A", "B", "Questions", *_DIFFERENCE], rows)


def models_against_each_other(runs: Runs) -> list[str]:
    reference, rows = MODELS[0][0], []
    for dataset in SEES:
        for model, name in MODELS[1:]:
            first, second = runs.folder(dataset, reference), runs.folder(dataset, model)
            if first and second:
                runs.scored(dataset, reference), runs.scored(dataset, model)
                rows.append([f"`{dataset}`", name, *_difference(first, second)])
            else:
                rows.append([f"`{dataset}`", name, str(runs.questions(dataset) or ""), *[""] * 5])
    return table(["Dataset", "B", "Questions", *_DIFFERENCE], rows)


def render(results: Path | None = None) -> str:
    """Sections 2 to 6 of docs/results.md: the headings and the tables, without the explanations."""
    runs, out = Runs(results or RESULTS), []

    def add(heading: str, lines: list[str]) -> None:
        out.extend([heading, "", *lines, ""])

    add("## 2. Overview: balanced accuracy of every model on every dataset", overview(runs))
    out.extend(["## 3. Results by dataset", ""])
    for number, dataset in enumerate(SEES, 1):
        add(f"### 3.{number} `{dataset}`: {SEES[dataset]}", by_dataset(runs, dataset))
    out.extend(["## 4. Recall of each class", ""])
    for number, dataset in enumerate(SEES, 1):
        add(f"### 4.{number} `{dataset}`", per_class(runs, dataset))
    out.extend(["## 5. Beside the main score", ""])
    add("### 5.1 Without actinic keratosis (published cases)", without_small_classes(runs))
    add("### 5.2 Questions with a complete record (clinic data)", complete_record(runs))
    out.extend(["## 6. Paired comparisons", ""])
    add("### 6.1 The same model with different inputs", same_model(runs))
    add("### 6.2 Models against each other", models_against_each_other(runs))
    return "\n".join(out).rstrip("\n") + "\n"

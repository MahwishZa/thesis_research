"""A free count before a case-report multimodal dataset is built: how many cases of the frozen text dataset have, in their own
article's open full text in PubMed Central, a figure whose caption shows the patient's skin lesion as a clinical or a dermoscopic
photo. A case report's figures show its own patient, so such a figure and the case text are one patient's photo and description.

    python -m src.multimodal.figures count     (needs the internet; downloads PubMed records and PMC full texts, no image)

Only captions are read: no image is downloaded or looked at. For each case of data/text/dataset.csv, PubMed gives the article's PMC
number, PMC gives the open full text (the same source the text dataset used), and each figure of it is judged by its caption with
fixed word lists (below):
- it shows the lesion when the caption names dermoscopy or a clinical photo, or names a lesion (a nodule, a plaque and so on);
- it is left out when the caption also names a microscope or a stain, a scan, the state after treatment or at follow-up, a
  removed specimen, or a drawing or chart; or when it names a numbered case or patient, or several, so it may show another patient.
A case counts when its article is open, holds no other case of the text dataset, and has at least one figure that shows the lesion
and is not left out. A figure that shows the lesion but is left out only for its other parts (a panel of histology beside the photo)
is counted apart: it could be used only if its panels were cut apart.

The rule that decides whether the dataset is built was fixed before the count (2026-11-18): at least 80 test cases that count, and
at least 6 of the 8 classes with at least 8 such test cases each.

Writes data/multimodal/figure_count.csv (one row per case, no caption text), data/multimodal/figure_count.md (the counts and the
rule) and data/multimodal/figures/captions.jsonl (every figure's caption and judgement, on this computer only, as the captions are
under their publishers' licences).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

from ..common import fetch
from ..text import prepare as textprep
from .prepare import DATA, TEXT, _read_csv, _write_csv

COUNT_TABLE, COUNT_REPORT = "figure_count.csv", "figure_count.md"
CAPTIONS = Path("figures") / "captions.jsonl"  # in data/multimodal/, not in Git
GO_TEST_CASES, GO_CLASSES, GO_PER_CLASS = 80, 6, 8  # fixed on 2026-11-18, before the count

DERMOSCOPY = re.compile(r"dermoscop|dermatoscop|epiluminescence")
CLINICAL = re.compile(r"clinical (?:photo|image|picture|appearance|presentation|view|aspect|feature|finding|examination)|photograph"
                      r"|close-up|\b(?:lesions?|nodules?|papules?|plaques?|macules?|patch(?:es)?|tumou?rs?|mass(?:es)?|ulcers?"
                      r"|ulceration|moles?|n(?:a)?evus|n(?:a)?evi|growths?|swellings?)\b")
LEFT_OUT = {  # a caption naming any of these shows more than the lesion before treatment, or something else
    "microscope or stain": re.compile(r"histolog|histopatholog|patholog|\bh\s*&\s*e\b|\bh&e\b|haematoxylin|hematoxylin|eosin|\bstain"
                                      r"|immunohisto|immunostain|magnification|(?:×|\bx)\s?\d{2,4}\b(?!\s*(?:mm|cm)\b)|microscop|high[- ]power"
                                      r"|low[- ]power|\bhpf\b|cytolog|confocal|optical coherence|\bsections?\b"),
    "scan": re.compile(r"\bct\b|computed tomograph|\bmri\b|magnetic resonance|ultrasound|ultrasonograph|sonograph|x-ray|radiograph"
                       r"|\bpet\b|scintigra|angiogra|doppler|mammogra"),
    "after treatment or at follow-up": re.compile(r"\bafter\b|post-?operative|post-?treatment|follow-?up|\blater\b|\bheal(?:ed|ing)\b"
                                                  r"|resolution|resolved|\bscars?\b|\bflaps?\b|grafts?\b|surgical defect"
                                                  r"|intra-?operative|\bduring (?:surgery|treatment|the operation)"),
    "removed specimen": re.compile(r"specimen|macroscop|\bgross\b"),
    "drawing or chart": re.compile(r"schem|diagram|illustrat|algorithm|flow ?chart|timeline|\bgraphs?\b|\bcharts?\b|kaplan|pedigree"),
}
ANOTHER_PATIENT = re.compile(r"\b(?:case|patient)\s*(?:no\.?\s*)?(?:\d+|i{1,3}|iv|v|one|two|three)\b|\b(?:patients|cases)\b"
                             r"|\b(?:first|second|third) (?:case|patient)\b")
MAY_BE_ANOTHER = "may show another patient"
CITED = re.compile(r"\bfig(?:ure)?s?\.?\s*(\d+)", re.IGNORECASE)

CASE_COLUMNS = ["patient_uid", "split", "class", "pmid", "pmcid", "licence", "figures", "lesion_figures", "dermoscopy_figures",
                "lesion_figures_with_other_parts", "lesion_figures_cited_in_question", "counts", "reason"]
REASONS = ("no PMC number", "no open full text in PMC", "no figure", "the article holds another case of the dataset",
           "no figure whose caption shows the lesion alone")


def judge(caption: str) -> dict:
    """What a figure's caption says it shows, by the fixed word lists: dermoscopy, a clinical photo, and why it is left out."""
    text = " ".join(caption.lower().split())
    reasons = [name for name, pattern in LEFT_OUT.items() if pattern.search(text)]
    if ANOTHER_PATIENT.search(text):
        reasons.append(MAY_BE_ANOTHER)
    dermoscopy, clinical = bool(DERMOSCOPY.search(text)), bool(CLINICAL.search(text))
    return {"dermoscopy": dermoscopy, "clinical": clinical, "lesion": dermoscopy or clinical, "left_out_for": reasons}


def _number(label: str, position: int) -> int:
    found = re.search(r"\d+", label or "")
    return int(found.group(0)) if found else position


def parse_figures(content: bytes) -> dict[str, dict]:
    """PMC's full texts, by PMC number (without "PMC"): whether the full text is open (it has a body), the licence, and every
    figure with its label, number, caption and number of pictures."""
    root = ElementTree.fromstring(content)
    if root.tag != "pmc-articleset":
        raise ValueError(f"PMC's records have an unexpected form (<{root.tag}>)")
    articles = {}
    for article in root.findall("article"):
        ids = {e.get("pub-id-type"): (e.text or "").strip() for e in article.findall("front/article-meta/article-id")}
        number = (ids.get("pmc") or ids.get("pmcid") or "").removeprefix("PMC")
        if not number:
            continue
        figures = []
        for position, fig in enumerate(article.iter("fig"), start=1):
            label = fig.find("label")
            label = " ".join("".join(label.itertext()).split()) if label is not None else ""
            caption = fig.find("caption")
            figures.append({"figure": fig.get("id") or label or str(position), "label": label, "number": _number(label, position),
                            "caption": " ".join("".join(caption.itertext()).split()) if caption is not None else "",
                            "pictures": len(fig.findall(".//graphic"))})
        articles[number] = {"open": article.find("body") is not None, "figures": figures,
                            "licence": textprep._licence(article.find("front/article-meta/permissions/license"))}
    return articles


def _pmc_figures(numbers: list[str]) -> dict[str, dict]:
    return fetch(f"{textprep.EUTILS}efetch.fcgi", "application/xml", parse_figures, delay=textprep.NCBI_DELAY,
                 attempts=textprep.NCBI_ATTEMPTS, form={"db": "pmc", "id": ",".join(numbers), "retmode": "xml"})


def _pmcids(pmids: list[str], say) -> dict[str, str]:
    found = {}
    for start in range(0, len(pmids), textprep.EFETCH_BATCH):
        found.update((r["pmid"], r["pmcid"].removeprefix("PMC")) for r in textprep._efetch(pmids[start:start + textprep.EFETCH_BATCH]))
        say(f"{min(start + textprep.EFETCH_BATCH, len(pmids))} of {len(pmids)} PubMed records read")
    return found


def _cited(question: str | None) -> set[int] | None:
    """The figure numbers the question's text cites ("(Figure 1A)", "Figs. 2"), or None when the question is not on this computer."""
    return None if question is None else {int(n) for n in CITED.findall(question)}


def _question(text: Path, row: dict) -> str | None:
    path = text / "dataset" / row["path"]
    if not path.is_file():
        return None
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != row["sha256"]:
        raise ValueError(f"{path.as_posix()} is not the question the frozen text dataset lists (its fingerprint differs)")
    return content.decode("utf-8")


def count(text: Path = TEXT, data: Path = DATA, show=None) -> dict:
    """The count (see the module's description): writes figure_count.csv, figure_count.md and the local captions."""
    text, data = Path(text), Path(data)
    say = show or (lambda message: None)
    cases = _read_csv(text / "dataset.csv")
    per_article = Counter(r["pmid"] for r in cases)
    pmcids = _pmcids(sorted({r["pmid"] for r in cases}, key=int), say)
    numbers = sorted({n for n in pmcids.values() if n}, key=int)
    articles = {}
    for start in range(0, len(numbers), textprep.FULLTEXT_BATCH):
        articles.update(_pmc_figures(numbers[start:start + textprep.FULLTEXT_BATCH]))
        say(f"{min(start + textprep.FULLTEXT_BATCH, len(numbers))} of {len(numbers)} PMC full texts asked")

    rows, captions = [], []
    for case in sorted(cases, key=lambda r: (r["split"], r["label"], r["patient_uid"])):
        number = pmcids.get(case["pmid"], "")
        article = articles.get(number) if number else None
        question = _question(text, case)
        cited = _cited(question)
        figures = (article or {}).get("figures", []) if (article or {}).get("open") else []
        judged = [{**f, **judge(f["caption"])} for f in figures]
        lesion = [f for f in judged if f["lesion"] and not f["left_out_for"]]
        for f in judged:
            captions.append({"patient_uid": case["patient_uid"], "pmcid": f"PMC{number}", **f})
        if not number:
            reason = REASONS[0]
        elif not article or not article["open"]:
            reason = REASONS[1]
        elif not figures:
            reason = REASONS[2]
        elif per_article[case["pmid"]] > 1:
            reason = REASONS[3]
        elif not lesion:
            reason = REASONS[4]
        else:
            reason = ""
        rows.append({"patient_uid": case["patient_uid"], "split": case["split"], "class": case["label"], "pmid": case["pmid"],
                     "pmcid": f"PMC{number}" if number else "", "licence": (article or {}).get("licence", ""), "figures": len(figures),
                     "lesion_figures": len(lesion), "dermoscopy_figures": sum(f["dermoscopy"] for f in lesion),
                     "lesion_figures_with_other_parts": sum(f["lesion"] and bool(f["left_out_for"]) and MAY_BE_ANOTHER not in f["left_out_for"]
                                                            for f in judged),
                     "lesion_figures_cited_in_question": "" if cited is None else sum(f["number"] in cited for f in lesion),
                     "counts": "no" if reason else "yes", "reason": reason})

    _write_csv(data / COUNT_TABLE, CASE_COLUMNS, rows)
    local = data / CAPTIONS
    local.parent.mkdir(parents=True, exist_ok=True)
    partial = local.with_name(local.name + ".partial")
    partial.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in captions), encoding="utf-8", newline="\n")
    os.replace(partial, local)
    decision = go(rows)
    report = data / COUNT_REPORT
    report.write_text("\n".join(_report(rows, captions, decision)) + "\n", encoding="utf-8", newline="\n")
    return {"rows": rows, "captions": len(captions), "decision": decision, "table": (data / COUNT_TABLE).as_posix(),
            "report": report.as_posix(), "local": local.as_posix()}


def go(rows: list[dict]) -> dict:
    """The rule fixed before the count: enough test cases that count, in enough classes."""
    test = Counter(r["class"] for r in rows if r["split"] == "test" and r["counts"] == "yes")
    enough = sorted(c for c, n in test.items() if n >= GO_PER_CLASS)
    return {"test_cases": sum(test.values()), "per_class": dict(test), "classes_with_enough": enough,
            "build": sum(test.values()) >= GO_TEST_CASES and len(enough) >= GO_CLASSES}


def licence_kind(licence: str) -> str:
    value = licence.lower()
    if "publicdomain/zero" in value or "cc0" in value:
        return "CC0"
    if "/by-nc" in value:
        return "CC BY-NC (any)"
    if "creativecommons.org/licenses/by" in value:
        return "CC BY (any, no NC)"
    return f"other: {licence}" if licence else "not stated"


def _report(rows: list[dict], captions: list[dict], decision: dict) -> list[str]:
    day = datetime.now(timezone.utc).date().isoformat()
    classes = sorted({r["class"] for r in rows})
    lines = ["# Figures of the text dataset's cases: a free count", "",
             f"Written by `python -m src.multimodal.figures count` on {day}. Only the captions were read: no image was downloaded or "
             "looked at. The rules are in `src/multimodal/figures.py`. A case counts when its article's full text is open in PubMed "
             "Central, the article holds no other case of the text dataset, and at least one figure's caption shows the lesion (it names "
             "dermoscopy, a clinical photo or a lesion) and names no microscope or stain, scan, state after treatment or at follow-up, "
             "removed specimen, drawing or chart, and no numbered case or several patients.", "",
             f"Rule fixed before the count: build the dataset when at least {GO_TEST_CASES} test cases count and at least {GO_CLASSES} "
             f"classes have at least {GO_PER_CLASS} test cases that count.", "",
             f"**Result: {decision['test_cases']} test cases count; {len(decision['classes_with_enough'])} classes have at least "
             f"{GO_PER_CLASS} ({', '.join(decision['classes_with_enough']) or 'none'}). The rule is "
             f"{'met: the dataset can be built' if decision['build'] else 'not met: the dataset is not built'}.**", "",
             "## Cases that count, per class and split", "",
             "| Class | Test: cases | Test: count | Test: with a dermoscopy figure | Validation: cases | Validation: count | Train: cases | "
             "Train: count |", "|---|---|---|---|---|---|---|---|"]
    for label in classes + ["all"]:
        def n(split, only=None):
            return sum(1 for r in rows if r["split"] == split and (label == "all" or r["class"] == label)
                       and (only is None or only(r)))
        lines.append(f"| {label} | {n('test')} | {n('test', lambda r: r['counts'] == 'yes')} | "
                     f"{n('test', lambda r: r['counts'] == 'yes' and int(r['dermoscopy_figures']) > 0)} | {n('val')} | "
                     f"{n('val', lambda r: r['counts'] == 'yes')} | {n('train')} | {n('train', lambda r: r['counts'] == 'yes')} |")
    reasons = Counter(r["reason"] for r in rows if r["reason"])
    lines += ["", "## Why the other cases do not count", "", "| Reason (the first that applies) | Cases |", "|---|---|"]
    lines += [f"| {reason} | {reasons[reason]} |" for reason in REASONS]
    others = sum(1 for r in rows if r["reason"] == REASONS[4] and int(r["lesion_figures_with_other_parts"]) > 0)
    lines += ["", f"Of the cases with no figure that shows the lesion alone, {others} have a figure that shows the lesion beside other "
              "parts (such as a panel of histology); they could count only if the figure's panels were cut apart.", ""]
    left = Counter(reason for c in captions if c["lesion"] for reason in c["left_out_for"])
    lines += ["## Figures that show the lesion but are left out, by what else the caption names", "",
              "| The caption also names | Figures |", "|---|---|"]
    lines += [f"| {name} | {left[name]} |" for name in [*LEFT_OUT, MAY_BE_ANOTHER]]
    counted = [r for r in rows if r["counts"] == "yes"]
    cited = [r for r in counted if str(r["lesion_figures_cited_in_question"]) not in ("", "0")]  # a number, or "" when unknown
    known = [r for r in counted if str(r["lesion_figures_cited_in_question"]) != ""]
    licences = Counter(licence_kind(r["licence"]) for r in counted)
    lines += ["", "## The cases that count", "",
              f"- Figures that show the lesion: {sum(int(r['lesion_figures']) for r in counted)} in {len(counted)} cases; cases with more "
              f"than one: {sum(int(r['lesion_figures']) > 1 for r in counted)}; with a dermoscopy figure: "
              f"{sum(int(r['dermoscopy_figures']) > 0 for r in counted)}",
              f"- Cases whose question cites one of these figures by number: {len(cited)} of {len(known)} whose question is on this "
              "computer",
              "- Licences of their articles: " + ("; ".join(f"{k}: {n}" for k, n in sorted(licences.items())) or "none"), "",
              "## What this count does not show", "",
              "- No image was looked at. A caption can be wrong or incomplete; the build step checks the images themselves (what kind of "
              "picture each is, writing in the picture, the same picture twice, a picture also in the image dataset).",
              "- A figure that counts may still show more than the lesion (an arrow, a ruler, a second panel the caption does not name).",
              "- The captions stay on this computer (`data/multimodal/figures/captions.jsonl`), under their publishers' licences.", ""]
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.multimodal.figures", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="step", required=True)
    sub.add_parser("count", help="count the cases whose article shows the lesion in a figure, by the captions (needs the internet)")
    return parser


def main(argv: list[str] | None = None) -> None:
    build_parser().parse_args(argv)
    try:
        done = count(show=print)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    d = done["decision"]
    print(f"{d['test_cases']} test cases count; classes with at least {GO_PER_CLASS}: {len(d['classes_with_enough'])}. Rule "
          f"{'met' if d['build'] else 'not met'}. Written: {done['table']}, {done['report']} and the captions in {done['local']} "
          f"(not in Git). Commit {done['table']} and {done['report']}.")


if __name__ == "__main__":
    main(sys.argv[1:])

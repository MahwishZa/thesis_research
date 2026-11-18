"""A free count before a case-report multimodal dataset is built: how many cases of the frozen text dataset have, in their own
article's open full text in PubMed Central, a figure whose caption shows the patient's skin lesion as a clinical or a dermoscopic
photo. A case report's figures show its own patient, so such a figure and the case text are one patient's photo and description.

    python -m src.build.cases.prepare count     (needs the internet; downloads PubMed records and PMC full texts, no image)
    python -m src.build.cases.prepare download  (needs the internet; the pictures of the figures that count, from PMC's open-access files)
    python -m src.build.cases.prepare check     (checks every saved picture by itself; needs PyTorch, open_clip and EasyOCR)
    python -m src.build.cases.prepare build     (the case dataset's table, data/multimodal/cases.csv, from the committed tables)

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

``download`` reads each counted case's article from PMC Article Datasets, NCBI's copy of PubMed Central's open-access articles in
the cloud (the AWS Open Data bucket pmc-oa-opendata; PMC's former open-access web service no longer answers). An article that is not
open access, or that is marked retracted, gives no picture. Otherwise the article's latest version, its XML (which names each
figure's picture files) and only the pictures of the figures that count are downloaded; each file is checked against the MD5
checksum PMC publishes for it, and the pictures are kept byte for byte in data/multimodal/figures/images/ (not in Git). Each picture is opened once to check that it is a readable picture. The table
data/multimodal/figure_files.csv lists every picture with its fingerprint; data/multimodal/figure_files.md gives the counts. The
download can be stopped and run again: an article already done is not asked again.

``check`` checks every saved picture by itself, with no person looking: it must be at least MIN_SIDE pixels on its shorter side;
no writing in it (read by EasyOCR) may name a diagnosis (any name in data/text/class_terms.csv, or a class's short form); an image
model (open_clip, asked which of KINDS the picture is) must see a clinical photo or a dermoscopy image, so a caption that missed a
panel of histology or a scan is caught; and it may not be the same picture as one of another case or of the image dataset (the
image dataset's own rule for "the same picture"). A case keeps the pictures that pass; the rule for building the dataset is applied
again to the cases that keep one. The table data/multimodal/figure_checks.csv gives every picture's results; figure_checks.md the
counts, the agreement between the captions and the image model on dermoscopy, and the versions of the models. The models' answers
are kept in data/multimodal/figures/checks.jsonl (not in Git), so a second run does not ask them again.

``build`` writes the case dataset: one question per case that keeps a picture, in data/multimodal/cases.csv. A question is the
text dataset's frozen question (path and fingerprint as in data/text/dataset.csv, same split and class) and the case's kept
pictures, in the order of their figures. The evaluation asks it with the text and the pictures (dataset "cases") or with the
pictures only ("case_photos"); the text only is the text dataset itself. No caption is shown. data/multimodal/cases.md gives the
counts.

``count`` writes data/multimodal/figure_count.csv (one row per case, no caption text), data/multimodal/figure_count.md (the counts and the
rule) and data/multimodal/figures/captions.jsonl (every figure's caption and judgement, on this computer only, as the captions are
under their publishers' licences).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

from PIL import Image

from ...common import fetch
from ..image import prepare as imageprep
from ..text import prepare as textprep
from ..image_metadata.prepare import DATA, TEXT, _read_csv, _write_csv

COUNT_TABLE, COUNT_REPORT = "figure_count.csv", "figure_count.md"
CAPTIONS = Path("figures") / "captions.jsonl"  # in data/multimodal/, not in Git
CLOUD = "https://pmc-oa-opendata.s3.amazonaws.com/"  # PMC Article Datasets: NCBI's copy of PMC's open-access articles (AWS Open Data)
CLOUD_BUCKET = "s3://pmc-oa-opendata/"
FILES_TABLE, FILES_REPORT = "figure_files.csv", "figure_files.md"
IMAGES = Path("figures") / "images"  # in data/multimodal/, not in Git: one folder per article
FILE_COLUMNS = ["patient_uid", "split", "class", "pmcid", "figure", "label", "picture", "file", "sha256", "width", "height", "licence",
                "status"]
PICTURE_TYPES = (".jpg", ".jpeg", ".png", ".gif", ".tif", ".tiff")  # in this order of preference when a figure has several
SAVED, NOT_OPEN, RETRACTED, NO_FILE, UNREADABLE = ("saved", "not open access in PMC", "the article is retracted",
                                                   "no picture file for the figure among the article's files", "the picture cannot be opened")
CHECKS_TABLE, CHECKS_REPORT = "figure_checks.csv", "figure_checks.md"
CHECK_CACHE = Path("figures") / "checks.jsonl"  # in data/multimodal/, not in Git: the models' answers, by picture fingerprint
MIN_SIDE = 128  # pixels on the shorter side; fixed before any picture was seen
IMAGE_MODEL = ("ViT-B-32", "laion2b_s34b_b79k")  # open_clip's model and weights
KINDS = {  # what the image model chooses from; fixed before any picture was seen
    "clinical photo": "a clinical photograph of a skin lesion",
    "dermoscopy": "a dermoscopy image of a skin lesion",
    "microscopy": "a microscope image of stained tissue",
    "scan": "a medical scan such as CT, MRI, X-ray or ultrasound",
    "drawing": "a chart, diagram, table or drawing"}
SKIN_KINDS = ("clinical photo", "dermoscopy")
CHECK_REASONS = ("smaller than the least size", "writing in the picture names a diagnosis", "the image model sees no photo of the skin",
                 "the same picture is in the image dataset", "the same picture is in another case", "the same picture twice in this case")
CHECK_COLUMNS = ["patient_uid", "split", "class", "pmcid", "figure", "file", "sha256", "width", "height", "caption_type", "model_kind",
                 "model_score", "writing", "diagnosis_words", "kept", "reason"]
CASES_TABLE, CASES_REPORT = "cases.csv", "cases.md"
CASES_COLUMNS = ["split", "label", "path", "sha256", "pmid", "patient_uid", "pictures", "n_pictures"]
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
             f"Written by `python -m src.build.cases.prepare count` on {day}. Only the captions were read: no image was downloaded or "
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


# ---------------------------------------------------------------------------------------------------------------------------------
# Step "download": the pictures of the figures that count


def parse_versions(content: bytes) -> list[int]:
    """The versions of an article in the cloud bucket, from its folder listing ("PMC123.1/", "PMC123.2/")."""
    root = ElementTree.fromstring(content)
    if not root.tag.endswith("ListBucketResult"):
        raise ValueError(f"the cloud bucket answered in an unexpected form (<{root.tag}>)")
    found = []
    for prefix in root.iter():
        if prefix.tag.endswith("}Prefix") and prefix.text and re.fullmatch(r"PMC\d+\.\d+/", prefix.text):
            found.append(int(prefix.text[:-1].rsplit(".", 1)[1]))
    return sorted(found)


def _latest(pmcid: str) -> str | None:
    """The folder of the article's latest version in the bucket ("PMC123.2/"), or None when the bucket does not hold it."""
    versions = fetch(f"{CLOUD}?list-type=2&delimiter=/&prefix={pmcid}.", "application/xml", parse_versions, delay=0.1,
                     attempts=textprep.NCBI_ATTEMPTS)
    return f"{pmcid}.{versions[-1]}/" if versions else None


def cloud_address(url: str) -> tuple[str, str]:
    """A file of the bucket as PMC's metadata names it ("s3://pmc-oa-opendata/KEY?md5=..."): its web address and its MD5."""
    if not url.startswith(CLOUD_BUCKET):
        raise ValueError(f"{url!r} is not a file of the bucket {CLOUD_BUCKET}")
    key, _, query = url[len(CLOUD_BUCKET):].partition("?")
    md5 = dict(part.split("=", 1) for part in query.split("&") if "=" in part).get("md5", "")
    if not re.fullmatch(r"[0-9a-f]{32}", md5):
        raise ValueError(f"{url!r} gives no MD5 checksum")
    return CLOUD + key, md5


def _get(url: str, md5: str | None = None) -> bytes:
    """A file of the bucket; with ``md5``, checked against it (a damaged download is asked again)."""
    def check(content: bytes) -> bytes:
        if md5 and hashlib.md5(content).hexdigest() != md5:
            raise ValueError(f"{url} does not match the MD5 checksum PMC gives for it")
        return content
    return fetch(url, "*/*", check, delay=0.1, attempts=textprep.NCBI_ATTEMPTS)


def figure_files(xml: bytes, media: list[str], wanted: list[dict]) -> dict[str, list[tuple[str, str, str]]]:
    """For each wanted figure (its id, else its label, in the article's XML), its picture files among the article's media, as
    (file name, web address, MD5), in the order of the figure's graphics; a figure with none gets an empty list."""
    root = ElementTree.fromstring(xml)
    by_stem = {}
    for url in media:
        address, md5 = cloud_address(url)
        name = address.rsplit("/", 1)[1]
        stem, kind = os.path.splitext(name)
        if kind.lower() in PICTURE_TYPES:
            by_stem.setdefault(stem, []).append((name, address, md5))
    figs = list(root.iter("fig"))
    by_id = {fig.get("id"): fig for fig in figs if fig.get("id")}
    by_label = {}
    for fig in figs:
        label = fig.find("label")
        by_label.setdefault(" ".join("".join(label.itertext()).split()) if label is not None else "", fig)
    found = {}
    for figure in wanted:
        fig = by_id.get(figure["figure"])
        if fig is None and figure["label"]:  # an element's truth value is not whether it exists
            fig = by_label.get(figure["label"])
        pictures = []
        for graphic in (fig.iter("graphic") if fig is not None else []):
            href = os.path.basename(graphic.get("{http://www.w3.org/1999/xlink}href") or "")
            stem, kind = os.path.splitext(href)
            files = by_stem.get(stem if kind.lower() in PICTURE_TYPES else href, [])
            if files:
                pictures.append(min(files, key=lambda f: PICTURE_TYPES.index(os.path.splitext(f[0])[1].lower())))
        found[figure["figure"]] = pictures
    return found


def _safe(name: str) -> str:
    return re.sub(r"[^\w.-]", "_", name)


def download(data: Path = DATA, show=None) -> dict:
    """The download step (see the module's description): figure_files.csv, figure_files.md and the pictures."""
    data = Path(data)
    say = show or (lambda message: None)
    cases = [r for r in _read_csv(data / COUNT_TABLE) if r["counts"] == "yes"]
    if not (data / CAPTIONS).is_file():
        raise FileNotFoundError(f"{(data / CAPTIONS).as_posix()} not found: run python -m src.build.cases.prepare count first")
    with (data / CAPTIONS).open(encoding="utf-8") as f:
        captions = [json.loads(line) for line in f if line.strip()]
    chosen = {}
    for c in captions:
        if c["lesion"] and not c["left_out_for"]:
            chosen.setdefault(c["patient_uid"], []).append(c)
    rows = []
    for number, case in enumerate(sorted(cases, key=lambda r: r["pmcid"]), start=1):
        figures = chosen.get(case["patient_uid"], [])
        if not figures:
            raise ValueError(f"{case['patient_uid']} counts in {COUNT_TABLE} but no figure of it does in {CAPTIONS.as_posix()}: run "
                             "python -m src.build.cases.prepare count again")
        folder = data / IMAGES / _safe(case["pmcid"])
        done = folder / "record.json"
        if done.is_file():
            rows += json.loads(done.read_text(encoding="utf-8"))
            continue
        base = {"patient_uid": case["patient_uid"], "split": case["split"], "class": case["class"], "pmcid": case["pmcid"]}
        version = _latest(case["pmcid"])
        record = json.loads(_get(f"{CLOUD}{version}{version[:-1]}.json")) if version else {}
        found, status = {}, ""
        if not record.get("is_pmc_openaccess"):
            status = NOT_OPEN
        elif record.get("is_retracted"):
            status = RETRACTED
        else:
            found = figure_files(_get(*cloud_address(record["xml_url"])), record.get("media_urls") or [], figures)
        licence = record.get("license_code") or ""
        mine = []
        for f in figures:
            pictures = found.get(f["figure"], [])
            if status or not pictures:
                mine.append({**base, "figure": f["figure"], "label": f["label"], "picture": "", "file": "", "sha256": "", "width": "",
                             "height": "", "licence": licence, "status": status or NO_FILE})
            for name, address, md5 in pictures:
                content = _get(address, md5)
                entry = {**base, "figure": f["figure"], "label": f["label"], "picture": name, "licence": licence}
                try:
                    with Image.open(io.BytesIO(content)) as picture:
                        picture.load()
                        width, height = picture.size
                except Exception:  # any picture Pillow cannot read
                    mine.append({**entry, "file": "", "sha256": "", "width": "", "height": "", "status": UNREADABLE})
                    continue
                path = folder / f"{_safe(f['figure'])}-{_safe(name)}"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.with_name(path.name + ".partial").write_bytes(content)
                os.replace(path.with_name(path.name + ".partial"), path)
                mine.append({**entry, "file": path.relative_to(data).as_posix(), "sha256": hashlib.sha256(content).hexdigest(),
                             "width": width, "height": height, "status": SAVED})
        folder.mkdir(parents=True, exist_ok=True)
        done.write_text(json.dumps(mine, ensure_ascii=False), encoding="utf-8")
        rows += mine
        say(f"{number} of {len(cases)} articles done ({case['pmcid']}: {sum(r['status'] == SAVED for r in mine)} pictures)")
    rows.sort(key=lambda r: (r["split"], r["class"], r["patient_uid"], r["figure"], r["picture"]))
    _write_csv(data / FILES_TABLE, FILE_COLUMNS, rows)
    report = data / FILES_REPORT
    report.write_text("\n".join(_files_report(rows, cases)) + "\n", encoding="utf-8", newline="\n")
    return {"rows": rows, "table": (data / FILES_TABLE).as_posix(), "report": report.as_posix()}


def _files_report(rows: list[dict], cases: list[dict]) -> list[str]:
    saved = [r for r in rows if r["status"] == SAVED]
    with_picture = {r["patient_uid"] for r in saved}
    lines = ["# Pictures of the figures that count", "",
             f"Written by `python -m src.build.cases.prepare download` on {datetime.now(timezone.utc).date().isoformat()}. The pictures "
             "come from PMC Article Datasets (NCBI's cloud copy of PMC's open-access articles), byte for byte, each checked against PMC's MD5 checksum, and stay on this computer (`data/multimodal/figures/images/`). "
             "Nothing in them was changed, and no person looked at them.", "",
             "| Class | Test: cases with a picture | Validation | Train |", "|---|---|---|---|"]
    for label in sorted({c["class"] for c in cases}) + ["all"]:
        def n(split):
            return sum(1 for c in cases if c["split"] == split and (label == "all" or c["class"] == label) and c["patient_uid"] in with_picture)
        lines.append(f"| {label} | {n('test')} | {n('val')} | {n('train')} |")
    status = Counter(r["status"] for r in rows)
    lines += ["", f"Pictures saved: {len(saved)} for {len(with_picture)} of {len(cases)} cases.", "",
              "| Figure or picture | Count |", "|---|---|"] + [f"| {s} | {status[s]} |" for s in (SAVED, NOT_OPEN, RETRACTED, NO_FILE, UNREADABLE)]
    return lines + [""]


# ---------------------------------------------------------------------------------------------------------------------------------
# Step "check": every saved picture, checked by itself


def _image_model():
    """open_clip's model as a function: a picture's path -> the probability of each of KINDS."""
    try:
        import open_clip
        import torch
    except ImportError as exc:
        raise FileNotFoundError("the check needs PyTorch and open_clip: pip install torch==2.15.0 torchvision==0.29.2 "
                                "open_clip_torch==4.2.2 easyocr==1.7.2") from exc
    model, _, preprocess = open_clip.create_model_and_transforms(IMAGE_MODEL[0], pretrained=IMAGE_MODEL[1])
    model.eval()
    tokens = open_clip.get_tokenizer(IMAGE_MODEL[0])(list(KINDS.values()))
    with torch.no_grad():
        texts = model.encode_text(tokens, normalize=True)

    def classify(path: Path) -> dict[str, float]:
        with Image.open(path) as picture:
            pixels = preprocess(picture.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            chances = (100.0 * model.encode_image(pixels, normalize=True) @ texts.T).softmax(dim=-1)[0].tolist()
        return {kind: round(p, 4) for kind, p in zip(KINDS, chances)}
    return classify, {"torch": torch.__version__, "open_clip": getattr(open_clip, "__version__", "?")}


def _reader():
    """EasyOCR as a function: a picture's path -> the pieces of writing it finds."""
    try:
        import easyocr
        import numpy
    except ImportError as exc:
        raise FileNotFoundError("the check needs EasyOCR: pip install torch==2.15.0 torchvision==0.29.2 open_clip_torch==4.2.2 "
                                "easyocr==1.7.2") from exc
    reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    def read(path: Path) -> list[str]:
        with Image.open(path) as picture:
            grey = numpy.asarray(picture.convert("L"))  # one channel: EasyOCR reads it the same way for every kind of file
        return [str(piece) for piece in reader.readtext(grey, detail=0)]
    return read, {"easyocr": getattr(easyocr, "__version__", "?")}


def diagnosis_words(pieces: list[str], names: list[str], short_forms: list[str]) -> list[str]:
    """The names of diagnoses in a picture's writing: a name of class_terms.csv as plain words (any case, singular or plural,
    either spelling), or a short form written as a whole word in capitals ("BCC")."""
    joined = " ".join(pieces)
    tokens = textprep._tokens(joined)
    found = [name for name in names if _spans(tokens, name)]
    found += [form for form in short_forms if re.search(rf"(?<![A-Za-z]){re.escape(form)}s?(?![A-Za-z])", joined)]
    return sorted(set(found))


def _spans(tokens: list[str], name: str) -> bool:
    words = textprep._tokens(name)
    return bool(words) and any(tokens[i:i + len(words)] == words for i in range(len(tokens) - len(words) + 1))


def _caption_type(caption: dict) -> str:
    return {(True, True): "clinical and dermoscopic", (True, False): "dermoscopic", (False, True): "clinical"}.get(
        (bool(caption.get("dermoscopy")), bool(caption.get("clinical"))), "")


def kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    """Cohen's kappa (Cohen, 1960) of two yes/no judgements of the same items."""
    n = len(pairs)
    if not n:
        return None
    observed = sum(a == b for a, b in pairs) / n
    a_yes, b_yes = sum(a for a, _ in pairs) / n, sum(b for _, b in pairs) / n
    expected = a_yes * b_yes + (1 - a_yes) * (1 - b_yes)
    return None if expected == 1 else (observed - expected) / (1 - expected)


def check(data: Path = DATA, text: Path = TEXT, image: Path = imageprep.DATA,
          classify=None, read=None, show=None) -> dict:
    """The check step (see the module's description): figure_checks.csv and figure_checks.md."""
    data, text, image = Path(data), Path(text), Path(image)
    say = show or (lambda message: None)
    pictures = [r for r in _read_csv(data / FILES_TABLE) if r["status"] == SAVED]
    with (data / CAPTIONS).open(encoding="utf-8") as f:
        captions = {(c["patient_uid"], c["figure"]): c for c in map(json.loads, f) if c}
    terms = [r["term"] for r in _read_csv(text / textprep.TERMS)]
    names = sorted(set(terms) | {r["label"] for r in _read_csv(image / "dataset.csv")})
    short_forms = sorted({form for forms in textprep.SHORT_FORMS.values() for form in forms})
    cache_path = data / CHECK_CACHE
    cache = {}
    if cache_path.is_file():
        with cache_path.open(encoding="utf-8") as f:
            cache = {e["sha256"]: e for e in map(json.loads, f) if e}
    versions = {}
    signatures, rows = [], []
    for number, picture in enumerate(pictures, start=1):
        path = data / picture["file"]
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != picture["sha256"]:
            raise ValueError(f"{path.as_posix()} is not the picture {FILES_TABLE} lists (its fingerprint differs): run python -m "
                             "src.build.cases.prepare download again")
        known = cache.get(picture["sha256"])
        if not known or known.get("model") != list(IMAGE_MODEL) or known.get("kinds") != list(KINDS):
            if classify is None:
                classify, found = _image_model()
                versions.update(found)
            if read is None:
                read, found = _reader()
                versions.update(found)
            known = {"sha256": picture["sha256"], "model": list(IMAGE_MODEL), "kinds": list(KINDS), "scores": classify(path),
                     "writing": read(path)}
            cache[picture["sha256"]] = known
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(known, ensure_ascii=False) + "\n")
        kind = max(known["scores"], key=known["scores"].get)
        words = diagnosis_words(known["writing"], names, short_forms)
        signatures.append(imageprep._signature(content))
        rows.append({"patient_uid": picture["patient_uid"], "split": picture["split"], "class": picture["class"], "pmcid": picture["pmcid"],
                     "figure": picture["figure"], "file": picture["file"], "sha256": picture["sha256"], "width": int(picture["width"]),
                     "height": int(picture["height"]), "caption_type": _caption_type(captions.get((picture["patient_uid"], picture["figure"]), {})),
                     "model_kind": kind, "model_score": known["scores"][kind], "writing": "yes" if known["writing"] else "no",
                     "diagnosis_words": "; ".join(words), "_reasons": set()})
        if number % 50 == 0:
            say(f"{number} of {len(pictures)} pictures checked")

    for row in rows:
        if min(row["width"], row["height"]) < MIN_SIDE:
            row["_reasons"].add(CHECK_REASONS[0])
        if row["diagnosis_words"]:
            row["_reasons"].add(CHECK_REASONS[1])
        if row["model_kind"] not in SKIN_KINDS:
            row["_reasons"].add(CHECK_REASONS[2])
    for entry in _read_csv(image / "dataset.csv"):
        theirs = imageprep._signature((image / "dataset" / entry["path"]).read_bytes())
        for row, mine in zip(rows, signatures):
            if _same(mine, theirs):
                row["_reasons"].add(CHECK_REASONS[3])
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if _same(signatures[i], signatures[j]):
                if rows[i]["patient_uid"] == rows[j]["patient_uid"]:
                    rows[j]["_reasons"].add(CHECK_REASONS[5])
                else:
                    rows[i]["_reasons"].add(CHECK_REASONS[4])
                    rows[j]["_reasons"].add(CHECK_REASONS[4])
    for row in rows:
        found = row.pop("_reasons")
        reasons = [r for r in CHECK_REASONS if r in found]
        row["kept"], row["reason"] = ("no", "; ".join(reasons)) if reasons else ("yes", "")

    _write_csv(data / CHECKS_TABLE, CHECK_COLUMNS, rows)
    cases = _read_csv(data / COUNT_TABLE)
    kept = {r["patient_uid"] for r in rows if r["kept"] == "yes"}
    decision = go([{**c, "counts": "yes" if c["patient_uid"] in kept else "no"} for c in cases])
    report = data / CHECKS_REPORT
    report.write_text("\n".join(_checks_report(rows, cases, kept, decision, versions)) + "\n", encoding="utf-8", newline="\n")
    return {"rows": rows, "decision": decision, "table": (data / CHECKS_TABLE).as_posix(), "report": report.as_posix()}


def _same(a: dict, b: dict) -> bool:
    """The image dataset's rule for "the same picture" (src/image/prepare.py, SAME)."""
    if (a["hash"] ^ b["hash"]).bit_count() > imageprep.SAME["hash_distance"]:
        return False
    found = imageprep._compare(a, b)
    return all(found[k] <= limit for k, limit in imageprep.SAME.items())


def _checks_report(rows, cases, kept, decision, versions) -> list[str]:
    classes = sorted({c["class"] for c in cases})
    lines = ["# Checks of the pictures, each by itself", "",
             f"Written by `python -m src.build.cases.prepare check` on {datetime.now(timezone.utc).date().isoformat()}. No person looked at "
             f"the pictures. Image model: open_clip {IMAGE_MODEL[0]} ({IMAGE_MODEL[1]}), asked which of these a picture is: "
             + "; ".join(f"{k} (\"{v}\")" for k, v in KINDS.items()) + ". Writing: EasyOCR (English). Versions of the models' answers "
             "made in this run: " + (", ".join(f"{k} {v}" for k, v in sorted(versions.items())) or "none (all answers were kept from an "
             "earlier run)") + ".", "",
             f"**After the checks: {decision['test_cases']} test cases keep a picture; {len(decision['classes_with_enough'])} classes have at "
             f"least {GO_PER_CLASS} ({', '.join(decision['classes_with_enough']) or 'none'}). The rule for building the dataset "
             f"(at least {GO_TEST_CASES} test cases and {GO_CLASSES} classes) is {'met' if decision['build'] else 'not met'}.**", "",
             "| Class | Test: cases with a picture kept | Validation | Train |", "|---|---|---|---|"]
    for label in classes + ["all"]:
        def n(split):
            return sum(1 for c in cases if c["split"] == split and (label == "all" or c["class"] == label) and c["patient_uid"] in kept)
        lines.append(f"| {label} | {n('test')} | {n('val')} | {n('train')} |")
    reasons = Counter(r for row in rows for r in row["reason"].split("; ") if r)
    lines += ["", f"Pictures checked: {len(rows)}; kept: {sum(r['kept'] == 'yes' for r in rows)}.", "",
              "| Left out because (a picture can have several reasons) | Pictures |", "|---|---|"]
    lines += [f"| {reason} | {reasons[reason]} |" for reason in CHECK_REASONS]
    skin = [r for r in rows if r["model_kind"] in SKIN_KINDS and r["caption_type"]]
    pairs = [("dermoscopic" in r["caption_type"], r["model_kind"] == "dermoscopy") for r in skin]
    agreement = kappa(pairs)
    lines += ["", "## Do the captions and the image model agree on dermoscopy?", "",
              f"Of {len(skin)} pictures the model sees as a skin photo, the caption and the model agree on whether it is a dermoscopy image "
              f"for {sum(a == b for a, b in pairs)}; Cohen's kappa {'–' if agreement is None else f'{agreement:.2f}'}. A low kappa means the "
              "picture type cannot be taken from either alone.", "",
              f"Pictures with any writing (labels, arrows' letters, scale bars): {sum(r['writing'] == 'yes' for r in rows)}; with the name of a "
              f"diagnosis: {sum(bool(r['diagnosis_words']) for r in rows)}.", ""]
    return lines


# ---------------------------------------------------------------------------------------------------------------------------------
# Step "build": the case dataset's table


def build(data: Path = DATA, text: Path = TEXT) -> dict:
    """The build step (see the module's description): cases.csv and cases.md, from figure_checks.csv, figure_files.csv and the
    text dataset's table."""
    data, text = Path(data), Path(text)
    kept = [r for r in _read_csv(data / CHECKS_TABLE) if r["kept"] == "yes"]
    labels = {(r["patient_uid"], r["figure"]): r["label"] for r in _read_csv(data / FILES_TABLE)}
    questions = {r["patient_uid"]: r for r in _read_csv(text / "dataset.csv")}
    by_case = {}
    for picture in kept:
        by_case.setdefault(picture["patient_uid"], []).append(picture)
    rows = []
    for uid, pictures in by_case.items():
        question = questions.get(uid)
        if question is None:
            raise ValueError(f"{uid} has pictures in {CHECKS_TABLE} but is no case of the text dataset")
        for picture in pictures:
            if (picture["split"], picture["class"]) != (question["split"], question["label"]):
                raise ValueError(f"{picture['file']} is listed for {uid} in {picture['split']}/{picture['class']}, but the text dataset "
                                 f"has the case in {question['split']}/{question['label']}")
        pictures.sort(key=lambda p: (_number(labels.get((uid, p["figure"]), ""), 10 ** 6), p["file"]))
        rows.append({"split": question["split"], "label": question["label"], "path": question["path"], "sha256": question["sha256"],
                     "pmid": question["pmid"], "patient_uid": uid, "pictures": ";".join(f"{p['file']}|{p['sha256']}" for p in pictures),
                     "n_pictures": len(pictures)})
    rows.sort(key=lambda r: r["path"])
    _write_csv(data / CASES_TABLE, CASES_COLUMNS, rows)
    report = data / CASES_REPORT
    report.write_text("\n".join(_cases_report(rows)) + "\n", encoding="utf-8", newline="\n")
    return {"rows": rows, "table": (data / CASES_TABLE).as_posix(), "report": report.as_posix()}


def _cases_report(rows: list[dict]) -> list[str]:
    lines = ["# The case dataset", "",
             "Written by `python -m src.build.cases.prepare build`. One question per case of the text dataset whose own article shows "
             "the lesion in pictures that passed every check (`figure_checks.md`): the frozen text question, word for word, and the "
             "pictures, byte for byte, in the order of their figures. No caption is shown. The splits are the text dataset's.", "",
             "| Class | Test | Validation | Train |", "|---|---|---|---|"]
    for label in sorted({r["label"] for r in rows}) + ["all"]:
        def n(split):
            return sum(1 for r in rows if r["split"] == split and label in ("all", r["label"]))
        lines.append(f"| {label} | {n('test')} | {n('val')} | {n('train')} |")
    many = Counter(int(r["n_pictures"]) for r in rows)
    lines += ["", f"Questions: {len(rows)}; pictures: {sum(int(r['n_pictures']) for r in rows)}. Pictures per question: "
              + ", ".join(f"{k}: {many[k]}" for k in sorted(many)) + ".", ""]
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.build.cases.prepare", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="step", required=True)
    sub.add_parser("count", help="count the cases whose article shows the lesion in a figure, by the captions (needs the internet)")
    sub.add_parser("download", help="download the pictures of the figures that count, from PMC's cloud copy of open-access articles (needs the internet)")
    sub.add_parser("check", help="check every saved picture by itself (size, writing, image model, the same picture twice)")
    sub.add_parser("build", help="write the case dataset's table, data/multimodal/cases.csv, from the committed tables")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.step == "build":
            done = build()
            print(f"{len(done['rows'])} questions ({sum(r['split'] == 'test' for r in done['rows'])} test). Written: {done['table']} and "
                  f"{done['report']}.")
            return
        if args.step == "check":
            done = check(show=print)
            d = done["decision"]
            print(f"{sum(r['kept'] == 'yes' for r in done['rows'])} of {len(done['rows'])} pictures kept; {d['test_cases']} test cases keep "
                  f"one; classes with at least {GO_PER_CLASS}: {len(d['classes_with_enough'])}. Rule {'met' if d['build'] else 'not met'}. "
                  f"Written: {done['table']} and {done['report']}. Commit both.")
            return
        if args.step == "download":
            done = download(show=print)
            saved = [r for r in done["rows"] if r["status"] == SAVED]
            print(f"{len(saved)} pictures saved for {len({r['patient_uid'] for r in saved})} cases. Written: {done['table']} and "
                  f"{done['report']}; the pictures stay in data/multimodal/figures/images/ (not in Git). Commit {done['table']} and "
                  f"{done['report']}.")
            return
        done = count(show=print)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    d = done["decision"]
    print(f"{d['test_cases']} test cases count; classes with at least {GO_PER_CLASS}: {len(d['classes_with_enough'])}. Rule "
          f"{'met' if d['build'] else 'not met'}. Written: {done['table']}, {done['report']} and the captions in {done['local']} "
          f"(not in Git). Commit {done['table']} and {done['report']}.")


if __name__ == "__main__":
    main(sys.argv[1:])

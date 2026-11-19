"""Additions to the knowledge base (data/knowledge/). The library (src/build/knowledge/library.py) holds little text that describes
how a lesion looks. Two steps add to it, each in a file of its own; no earlier file changes:

    python -m src.build.knowledge.enrich appearance   open full texts of reviews on dermoscopy and clinical appearance (needs the internet)
    python -m src.build.knowledge.enrich aspects      what each library passage is about (appearance, histology, treatment, epidemiology)

``appearance`` searches PubMed Central, with the library's own rules (the class's MeSH headings as a main subject, reviews on the
skin, English, about people, not case reports, the dataset's articles left out) and finds the reviews that mention dermoscopy or
morphology; their open full texts are cut into passages as the library's are, and a passage is kept only when it describes
appearance, is not already in the library, and shares no run of 12 words with a question of the text dataset. The table is
data/knowledge/appearance.csv (committed); the texts are in data/knowledge/texts/appearance.jsonl (not committed, as the library's).

For three classes the first search found almost nothing (0, 0 and 1 reviews: dermatofibroma, pigmented benign keratosis and vascular
lesion, left short of the target by the first run on 2026-11-17), so they get a wider search: the reviews indexed with the class's
headings as a main subject OR naming one of the class's names (data/text/build/class_terms.csv, the ones that count) in the title or
abstract, without the dermoscopy and morphology words. A review that names a condition class_terms.csv says does not count (and not only inside a name
that counts) is left out. Nothing else changes, and the target of 30 passages from 8 documents is not lowered.

``aspects`` tags every passage of the library and of the appearance corpus by words (fixed lists below) and checks its own tags
against an independent signal: the title of the section a passage sits in, which its authors chose (a passage under "Treatment" is
about treatment). It then counts, per class, the passages that describe appearance against a target fixed here. No person reads
anything: the tags are a word count, and the check shows whether they follow the authors' headings, not whether the medicine is right.

Nothing here describes a patient who was not recorded: a retrieved passage is about the disease. The link between a question and a
passage is made at question time by retrieval, not stored.

(Version 1 had a third step, ``cards``: one text card per training photo of the ISIC image dataset. It was removed on 2026-11-19 with the
Version 1 photo datasets; it is in the Git history, commit a1c9209.)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from ...common import read_rows, write_rows, write_section
from . import library as textlib
from ..text import prepare as textprep
from .library import KNOWLEDGE, REPORT, REPORT_HEAD, STEPS

TEXT = textprep.DATA  # the text dataset: its tables, and on this computer its downloads

COMMANDS = ("appearance", "aspects")  # this module's steps; the report's sections are STEPS (the library first)

APPEARANCE_TABLE = "appearance.csv"
APPEARANCE_FILE = "appearance.jsonl"  # in data/knowledge/texts/, not in Git
ASPECTS_TABLE = "passage_aspects.csv"

TARGET_PASSAGES, TARGET_DOCUMENTS = 30, 8  # per class: passages that describe appearance, and the documents they come from. Fixed
# before the corpus was built; a class that falls short is reported as short, and the target is not lowered.
REVIEWS_ASKED, FULLTEXTS_PER_CLASS, MAX_PER_CLASS = 100, 40, 150
APPEARANCE_QUERY = (f"{textlib.REVIEW_QUERY} AND free full text[sb] AND (dermoscopy[Title/Abstract] OR dermoscopic[Title/Abstract] "
                    "OR dermatoscopy[Title/Abstract] OR dermatoscopic[Title/Abstract] OR morphology[Title/Abstract])")

WIDER_CLASSES = ("dermatofibroma", "pigmented benign keratosis", "vascular lesion")  # decided on 2026-11-17, after the first search left
# these three short (0, 1 and 0 reviews). They are searched by their names as well, and without the dermoscopy and morphology words.
WIDER_QUERY = f"{textlib.REVIEW_QUERY} AND free full text[sb]"

# What a passage is about, found by words. A passage has an aspect when it uses at least THRESHOLDS[aspect] different words of the
# list (a word counts once). The lists and thresholds are fixed here; they were not tuned on any result.
ASPECT_TERMS = {
    "dermoscopy": ["dermoscop", "dermatoscop", "pigment network", "globules", "streaks", "blue-white", "blue-gray", "blue-grey", "arborizing",
                   "arborising", "milia-like", "comedo-like", "lacunae", "rosettes", "strawberry pattern", "glomerular vessels",
                   "dotted vessels", "pseudonetwork", "leaf-like", "spoke-wheel", "spoke wheel", "regression structures",
                   "central white patch", "shiny white", "fingerprint-like", "moth-eaten"],
    "clinical look": ["papule", "plaque", "nodule", "macule", "scaly", "scaling", "pearly", "telangiectasia", "erythematous",
                      "hyperkeratotic", "ulcerat", "asymmetr", "irregular border", "well-demarcated", "well demarcated", "dome-shaped",
                      "verrucous", "crust", "pigmented lesion", "dimple sign"],
    "histology": ["histolog", "histopatholog", "biopsy", "immunohisto", "microscop", "dermis", "keratinocyte", "melanocyte"],
    "treatment": ["treatment", "therapy", "excision", "surgery", "surgical", "chemotherapy", "immunotherapy", "cryotherapy",
                  "radiotherapy", "imiquimod", "fluorouracil"],
    "epidemiology": ["incidence", "prevalence", "epidemiolog", "risk factor", "mortality", "survival"]}
THRESHOLDS = {"dermoscopy": 2, "clinical look": 3, "histology": 3, "treatment": 3, "epidemiology": 2}
ASPECTS = ("appearance", "histology", "treatment", "epidemiology")  # "appearance" is dermoscopy or clinical look
HEADINGS = (  # a section title that names one aspect; a title that names two names none
    ("treatment", re.compile(r"treat|manag|therap|surg|excis|prevent")),
    ("histology", re.compile(r"histolog|patholog|immunohisto|microscop")),
    ("appearance", re.compile(r"dermosc|dermatosc|clinical (?:present|feature|finding|manifest|appear|pictur|descript|characteristic)"
                              r"|history and physical|morpholog|appearance|signs? and symptoms|presentation")),
    ("epidemiology", re.compile(r"epidemiolog|incidence|prevalence|risk factor|etiolog")))

ASPECT_COLUMNS = ["passage_id", "origin", "source", "section", "heading_aspect", "appearance", "dermoscopy", "clinical_look", "histology",
                  "treatment", "epidemiology", "dermoscopy_words", "clinical_look_words"]


# ---------------------------------------------------------------------------------------------------------------------------------
# What a passage is about


def aspect_words(text: str) -> dict[str, int]:
    """For each aspect, how many different words of its list the text uses."""
    lowered = text.lower()
    return {aspect: sum(term in lowered for term in terms) for aspect, terms in ASPECT_TERMS.items()}


def aspect_flags(text: str) -> dict[str, bool]:
    """The aspects a text has: appearance (dermoscopy or clinical look), histology, treatment, epidemiology; and dermoscopy and
    clinical look on their own."""
    words = aspect_words(text)
    flags = {aspect: words[aspect] >= THRESHOLDS[aspect] for aspect in ASPECT_TERMS}
    flags["appearance"] = flags["dermoscopy"] or flags["clinical look"]
    return flags


def heading_aspect(section: str) -> str | None:
    """The one aspect a section title names, or None (no aspect named, or two)."""
    found = [aspect for aspect, pattern in HEADINGS if pattern.search(section.lower())]
    return found[0] if len(found) == 1 else None


def agreement(rows: list[dict]) -> dict[str, dict]:
    """How far the word tags follow the authors' section titles, per aspect, over the passages whose title names one aspect.
    recall: of the passages under a title of the aspect, the share the words tag with it. precision: of the passages the words tag
    with it, among those under a title that names an aspect, the share under a title of this aspect. base: the share of all titled
    passages that sit under a title of the aspect. lift = precision / base: above 1, the tag tells more than chance would."""
    titled = [r for r in rows if r["heading_aspect"]]
    out = {}
    for aspect in ASPECTS:
        under = [r for r in titled if r["heading_aspect"] == aspect]
        tagged = [r for r in titled if r[aspect] == "yes"]
        right = sum(r["heading_aspect"] == aspect for r in tagged)
        recall = sum(r[aspect] == "yes" for r in under) / len(under) if under else None
        precision = right / len(tagged) if tagged else None
        base = len(under) / len(titled) if titled else None
        out[aspect] = {"under_title": len(under), "tagged_under_title": len(tagged), "recall": recall, "precision": precision,
                       "base": base, "lift": precision / base if precision is not None and base else None}
    return out


def _class_names(text: Path, label: str) -> tuple[list[str], list[str], list[str]]:
    """The class's names (term_check.csv, which the survey checked against class_terms.csv): those that count and that PubMed
    finds in a title or abstract (searched for), all those that count, and those that do not count."""
    rows = [r for r in textlib._read_csv(Path(text) / textprep.CHECK, textprep.CHECK_COLUMNS, "survey") if r["class"] == label]
    return ([r["term"] for r in rows if r["counts"] == "yes" and int(r["pubmed_articles"] or 0)],
            [r["term"] for r in rows if r["counts"] == "yes"], [r["term"] for r in rows if r["counts"] == "no"])


def _spans(tokens: list[str], name: str) -> list[tuple[int, int]]:
    """Where a name stands in a text, both as plain words (lower case, American spelling, singular)."""
    words = textlib._tokens(name)
    return [(i, i + len(words)) for i in range(len(tokens) - len(words) + 1) if words and tokens[i:i + len(words)] == words]


def other_condition(record: dict, counts: list[str], not_counts: list[str]) -> str | None:
    """The first name that does not count (class_terms.csv) that a review's title or abstract uses, unless it only stands inside a
    longer name that counts ("lobular capillary hemangioma" holds "hemangioma"); None when there is none."""
    tokens = textlib._tokens(" ".join([record["title"]] + [text for _, text in record["sections"]]))
    inside = [span for name in counts for span in _spans(tokens, name)]
    for name in not_counts:
        for start, end in _spans(tokens, name):
            if not any(a <= start and end <= b for a, b in inside):
                return name
    return None


def _wider(label: str, headings: list[str], names: list[str], excluded: set[str], say) -> list[str]:
    """The newest reviews (free full text, no dermoscopy words asked) PubMed finds for the class's headings as a main subject OR its
    names in the title or abstract, without the dataset's articles."""
    subjects = ([textlib._subject(headings)] if headings else []) + (
        ["(" + " OR ".join(f'"{name}"[Title/Abstract]' for name in names) + ")"] if names else [])
    ids = []
    if subjects:
        answer = textlib._search("pubmed", f"({' OR '.join(subjects)}) AND {WIDER_QUERY}", retmax=REVIEWS_ASKED, sort="pub_date")
        ids = [p for p in answer.get("idlist") or [] if p not in excluded]
    say(f"{label}: {len(ids)} reviews found by the wider search")
    return ids


def _read_csv(path: Path) -> list[dict]:
    return read_rows(path, "build the text dataset and the library first")


_write_csv = write_rows


def _topics(value: str) -> list[str]:
    return [t.strip() for t in value.split(";") if t.strip()]


def _read_jsonl(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as f:
        return {p["passage_id"]: p for p in map(json.loads, f) if p}


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.0f}%"


# ---------------------------------------------------------------------------------------------------------------------------------
# Step "appearance": reviews on dermoscopy and clinical appearance, from PubMed Central


def appearance(text: Path = TEXT, data: Path = KNOWLEDGE, classes: list[str] | None = None, show=None) -> dict:
    """Finds, cuts and keeps the appearance passages (see the module's description); writes data/knowledge/appearance.csv, the
    texts into data/knowledge/texts/ and the "appearance" section of the report."""
    text, data = Path(text), Path(data)
    classes = classes if classes is not None else list(textprep.CLASSES)
    _, headings = textlib._checked_headings(text, classes)
    runs = textlib._question_runs(text)
    excluded = textlib._dataset_articles(text)
    say = show or (lambda message: None)
    day = datetime.now(timezone.utc).date().isoformat()
    library = {r["sha256"] for r in _read_csv(data / textlib.LIBRARY_TABLE)}

    found = textlib._newest(classes, headings, APPEARANCE_QUERY, REVIEWS_ASKED, excluded, say, "reviews on appearance")
    names = {}
    for label in (c for c in classes if c in WIDER_CLASSES):
        names[label] = _class_names(text, label)
        found[label] = _wider(label, headings[label], names[label][0], excluded, say)
    records = textlib._records(sorted({p for ids in found.values() for p in ids}, key=int), say)
    refused = {label: Counter() for label in names}  # per wider class: the reviews left out, by the name that does not count
    usable = {}
    for label in classes:
        usable[label] = [records[p] for p in found[label] if p in records]
        if label in names:
            checked = [(r, other_condition(r, names[label][1], names[label][2])) for r in usable[label]]
            refused[label].update(name for _, name in checked if name)
            usable[label] = [r for r, name in checked if not name]
    bodies = {}
    full = {label: textlib._full_texts(usable[label], bodies, say, label, FULLTEXTS_PER_CLASS) for label in classes}
    topics, chosen = defaultdict(list), {}
    for label in classes:
        for record in full[label]:
            topics[record["pmcid"]].append(label)
            chosen[record["pmcid"]] = record
    documents = [textlib._document("pmc", pmcid, chosen[pmcid], bodies[pmcid.removeprefix("PMC")]["sections"],
                                   bodies[pmcid.removeprefix("PMC")]["licence"], topics[pmcid])
                 for pmcid in sorted(chosen, key=lambda p: int(p.removeprefix("PMC")))]
    rows, passages, dropped = textlib._passages_of(documents, runs)

    kept, count = [], Counter()
    for row, passage in sorted(zip(rows, passages), key=lambda x: (-int(x[0]["year"] or 0), x[0]["source"], x[0]["passage_id"])):
        if not aspect_flags(passage["text"])["appearance"]:
            dropped["does not describe appearance"] += 1
        elif row["sha256"] in library:
            dropped["already in the library"] += 1
        elif all(count[c] >= MAX_PER_CLASS for c in _topics(row["topics"])):
            dropped[f"over the limit of {MAX_PER_CLASS} per class"] += 1
        else:
            for c in _topics(row["topics"]):
                count[c] += 1
            kept.append((row, passage))
    table, texts = [], []
    for row, passage in kept:
        pid = "appearance-" + row["passage_id"].removeprefix("pmc-")
        table.append({**row, "passage_id": pid, "corpus": "appearance"})
        texts.append({**passage, "passage_id": pid, "corpus": "appearance"})

    folder = data / textlib.FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    partial = folder / f"{APPEARANCE_FILE}.partial"
    partial.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in texts), encoding="utf-8", newline="\n")
    os.replace(partial, folder / APPEARANCE_FILE)
    _write_csv(data / APPEARANCE_TABLE, textlib.LIBRARY_COLUMNS, table)
    lines = _appearance_summary(day, classes, headings, found, full, documents, table, dropped, names, refused)
    write_section(data / REPORT, REPORT_HEAD, STEPS, "appearance", lines)
    return {"table": (data / APPEARANCE_TABLE).as_posix(), "passages": (folder / APPEARANCE_FILE).as_posix(), "rows": table,
            "documents": documents, "dropped": dropped, "report": (data / REPORT).as_posix()}


def _appearance_summary(day, classes, headings, found, full, documents, table, dropped, names, refused) -> list[str]:
    lines = ["# Appearance corpus: open full texts of reviews on dermoscopy and clinical appearance", "",
             f"Written on {day}. For each class, PubMed's reviews (the class's MeSH headings as a main subject, on the skin, in English, "
             "about people, not case reports, not guidelines, with a free full text, and the words dermoscopy, dermoscopic, dermatoscopy, "
             f"dermatoscopic or morphology in the title or abstract), the newest {REVIEWS_ASKED} asked, and of those the newest "
             f"{FULLTEXTS_PER_CLASS} that PubMed Central gives in full. Their paragraphs are cut into passages as the library's are. A "
             "passage is kept when it describes appearance (words of the lists in `enrich.py`), is not already in the library and shares "
             f"no run of {textlib.SHARED_RUN} words with a question of the text dataset; at most {MAX_PER_CLASS} per class, the newest first.", ""]
    if names:
        lines += [f"The first search found almost no review for {', '.join(names)}, so these classes were searched wider (column "
                  "Search): the same reviews as above, but without the dermoscopy and morphology words, and found by the class's "
                  "headings as a main subject OR by one of its names in the title or abstract. A review that names a condition "
                  "class_terms.csv says does not count is left out, unless the name stands only inside a longer name that counts. "
                  "The target is the same for every class.", ""]
        lines += [f"- {label}: searched for {'; '.join(names[label][0]) or 'no name'}; not counted: {'; '.join(names[label][2]) or 'none'}"
                  for label in names] + [""]
    lines += ["| Class | MeSH headings (main topic) | Search | Reviews found | Left out: names another condition | Open full texts | "
              "Passages kept | Documents of those passages |", "|---|---|---|---|---|---|---|---|"]
    for label in classes:
        mine = [r for r in table if label in _topics(r["topics"])]
        left = ", ".join(f"{n} ({k})" for n, k in sorted(refused[label].items())) if label in refused else ""
        lines.append(f"| {label} | {'; '.join(headings[label]) or 'none'} | {'wider' if label in names else 'first'} | {len(found[label])} | "
                     f"{sum(refused[label].values()) if label in refused else 0}{' (' + left + ')' if left else ''} | {len(full[label])} | "
                     f"{len(mine)} | {len({r['source'] for r in mine})} |")
    licences = Counter(d["licence"] or "not stated" for d in documents if any(r["source"] == d["source"] for r in table))
    lines += ["", f"Passages kept: {len(table)} from {len({r['source'] for r in table})} documents. Licences (documents): "
              + ("; ".join(f"{k}: {n}" for k, n in sorted(licences.items())) or "none"), "",
              "Left out: " + (", ".join(f"{k}: {n}" for k, n in sorted(dropped.items())) or "nothing"), "",
              f"Query: {APPEARANCE_QUERY}." + (f" Wider search, with the class's headings OR names: {WIDER_QUERY}." if names else ""), "",
              "No person has read the passages. A passage is about its review's class(es), found by the search, and describes appearance "
              "by a word count; neither is a medical judgement. The texts are under their publishers' licences (in the table) and stay "
              "on this computer for research use.", ""]
    return lines


# ---------------------------------------------------------------------------------------------------------------------------------
# Step "aspects": what each passage is about, and how much of the library describes appearance


def aspects(data: Path = KNOWLEDGE) -> dict:
    """Tags every passage of the library and of the appearance corpus; writes data/knowledge/passage_aspects.csv and the
    "aspects" section of the report (the tags checked against the section titles, and the coverage of appearance per class)."""
    data = Path(data)
    sources = [("library", data / textlib.LIBRARY_TABLE, data / textlib.FOLDER / "passages.jsonl")]
    if (data / APPEARANCE_TABLE).is_file() or (data / textlib.FOLDER / APPEARANCE_FILE).is_file():
        sources.append(("appearance", data / APPEARANCE_TABLE, data / textlib.FOLDER / APPEARANCE_FILE))
    rows, meta = [], {}
    for origin, table, texts in sources:
        for path in (table, texts):
            if not path.is_file():
                raise FileNotFoundError(f"{path.as_posix()} not found" + (": run python -m src.build.knowledge.enrich appearance first"
                                                                           if origin == "appearance" else ": run python -m src.build.knowledge.library first"))
        passages = _read_jsonl(texts)
        listed = _read_csv(table)
        if {r["passage_id"] for r in listed} != set(passages):
            raise ValueError(f"{table.as_posix()} and {texts.as_posix()} do not list the same passages")
        for entry in listed:
            text_of = passages[entry["passage_id"]]["text"]
            if hashlib.sha256(text_of.encode("utf-8")).hexdigest() != entry["sha256"]:
                raise ValueError(f"the text of {entry['passage_id']} is not the one its fingerprint in {table.as_posix()} names")
            flags, words = aspect_flags(text_of), aspect_words(text_of)
            rows.append({"passage_id": entry["passage_id"], "origin": origin, "source": entry["source"], "section": entry["section"],
                         "heading_aspect": heading_aspect(entry["section"]) or "",
                         **{a: "yes" if flags[a] else "no" for a in ("appearance", "dermoscopy", "histology", "treatment", "epidemiology")},
                         "clinical_look": "yes" if flags["clinical look"] else "no",
                         "dermoscopy_words": words["dermoscopy"], "clinical_look_words": words["clinical look"]})
            meta[entry["passage_id"]] = _topics(entry["topics"])
    _write_csv(data / ASPECTS_TABLE, ASPECT_COLUMNS, rows)
    labels = sorted({c for topics in meta.values() for c in topics})
    coverage = _coverage(rows, meta, labels)
    check = agreement(rows)
    write_section(data / REPORT, REPORT_HEAD, STEPS, "aspects", _aspects_summary(rows, labels, coverage, check, [o for o, *_ in sources]))
    return {"table": (data / ASPECTS_TABLE).as_posix(), "rows": rows, "coverage": coverage, "agreement": check,
            "short": [c for c in labels if not coverage[c]["meets_target"]], "report": (data / REPORT).as_posix()}


def _coverage(rows: list[dict], meta: dict, labels: list[str]) -> dict[str, dict]:
    out = {}
    for label in labels:
        mine = [r for r in rows if label in meta[r["passage_id"]]]
        look = [r for r in mine if r["appearance"] == "yes"]
        library, new = [r for r in look if r["origin"] == "library"], [r for r in look if r["origin"] == "appearance"]
        documents = {r["source"] for r in look}
        out[label] = {"passages": len(mine), "library": len(library), "new": len(new), "total": len(look),
                      "dermoscopy": sum(r["dermoscopy"] == "yes" for r in look), "documents": len(documents),
                      "meets_target": len(look) >= TARGET_PASSAGES and len(documents) >= TARGET_DOCUMENTS}
    return out


def _aspects_summary(rows, labels, coverage, check, origins) -> list[str]:
    per_origin = ", ".join(o + ": " + str(sum(r["origin"] == o for r in rows)) for o in origins)
    lines = ["# Aspects: what each passage is about", "",
             f"{len(rows)} passages tagged ({per_origin}), in `{ASPECTS_TABLE}`. "
             "A passage has an aspect when it uses several different words of that aspect's list (fixed in `enrich.py`): "
             "appearance is dermoscopy (2 words) or clinical look (3 words); histology 3, treatment 3, epidemiology 2. A passage can have "
             "several aspects, and a passage about several classes counts for each.", "",
             "## Do the tags follow the authors' section titles?", "",
             "The check uses an independent signal: the title of the section a passage sits in. Only passages under a title that names one "
             "aspect count (for example \"Treatment / Management\" names treatment). Recall: of the passages under a title of the aspect, "
             "the share the words tag with it. Precision: of the tagged passages, the share under a title of the aspect. Lift: precision "
             "against the share of all titled passages under that aspect; above 1.0, the tag tells more than chance.", "",
             "| Aspect | Passages under its titles | Recall | Tagged passages with a title | Precision | Lift |", "|---|---|---|---|---|---|"]
    for aspect in ASPECTS:
        c = check[aspect]
        lift = "–" if c["lift"] is None else f"{c['lift']:.1f}"
        lines.append(f"| {aspect} | {c['under_title']} | {_pct(c['recall'])} | {c['tagged_under_title']} | {_pct(c['precision'])} | {lift} |")
    informative = all(check[a]["lift"] is not None and check[a]["lift"] > 1.0 for a in ASPECTS)
    lines += ["", ("Every aspect has a lift above 1.0: the tags follow the authors' titles better than chance." if informative else
                   "**Not every aspect has a lift above 1.0 (or has titled passages to check): read the tags of that aspect with caution.**"),
              "This is a check of consistency with the authors' own headings. It does not show that a passage is medically right or "
              "sufficient, and a passage under no informative title is not checked at all.", "",
              "## Coverage of appearance", "",
              f"Target per class, fixed before the corpus was built: at least {TARGET_PASSAGES} passages that describe appearance, from at "
              f"least {TARGET_DOCUMENTS} different documents.", "",
              "| Class | Passages | In the library | In the appearance corpus | Describing appearance | Of which dermoscopy | Documents | Target |",
              "|---|---|---|---|---|---|---|---|"]
    for label in labels:
        c = coverage[label]
        lines.append(f"| {label} | {c['passages']} | {c['library']} | {c['new']} | {c['total']} | {c['dermoscopy']} | {c['documents']} | "
                     f"{'met' if c['meets_target'] else '**short**'} |")
    short = [c for c in labels if not coverage[c]["meets_target"]]
    lines += ["", (f"Classes short of the target: {', '.join(short)}. They stay short: the target is not lowered, and what is missing "
                    "is a limit of the retrieval experiments for these classes." if short else "Every class meets the target."), ""]
    return lines


# ---------------------------------------------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.build.knowledge.enrich", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="step", required=True)
    sub.add_parser("appearance", help="find, cut and keep passages on dermoscopy and clinical appearance (needs the internet)")
    sub.add_parser("aspects", help="tag every passage by aspect, check the tags against section titles, count the coverage per class")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.step == "appearance":
            done = appearance(show=print)
            left = ", ".join(k + ": " + str(n) for k, n in done["dropped"].items()) or "nothing"
            print(f"Kept {len(done['rows'])} passages from {len({r['source'] for r in done['rows']})} documents; left out: {left}. "
                  f"Written: {done['table']}, the passages in {done['passages']} (not in Git) and {done['report']}. "
                  f"Commit {done['table']} and {done['report']}.")
        else:
            done = aspects()
            short = ", ".join(done["short"]) or "none"
            print(f"{len(done['rows'])} passages tagged. Classes short of the target ({TARGET_PASSAGES} passages from {TARGET_DOCUMENTS} "
                  f"documents): {short}. Written: {done['table']} and {done['report']}.")
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main(sys.argv[1:])

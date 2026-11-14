"""Checks the built multimodal dataset with code of its own, and measures how much the patient facts alone give the class away.

    python -m src.multimodal.audit

The checks do not reuse the build's code (src/multimodal/prepare.py): each rule is written again here, so a mistake in the build
is not repeated in its check.

1. Questions: one per image of the image dataset, with the same split, class, file, fingerprint and lesion group; each fact
   sentence worked out again from the photo's ISIC records (each record checked against the fingerprint the image dataset saved).
2. Files on this computer: every image and training case the tables name exists and matches its fingerprint, and every passage is
   in data/text/library/passages.jsonl with the text its fingerprint was made from.
3. Knowledge: every training image and training case once, every library passage once, no validation or test item of either
   dataset, and no passage from an article that was a candidate case of the text dataset.
4. Diseases: every count recounted from the sources.

Then the shortcut check: a counting model learns from the training questions which class each value of a fact points to, with
every class weighing the same, and answers the test questions from the facts alone, with no photo. Its balanced accuracy (95%
interval: bootstrap over lesions, as in the evaluation) is what the facts alone are worth: the baseline a multimodal score must be
read against. Chance is 12.5% for the 8 classes; the same is measured for the photo collection (chance 50%), which the classes
follow almost exactly.

The result goes into the audit section of data/multimodal/dataset_report.md. A failed check stops with exit code 1.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

from ..common import write_section
from ..evaluate import summarize
from .prepare import DATA, IMAGE, REPORT, REPORT_HEAD, STEPS, TEXT

AUDIT_STEPS = STEPS
SITE_LEVEL = "anatom_site_1"  # the broadest body site; the finer levels are not shown
MISSING = "not recorded"
FEATURES = ("age", "sex", "site")


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"{path.as_posix()} not found: run the multimodal build first (python -m src.multimodal.prepare build)")
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _clinical(record) -> dict:
    """The record's clinical fields, wherever ISIC keeps them (metadata.clinical or metadata.patient)."""
    metadata = (record or {}).get("metadata") or {}
    out = {}
    for group in ("clinical", "patient"):
        for key, value in (metadata.get(group) or {}).items():
            if value not in (None, ""):
                out.setdefault(key, []).append(value)
    return out


def expected_facts(records: list[dict]) -> dict:
    """The facts a photo's ISIC records give: the one value they all agree on, else "not recorded"."""
    seen = defaultdict(list)
    for record in records:
        for key, values in _clinical(record).items():
            seen[key] += values
    out = {}
    for name, key in (("age", "age_approx"), ("sex", "sex"), ("site", SITE_LEVEL)):
        values = seen.get(key, [])
        if not values or any(v != values[0] for v in values):
            out[name] = MISSING
        elif name == "age":
            out[name] = f"about {float(values[0]):g} years"
        else:
            out[name] = str(values[0]).strip().lower()
    return out


def _case_licence(case: dict, full_texts: dict) -> str:
    """A case's licence: PMC-Patients' own, the publisher's copyright of an abstract, or the licence PMC gives its full text."""
    source = case.get("source")
    if source == "PMC-Patients":
        return "CC BY-NC-SA 4.0 (PMC-Patients)"
    if source == "PubMed abstract":
        return "publisher's copyright (abstract in PubMed)"
    return full_texts.get(case.get("pmid"), "") or "(no licence given by PMC)"


def _sentence(facts: dict) -> str:
    return f"Age: {facts['age']}. Sex: {facts['sex']}. Body site: {facts['site']}."


def audit(image: Path = IMAGE, text: Path = TEXT, data: Path = DATA) -> dict:
    questions, knowledge, diseases = (_rows(data / name) for name in ("questions.csv", "knowledge.csv", "diseases.csv"))
    images = {r["path"]: r for r in _rows(image / "dataset.csv")}
    photos = {r["photo_id"]: r for r in _rows(image / "raw_photos.csv")}
    cases = {r["path"]: r for r in _rows(text / "dataset.csv")}
    passages = {r["passage_id"]: r for r in _rows(text / "library.csv")}
    failures = defaultdict(list)
    full_texts = {}
    if (text / "sources" / "pubmed" / "abstracts.jsonl").is_file():
        for line in (text / "sources" / "pubmed" / "abstracts.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                full_texts[str(record["pmid"])] = record.get("licence") or ""

    # 1. Questions against the image dataset and the ISIC records
    by_path = Counter(q["path"] for q in questions)
    failures["a question twice, or an image without its question"] = sorted(
        [p for p, n in by_path.items() if n > 1] + [p for p in images if p not in by_path])
    for q in questions:
        source = images.get(q["path"])
        if source is None or any(q[k] != source[k] for k in ("split", "label", "sha256", "lesion_group")):
            failures["question differs from the image dataset"].append(q["path"])
            continue
        photo = photos[source["photo_id"]]
        numbers, prints = photo["isic_ids"].split(), photo["isic_record_sha256"].split()
        records = []
        for number, fingerprint in zip(numbers, prints, strict=True):
            path = image / "downloads" / "isic" / f"{number}.json"
            entry = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
            record = entry.get("record")
            if not record or _digest(json.dumps(record, sort_keys=True, ensure_ascii=False).encode("utf-8")) != fingerprint:
                failures["ISIC record missing or not the one the image dataset used"].append(number)
            records.append(record)
        facts = expected_facts(records)
        if q["isic_id"] != " ".join(numbers) or [q[f] for f in FEATURES] != [facts[f] for f in FEATURES] \
                or q["facts"] != _sentence(facts) or q["complete"] != ("no" if MISSING in facts.values() else "yes"):
            failures["facts differ from the ISIC records"].append(q["path"])

    # 2. Files on this computer
    for q in questions:
        file = image / "dataset" / q["path"]
        if not file.is_file() or _digest(file.read_bytes()) != q["sha256"]:
            failures["image file missing or changed"].append(q["path"])
    texts = {}
    library = text / "library" / "passages.jsonl"
    if library.is_file():
        for line in library.read_text(encoding="utf-8").splitlines():
            if line.strip():
                passage = json.loads(line)
                texts[passage["passage_id"]] = passage["text"]
    for k in knowledge:
        if k["kind"] == "case":
            file = Path(k["path"])
            if not file.is_file() or _digest(file.read_bytes()) != k["sha256"]:
                failures["case file missing or changed"].append(k["item_id"])
        elif k["kind"] == "passage":
            body = texts.get(k["item_id"].removeprefix("passage:"))
            if body is None or _digest(body.encode("utf-8")) != k["sha256"]:
                failures["passage missing from passages.jsonl or changed"].append(k["item_id"])

    # 3. Knowledge: the training items and the library, once each, nothing held out
    expected = ({f"image:{p}" for p, r in images.items() if r["split"] == "train"}
                | {f"case:{r['patient_uid']}" for r in cases.values() if r["split"] == "train"}
                | {f"passage:{p}" for p in passages})
    listed = Counter(k["item_id"] for k in knowledge)
    failures["knowledge item twice"] = sorted(i for i, n in listed.items() if n > 1)
    failures["knowledge item missing"] = sorted(expected - set(listed))
    failures["knowledge item that is not a training item or a passage"] = sorted(set(listed) - expected)
    held_out = ({f"data/image/dataset/{p}" for p, r in images.items() if r["split"] != "train"}
                | {f"data/text/dataset/{p}" for p, r in cases.items() if r["split"] != "train"})
    failures["validation or test item in the knowledge"] = sorted(k["item_id"] for k in knowledge if k["path"] in held_out)
    for k in knowledge:
        if k["link"] != "disease":
            failures["knowledge item linked other than by disease"].append(k["item_id"])
        if k["kind"] == "image" and (k["path"] != f"data/image/dataset/{k['item_id'].removeprefix('image:')}"
                                     or k["sha256"] != images.get(k["item_id"].removeprefix("image:"), {}).get("sha256")):
            failures["knowledge image differs from the image dataset"].append(k["item_id"])
        if k["kind"] == "case":
            case = cases.get(k["path"].removeprefix("data/text/dataset/"), {})
            if case.get("sha256") != k["sha256"] or k["diseases"] != case.get("label"):
                failures["knowledge case differs from the text dataset"].append(k["item_id"])
            if k["licence"] != _case_licence(case, full_texts):
                failures["knowledge case licence differs from its source"].append(k["item_id"])
        if k["kind"] == "passage":
            passage = passages.get(k["item_id"].removeprefix("passage:"), {})
            if passage.get("sha256") != k["sha256"] or k["diseases"] != passage.get("topics"):
                failures["knowledge passage differs from the library"].append(k["item_id"])
    candidates = [r for name in ("case_candidates.csv", "abstract_candidates.csv") if (text / name).is_file()
                  for r in _rows(text / name)]
    pmids = {r["pmid"] for r in candidates} | {r["pmid"] for r in cases.values()}
    pmcids = {"PMC" + r["patient_uid"].split("-")[0] for r in candidates + list(cases.values())
              if not r["patient_uid"].startswith("pubmed-")}
    failures["passage from an article of the text dataset's candidates"] = sorted(
        p for p, r in passages.items() if r["source"] in {f"PMID {n}" for n in pmids} | pmcids)

    # 4. Diseases, recounted
    labels = sorted({r["label"] for r in images.values()})
    if [d["class"] for d in diseases] != labels:
        failures["disease table does not list the classes"].append(", ".join(d["class"] for d in diseases))
    for d in diseases:
        label = d["class"]
        counts = {**{f"questions_{s}": sum(r["label"] == label and r["split"] == s for r in images.values()) for s in ("train", "val", "test")},
                  "complete_test": sum(q["label"] == label and q["split"] == "test" and q["complete"] == "yes" for q in questions),
                  **{f"cases_{s}": sum(r["label"] == label and r["split"] == s for r in cases.values()) for s in ("train", "val", "test")},
                  "passages": sum(label in r["topics"].split("; ") for r in passages.values())}
        if any(str(v) != d[k] for k, v in counts.items()):
            failures["disease count differs from the sources"].append(label)

    shortcuts = _shortcuts(questions)
    write_section(data / REPORT, REPORT_HEAD, AUDIT_STEPS, "audit", _summary(questions, knowledge, failures, shortcuts, bool(texts)))
    return {"failures": dict(failures), "shortcuts": shortcuts, "questions": len(questions), "knowledge": len(knowledge),
            "report": (data / REPORT).as_posix()}


# ---------------------------------------------------------------------------------------------------------------------------------
# The shortcut check: how much the facts alone tell, with no photo

def _counting_model(train: list[dict], features: tuple[str, ...], target: str):
    """For each target value, how often it has each value of each feature (add-one smoothing); an answer weighs every target value
    the same (the product of the features' shares within it, as naive Bayes with equal priors), so a frequent class is no default."""
    targets = sorted({q[target] for q in train})
    values = {f: sorted({q[f] for q in train}) for f in features}
    counts = {t: {f: Counter(q[f] for q in train if q[target] == t) for f in features} for t in targets}
    size = Counter(q[target] for q in train)

    def answer(q: dict) -> str:
        def score(t):
            return sum(math.log((counts[t][f][q[f]] + 1) / (size[t] + len(values[f]) + 1)) for f in features)
        return max(targets, key=lambda t: (score(t), [-ord(c) for c in t]))  # a tie goes to the first name

    return answer


def _shortcuts(questions: list[dict]) -> list[dict]:
    for q in questions:
        q["missing"] = " ".join(f for f in FEATURES if q[f] == MISSING) or "none"
    train = [q for q in questions if q["split"] == "train"]
    test = [q for q in questions if q["split"] == "test"]
    complete = [q for q in test if q["complete"] == "yes"]
    out = []
    for name, features, rows in (("age", ("age",), test), ("sex", ("sex",), test), ("body site", ("site",), test),
                                 ("which facts are missing", ("missing",), test), ("all three facts", FEATURES, test),
                                 ("all three facts, questions with all three", FEATURES, complete)):
        row = {"facts": name, "n": len(rows)}
        for target in ("label", "collection"):
            model = _counting_model(train, features, target)
            labels = sorted({q[target] for q in questions})  # every class is an answer option, even one without training questions
            scored = summarize([q[target] for q in rows], [model(q) for q in rows], labels, [q["lesion_group"] for q in rows])
            row[target] = (scored["balanced_accuracy"], scored["balanced_accuracy_ci95"])
        out.append(row)
    return out


def _summary(questions, knowledge, failures, shortcuts, have_library) -> list[str]:
    failed = {k: v for k, v in failures.items() if v}
    lines = ["## Audit: the tables against their sources, and what the facts alone tell", "",
             f"Checked with code of its own: {len(questions)} questions and {len(knowledge)} knowledge items.", ""]
    checks = ["a question twice, or an image without its question", "question differs from the image dataset",
              "ISIC record missing or not the one the image dataset used", "facts differ from the ISIC records",
              "image file missing or changed", "case file missing or changed", "passage missing from passages.jsonl or changed",
              "knowledge item twice", "knowledge item missing", "knowledge item that is not a training item or a passage",
              "validation or test item in the knowledge", "knowledge item linked other than by disease",
              "knowledge image differs from the image dataset", "knowledge case differs from the text dataset",
              "knowledge case licence differs from its source",
              "knowledge passage differs from the library", "passage from an article of the text dataset's candidates",
              "disease table does not list the classes", "disease count differs from the sources"]
    lines += ["| Check | Failures |", "|---|---|"]
    lines += [f"| {c} | {len(failures.get(c, []))}" + (f" (for example {', '.join(failures[c][:3])})" if failures.get(c) else "") + " |"
              for c in checks]
    lines += ["", "**All checks passed.**" if not failed else f"**{len(failed)} checks failed.**", ""]
    if not have_library:
        lines += ["data/text/library/passages.jsonl was not found, so every passage counts as missing: build the library first.", ""]
    lines += ["### What the facts alone tell, with no photo", "",
              "A counting model learns from the 1,739 training questions which class (and which photo collection) each value of a "
              "fact points to, every class weighing the same, and answers the test questions from the facts alone. Balanced "
              "accuracy, with its 95% interval (bootstrap over lesions). Chance: 12.5% for the class, 50% for the collection.", "",
              "| Facts used | Test questions | Class | Photo collection |", "|---|---|---|---|"]
    for row in shortcuts:
        cells = [f"{100 * v:.1f}% ({100 * ci[0]:.1f}–{100 * ci[1]:.1f}%)" if v is not None else "–" for v, ci in (row["label"], row["collection"])]
        lines.append(f"| {row['facts']} | {row['n']} | {cells[0]} | {cells[1]} |")
    lines += ["", "How to read it: a multimodal score is worth what it adds to the photo alone (the image dataset's run) and to the "
              "facts alone (this table). Age carries real clinical signal (nevi are found in younger patients); which facts are "
              "missing tells the photo collection, and so mostly melanoma or nevus, and no clinic works that way. Each multimodal "
              "report therefore also gives the score on the questions with all three facts.", ""]
    return lines


def main(argv: list[str] | None = None) -> None:
    if argv:
        raise SystemExit("usage: python -m src.multimodal.audit")
    try:
        done = audit()
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    failed = {c: ids for c, ids in done["failures"].items() if ids}
    print(f"Checked: {done['questions']} questions, {done['knowledge']} knowledge items")
    for check, ids in failed.items():
        print(f"  FAILED {check}: {len(ids)} (for example {', '.join(ids[:5])})")
    print("All checks passed." if not failed else "Some checks failed.")
    print("What the facts alone tell (balanced accuracy on test: class, photo collection):")
    for row in done["shortcuts"]:
        print(f"  {row['facts']}: " + ", ".join("–" if v is None else f"{100 * v:.1f}%" for v, _ in (row["label"], row["collection"])))
    print(f"Written: {done['report']} (the audit section)")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1:])

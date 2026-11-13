"""Independent check of the text dataset, run where its texts are (they are not in Git):

    python -m src.text.audit

It reads dataset.csv, the question files in data/text/dataset/ and the downloaded sources, and checks, with code of its own (not
the build's), that every question:
    - has its file, matching its fingerprint, and no file is left over;
    - stands word for word in its source (the PMC-Patients patient, or the PubMed record's abstract or full text);
    - names neither its class (any name of the class in class_terms.csv, counting or not, except "mole") nor the class's short
      form, nor a short form its source defines for the class (such as "pyogenic granuloma (PG)"), and reports no test result
      (a biopsy, histology, pathology, immunohistochemistry, an excision, or what a microscope showed);
    - has at least 25 words, describes a lesion and a patient (an age or a sex), and has no sentence that puts
      a lesion inside the body (the mouth, the eye, the ear, the genital mucosa, the mediastinum ...), and tells what the
      lesion looks like (a size in mm or cm, a lesion type such as a nodule, or a colour, surface or shape);
    - is not the same as another question, and is not the same patient's case as a question of another article (they share at
      least 10% of the shorter one's runs of 12 words; questions of one article are kept together in one split anyway). Smaller
      overlaps, such as a common phrase, are only listed on the screen.
It writes the result into the audit section of dataset_report.md, and 5 questions per class (a fixed random choice) into
data/text/sources/audit_sample.txt (not committed) for anyone who wants to read real questions. It exits with an error when a check
fails.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from ..common import write_section
from ..evaluate import FILES, SYSTEM_FILES, TABLE_NAME
from .prepare import (ABSTRACTS, DATA, FEATURES, INSIDE, PMC_PATIENTS, REPORT, REPORT_HEAD, SHORT_FORMS, SOURCES, STEPS, TERMS, _json_list)

MIN_WORDS = 25
VAGUE = {"mole"}  # too vague to be a diagnosis (class_terms.csv): it does not reveal the class
RESULT = re.compile(r"\b(biops|histopatholog|histolog|patholog|dermatopatholog|immunohistochem|histochem|immunostain|immunoreactiv|"
                    r"excis|microscop|cytolog|cytoplasm|nucle|mitos|mitot|specimen|multinucleat|fibroblastic|histiocytic|aspirat|dermis|"
                    r"epidermis|collagen|macrophag|storiform|spindle|epithelioid|keratinocyt|lymphocyt|atypia)\w*")
SIZE = re.compile(r"\d\s*(mm|cm)\b", re.I)
TYPES = re.compile(r"\b(nodules?|papules?|plaques?|patch(es)?|macules?|ulcers?|spots?)\b")  # lesion words that tell what it looks like
DEFINED = r"(?=[A-Za-z]*[A-Z][A-Za-z]*[A-Z])[A-Za-z]{2,6}"  # an author's short form: 2 to 6 letters, 2 or more capitals
LESION = re.compile(r"\b(lesions?|nodules?|papules?|plaques?|patch(es)?|macules?|mass(es)?|tumou?rs?|ulcers?|growths?|spots?)\b")
SPELLING = (("’", "'"), ("naev", "nev"), ("haem", "hem"), ("oedem", "edem"), ("seborrhoe", "seborrhe"), ("tumour", "tumor"),
            ("oesophag", "esophag"))
PATIENT = re.compile(r"\b(\d+[- ]?(year|yr|month|week|day)s?[- ]?old|aged? \d+|\d+ years? of age|man|woman|boy|girl|male|female|"
                     r"infant|newborn|neonate|child|gentleman|lady)\b", re.I)  # a patient: an age or a sex
SAMPLE, SAMPLE_PER_CLASS = "audit_sample.txt", 5
SAMPLE_SEED = "text-audit-4"  # changed after rules were drawn from a reading of the sample: the next reading is of unseen questions
SHARED_RUN, SAME_PATIENT = 12, 0.10  # questions of two articles sharing this share of their runs of 12 words: one patient
DUPLICATE = f"question tells the same patient's case as a question of another article (at least {SAME_PATIENT:.0%} of its " \
            f"runs of {SHARED_RUN} words)"


def _plain(text: str) -> str:
    """Lower case, American spelling, words separated by single spaces, "'s" dropped."""
    text = text.lower()
    for british, american in SPELLING:
        text = text.replace(british, american)
    return " " + " ".join(re.findall(r"[a-z0-9]+", text.replace("'s", ""))) + " "


def _forms(word: str) -> set[str]:
    forms = {word, word + "s", word + "es"}
    if word.endswith("us"):
        forms.add(word[:-2] + "i")  # nevus, nevi
    if word.endswith("is"):
        forms.add(word[:-2] + "es")  # keratosis, keratoses
    if word.endswith("o"):
        forms.add(word[:-1] + "ines")  # lentigo, lentigines
    return forms


def _name_pattern(names: list[str]) -> re.Pattern:
    """Any of the names, in either word order for two-word names, with each word also in its plural forms."""
    parts = []
    for name in names:
        words = ["(?:" + "|".join(sorted(_forms(word))) + ")" for word in _plain(name).split()]
        if words:
            parts += [" ".join(words)] + ([" ".join(words[::-1])] if len(words) == 2 else [])
    return re.compile(r" (?:" + "|".join(parts) + r") ")


def audit(data: str | Path = DATA, show=None) -> dict:
    """Checks every question of the dataset (see the module's description); writes the audit section of dataset_report.md and
    the sample file; returns the failures per check."""
    data = Path(data)
    with (data / TABLE_NAME).open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    with (data / TERMS).open(encoding="utf-8", newline="") as f:
        names = defaultdict(list)
        for t in csv.DictReader(f):
            if t["term"].lower() not in VAGUE:
                names[t["class"]].append(t["term"])
    patterns = {label: _name_pattern(names[label]) for label in names}
    short = {label: re.compile(r"\b(?:" + "|".join(re.escape(f) for f in forms) + r")s?\b", re.I)
             for label, forms in SHORT_FORMS.items()}
    folder = data / FILES
    texts, failures = {}, defaultdict(list)
    for row in rows:
        path = folder / row["path"]
        if not path.is_file():
            failures["question file missing"].append(row["patient_uid"])
            continue
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != row["sha256"]:
            failures["question file differs from its fingerprint"].append(row["patient_uid"])
        texts[row["patient_uid"]] = content.decode("utf-8")
    listed = {(folder / r["path"]).resolve() for r in rows}
    extra = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*")
                   if p.is_file() and p.resolve() not in listed and p.name.lower() not in SYSTEM_FILES)
    failures["file in the dataset folder but not in dataset.csv"] += extra
    sources = _sources(data, rows, show)
    seen = {}
    for row in rows:
        uid, question = row["patient_uid"], texts.get(row["patient_uid"])
        if question is None:
            continue
        if not any(question in source for source in sources.get(uid, ())):
            failures["question not word for word in its source"].append(uid)
        plain = _plain(question)
        if patterns[row["label"]].search(plain) or (row["label"] in short and short[row["label"]].search(question)):
            failures["question names its class or its short form"].append(uid)
        if any(re.search(rf"\b{re.escape(form)}s?\b", question) for form in _defined(sources.get(uid, ()), patterns[row["label"]])):
            failures["question uses a short form its source defines for the class"].append(uid)
        if RESULT.search(plain):
            failures["question reports a test result"].append(uid)
        if len(question.split()) < MIN_WORDS:
            failures[f"question shorter than {MIN_WORDS} words"].append(uid)
        if not LESION.search(plain):
            failures["question describes no lesion"].append(uid)
        if not PATIENT.search(question):
            failures["question describes no patient"].append(uid)
        if any(LESION.search(_plain(sentence)) and any(f" {_plain(site).strip()} " in _plain(sentence) for site in INSIDE)
               for sentence in re.split(r"(?<=[.!?])\s+", question)):
            failures["question puts a lesion inside the body"].append(uid)
        if not (SIZE.search(question) or TYPES.search(plain) or any(f" {_plain(word).strip()} " in plain for word in FEATURES)):
            failures["question tells nothing of what the lesion looks like"].append(uid)
        key = " ".join(question.split())
        if key in seen:
            failures["question the same as another"].append(f"{uid} = {seen[key]}")
        seen.setdefault(key, uid)
    pairs = _shared_runs(rows, texts)
    failures[DUPLICATE] += [f"{a['patient_uid']} ~ {b['patient_uid']} ({a['split']}/{b['split']}, {share:.0%})"
                            for a, b, share, _ in pairs if share >= SAME_PATIENT]
    _write_sample(data, rows, texts)
    write_section(data / REPORT, REPORT_HEAD, STEPS, "audit", _summary(rows, failures))
    return {"rows": rows, "failures": dict(failures), "pairs": pairs, "summary": (data / REPORT).as_posix(),
            "sample": (data / SOURCES / SAMPLE).as_posix()}


def _shared_runs(rows: list[dict], texts: dict[str, str]) -> list[tuple[dict, dict, float, str]]:
    """The pairs of questions of different articles that share a run of SHARED_RUN words (plain, lower case), in the order of
    dataset.csv: each with how much of the shorter question's runs they share, and the first run they share."""
    runs, owners = {}, defaultdict(list)
    for row in rows:
        if row["patient_uid"] in texts:
            words = re.findall(r"[a-z0-9]+", texts[row["patient_uid"]].lower())
            runs[row["patient_uid"]] = [tuple(words[i:i + SHARED_RUN]) for i in range(len(words) - SHARED_RUN + 1)]
            for run in set(runs[row["patient_uid"]]):
                owners[run].append(row)
    pairs = {}
    for run, owned in owners.items():
        for n, a in enumerate(owned):
            for b in owned[n + 1:]:
                if a["pmid"] != b["pmid"]:
                    pairs.setdefault((a["patient_uid"], b["patient_uid"]), (a, b))
    found = []
    for a, b in pairs.values():
        mine, theirs = runs[a["patient_uid"]], set(runs[b["patient_uid"]])
        shared = [run for run in mine if run in theirs]
        found.append((a, b, len(set(shared)) / min(len(set(mine)), len(theirs)), " ".join(shared[0])))
    order = {row["patient_uid"]: n for n, row in enumerate(rows)}
    return sorted(found, key=lambda pair: (order[pair[0]["patient_uid"]], order[pair[1]["patient_uid"]]))


def _defined(sources: list[str], pattern: re.Pattern) -> set[str]:
    """The short forms the source texts define for the class (``pattern``): "pyogenic granuloma (PG)" or "PG (pyogenic
    granuloma)", a plural "s" dropped."""
    found = set()
    for source in sources:
        for m in re.finditer(rf"\(\s*({DEFINED})\s*[),;]", source):
            if pattern.search(_plain(" ".join(source[max(0, m.start() - 80):m.start()].split()[-5:]))):
                found.add(m.group(1))
        for m in re.finditer(rf"\b({DEFINED})\s*\(([^()]{{3,80}})\)", source):
            if pattern.search(_plain(m.group(2))):
                found.add(m.group(1))
    return {f[:-1] if f.endswith("s") and sum(c.isupper() for c in f) >= 2 else f for f in found}


def _sources(data: Path, rows: list[dict], show=None) -> dict[str, list[str]]:
    """The source texts of the dataset's cases: the PMC-Patients patient, or the record's full text and every run of its
    abstract's sections, joined as PubMed gives them."""
    found = defaultdict(list)
    wanted = {r["patient_uid"] for r in rows if not r["patient_uid"].startswith("pubmed-")}
    if wanted:
        if show:
            show("reading PMC-Patients for the source texts")
        for record in _json_list(data / SOURCES / "pmc_patients" / PMC_PATIENTS["file"]):
            if isinstance(record, dict) and record.get("patient_uid") in wanted:
                found[record["patient_uid"]].append(str(record["patient"]).strip())
    path = data / SOURCES / ABSTRACTS / "abstracts.jsonl"
    if path.is_file():
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    parts = [text for _, text in record.get("sections", [])]
                    texts = [record.get("fulltext") or ""] + [" ".join(parts[i:j]).strip() for i in range(len(parts))
                                                               for j in range(i + 1, len(parts) + 1)]
                    found[f"pubmed-{record['pmid']}"] += [t for t in texts if t]
    return found


def _write_sample(data: Path, rows: list[dict], texts: dict[str, str]) -> None:
    by_class = defaultdict(list)
    for row in rows:
        if row["patient_uid"] in texts:
            by_class[row["label"]].append(row)
    lines = []
    for label in sorted(by_class):
        chosen = sorted(by_class[label], key=lambda r: hashlib.sha256(f"{SAMPLE_SEED}:{r['patient_uid']}".encode()).hexdigest())
        for row in chosen[:SAMPLE_PER_CLASS]:
            lines += [f"== {label} | {row['patient_uid']} | {row['source']} | {row['split']}", texts[row["patient_uid"]], ""]
    (data / SOURCES).mkdir(parents=True, exist_ok=True)
    (data / SOURCES / SAMPLE).write_text("\n".join(lines), encoding="utf-8", newline="\n")


def _summary(rows, failures) -> list[str]:
    checks = ["question file missing", "question file differs from its fingerprint",
              "file in the dataset folder but not in dataset.csv", "question not word for word in its source",
              "question names its class or its short form", "question uses a short form its source defines for the class",
              "question reports a test result",
              f"question shorter than {MIN_WORDS} words", "question describes no lesion", "question describes no patient",
              "question puts a lesion inside the body", "question tells nothing of what the lesion looks like",
              "question the same as another", DUPLICATE]
    lines = ["# Audit: an independent check of the questions", "",
             "Written by `python -m src.text.audit`, on the computer that holds the texts, with code of its own (not the "
             "build's): every question file is checked against its fingerprint and its source, and for the class's names, short "
             "forms (also those its source defines) and test results it must not contain, and for a patient and a lesion on the "
             f"skin it must describe. {SAMPLE_PER_CLASS} questions per class are written into "
             "`data/text/sources/audit_sample.txt` for reading (not committed).", "",
             f"- Questions checked: {len(rows)}", "", "| Check | Failed | Examples |", "|---|---|---|"]
    for check in checks:
        found = failures.get(check, [])
        lines.append(f"| {check} | {len(found)} | {', '.join(found[:5])} |")
    lines += ["", "All checks passed." if not any(failures.values()) else "Some checks failed: see the table.", ""]
    return lines


def main(argv: list[str] | None = None) -> None:
    if argv:
        raise SystemExit("usage: python -m src.text.audit")
    try:
        done = audit(show=print)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    failed = {check: ids for check, ids in done["failures"].items() if ids}
    print(f"Questions checked: {len(done['rows'])}")
    for check, ids in failed.items():
        print(f"  FAILED {check}: {len(ids)} (for example {', '.join(ids[:5])})")
    if done["pairs"]:
        print(f"Questions of different articles that share {SHARED_RUN} words in a row (from {SAME_PATIENT:.0%} of the shorter "
              "question on: the same patient, a failure; below: a common phrase, for information):")
        for a, b, share, run in done["pairs"][:50]:
            print(f"  {a['patient_uid']} ({a['label']}, {a['split']}) ~ {b['patient_uid']} ({b['label']}, {b['split']}): "
                  f"{share:.0%} of the shorter question; shared: \"{run}\"")
    print("All checks passed." if not failed else "Some checks failed.")
    print(f"Written: {done['summary']} (the audit section) and {done['sample']} ({SAMPLE_PER_CLASS} questions per class to read; not in Git).")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1:])

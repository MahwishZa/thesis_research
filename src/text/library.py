"""The reference library for retrieval: the texts a model may look up when it answers a question of the text dataset. Run it
after the build:

    python -m src.text.library

Like RAG² (Sohn et al., NAACL 2025), which retrieves evenly from four corpora (PubMed, PMC, clinical practice guidelines and
textbooks), it has four corpora, all from public sources of the US National Library of Medicine, found by fixed rules with no
person choosing a text:
    pubmed      the abstracts of PubMed's reviews about each class: the class's headings as a main subject, on the skin, in
                English, about people, not case reports and not guidelines; the newest 200 per class
    pmc         the open full texts in PubMed Central of those reviews (the body's paragraphs; figures, tables and references
                left out): per class, the newest 25 that PMC gives in full
    guidelines  PubMed's clinical practice guidelines about each class (the same headings and filters): the newest 50 per class,
                each as its open full text where PMC gives it, else its abstract
    textbook    the chapters of StatPearls, the open medical textbook on NCBI's Bookshelf, whose title or summary (its
                "Continuing Education Activity") names a class by a name that counts in class_terms.csv, and whose title names
                no site other than the skin and no animal (the words of the cases step); their main text (paragraphs and list
                items), without that summary, the review questions, figures, tables and references. StatPearls is read from
                its archive in the Bookshelf's Open Access subset, NCBI's route for downloading Bookshelf texts in bulk
Every text is cut, word for word, into passages of whole paragraphs of at most 300 words (a longer paragraph at its sentence
ends). Nothing that could hold a test answer gets in: no case report, no article of the dataset's candidates, and no passage that
shares a run of 12 words with a question of the dataset.

It writes library.csv (committed: one row per passage, with its source, licence and fingerprint, not its text), the passages into
data/text/library/passages.jsonl (not committed: their licences allow research use only) and the library section of
dataset_report.md. The StatPearls archive (several gigabytes) is downloaded once into data/text/sources/library/statpearls/ (a
stopped download continues) and checked against its fingerprint on every later run; everything else is asked anew on every run
(PubMed grows every day, so the library is dated).
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
import tarfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

from ..common import fetch, write_section
from ..evaluate import FILES, TABLE_NAME
from .prepare import (ABSTRACT_CANDIDATES, ANIMALS, CANDIDATE_COLUMNS, CANDIDATES, DATA, EFETCH_BATCH, FULLTEXT_BATCH, FULLTEXT_SKIP,
                      HUMANS, IMAGE_TABLE, NOT_SKIN, REPORT, REPORT_HEAD, SKIN, SOURCES, STEPS, _checked_headings, _efetch,
                      _found, _licence, _matcher, _named, _pmc_articles, _read_csv, _search, _sentence_starts, _sha256, _stream,
                      _subject, _tokens, _write_csv, image_classes, read_table)

LIBRARY_TABLE = "library.csv"
LIBRARY_COLUMNS = ["passage_id", "corpus", "source", "title", "section", "year", "licence", "topics", "words", "sha256"]
FOLDER = "library"  # data/text/library/passages.jsonl, and the StatPearls pages in data/text/sources/library/statpearls/
CORPORA = ("pubmed", "pmc", "guidelines", "textbook")
NOT_CASES = '"case reports"[Publication Type]'
GUIDELINE = '(guideline[Publication Type] OR "practice guideline"[Publication Type])'
REVIEW_QUERY = f'{SKIN} AND review[Publication Type] NOT {GUIDELINE} NOT {NOT_CASES} AND english[Language] {HUMANS}'
GUIDELINE_QUERY = f'{SKIN} AND {GUIDELINE} NOT {NOT_CASES} AND english[Language] {HUMANS}'
REVIEWS_PER_CLASS, FULLTEXTS_PER_CLASS, GUIDELINES_PER_CLASS = 200, 25, 50  # the newest, per class
OPEN_ACCESS = "https://ftp.ncbi.nlm.nih.gov/pub/litarch/"  # the Bookshelf's Open Access subset: file_list.csv and the archives
STATPEARLS = "NBK430685"  # StatPearls' accession on the Bookshelf
ABSTRACT_LICENCE = "publisher's copyright (abstract in PubMed)"
LEFT_OUT = ("continuing education activity",)  # a chapter's sections left out: a summary for course credits
LAST = ("review questions", "references")  # a chapter's main text ends at the first of these sections
PASSAGE_WORDS, SHORTEST = 300, 20  # words in a passage: at most, and at least (a shorter piece is left out)
SHARED_RUN = 12  # a passage sharing this many words in a row with a question of the dataset is left out


def library(data: str | Path = DATA, classes: list[str] | None = None, show=None) -> dict:
    """Builds the four corpora (see the module's description); writes library.csv, data/text/library/passages.jsonl and the
    library section of dataset_report.md."""
    data = Path(data)
    classes = classes if classes is not None else image_classes(read_table(IMAGE_TABLE))
    terms, headings = _checked_headings(data, classes)
    runs = _question_runs(data)
    dataset_articles = _dataset_articles(data)
    day = datetime.now(timezone.utc).date().isoformat()
    say = show or (lambda message: None)

    reviews = _newest(classes, headings, REVIEW_QUERY, REVIEWS_PER_CLASS, dataset_articles, say, "reviews")
    guidelines = _newest(classes, headings, GUIDELINE_QUERY, GUIDELINES_PER_CLASS, dataset_articles, say, "guidelines")
    records = _records(sorted({p for found in (reviews, guidelines) for ids in found.values() for p in ids}, key=int), say)
    bodies = {}
    full = {label: _full_texts([records[p] for p in reviews[label] if p in records], bodies, say, label, FULLTEXTS_PER_CLASS)
            for label in classes}
    _full_texts([records[p] for ids in guidelines.values() for p in ids if p in records], bodies, say, "guidelines")

    documents = []
    for corpus, found in (("pubmed", reviews), ("guidelines", guidelines)):
        topics = defaultdict(list)
        for label in classes:
            for pmid in found[label]:
                topics[pmid].append(label)
        for pmid in sorted(topics, key=int):
            record = records.get(pmid)
            if not record:
                continue
            body = bodies.get(record["pmcid"].removeprefix("PMC")) if corpus == "guidelines" and record["pmcid"] else None
            if body:
                documents.append(_document(corpus, record["pmcid"], record, body["sections"], body["licence"], topics[pmid]))
            elif any(text for _, text in record["sections"]):
                sections = [["Abstract", [text for _, text in record["sections"] if text]]]
                documents.append(_document(corpus, f"PMID {pmid}", record, sections, ABSTRACT_LICENCE, topics[pmid]))
    topics, chosen = defaultdict(list), {}
    for label in classes:
        for record in full[label]:
            topics[record["pmcid"]].append(label)
            chosen[record["pmcid"]] = record
    for pmcid in sorted(chosen, key=lambda p: int(p.removeprefix("PMC"))):
        body = bodies[pmcid.removeprefix("PMC")]
        documents.append(_document("pmc", pmcid, chosen[pmcid], body["sections"], body["licence"], topics[pmcid]))
    textbook, archive = _textbook(data, terms, classes, say)
    documents += textbook

    rows, passages, dropped = _passages_of(documents, runs)
    folder = data / FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    partial = folder / "passages.jsonl.partial"
    partial.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in passages), encoding="utf-8", newline="\n")
    os.replace(partial, folder / "passages.jsonl")
    _write_csv(data / LIBRARY_TABLE, LIBRARY_COLUMNS, rows)
    write_section(data / REPORT, REPORT_HEAD, STEPS, "library",
                  _summary(day, classes, headings, reviews, guidelines, full, documents, rows, dropped, archive))
    return {"rows": rows, "documents": documents, "classes": classes, "dropped": dropped, "archive": archive,
            "table": (data / LIBRARY_TABLE).as_posix(), "passages": (folder / "passages.jsonl").as_posix(),
            "summary": (data / REPORT).as_posix()}


def _question_runs(data: Path) -> set[tuple[str, ...]]:
    """Every run of SHARED_RUN words (plain, lower case) in the dataset's questions."""
    table, folder = data / TABLE_NAME, data / FILES
    if not table.is_file():
        raise FileNotFoundError(f"{table.as_posix()} not found: run python -m src.text.prepare build first")
    runs = set()
    with table.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            path = folder / row["path"]
            if not path.is_file():
                raise FileNotFoundError(f"{path.as_posix()} not found: run python -m src.text.prepare build first")
            runs |= _runs(path.read_text(encoding="utf-8"))
    return runs


def _runs(text: str) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {tuple(words[i:i + SHARED_RUN]) for i in range(len(words) - SHARED_RUN + 1)}


def _dataset_articles(data: Path) -> set[str]:
    """The PubMed numbers of every candidate article of the dataset: none of them enters the library."""
    return {r["pmid"] for name, step in ((CANDIDATES, "cases"), (ABSTRACT_CANDIDATES, "abstracts"))
            for r in _read_csv(data / name, CANDIDATE_COLUMNS, step)}


def _newest(classes, headings, query, per_class, left_out, say, what) -> dict[str, list[str]]:
    """Per class: the newest articles PubMed finds for its headings (as a main subject) and the query, newest first, at most
    ``per_class``, without the dataset's articles."""
    found = {}
    for label in classes:
        ids = []
        if headings[label]:
            answer = _search("pubmed", f"{_subject(headings[label])} AND {query}", retmax=per_class, sort="pub_date")
            ids = [p for p in answer.get("idlist") or [] if p not in left_out]
        found[label] = ids
        say(f"{label}: {len(ids)} {what}")
    return found


def _records(pmids: list[str], say) -> dict[str, dict]:
    records = {}
    for start in range(0, len(pmids), EFETCH_BATCH):
        records.update((r["pmid"], r) for r in _efetch(pmids[start:start + EFETCH_BATCH]))
        say(f"{min(start + EFETCH_BATCH, len(pmids))} of {len(pmids)} PubMed records downloaded")
    asked = set(pmids)
    return {pmid: record for pmid, record in records.items() if pmid in asked}


def _full_texts(records: list[dict], bodies: dict, say, what: str, cap: int | None = None) -> list[dict]:
    """The first ``cap`` records (all when None), in their order, whose open full text PMC gives. PMC is asked, in that order,
    only as far as needed and only for numbers not asked before: ``bodies`` keeps every answer (None for an article PMC does
    not give in full)."""
    numbers = list(dict.fromkeys(r["pmcid"].removeprefix("PMC") for r in records if r["pmcid"]))

    def chosen() -> list[str] | None:  # None while an article before the cap is not asked yet
        given = []
        for number in numbers:
            if number not in bodies:
                return None
            if bodies[number]:
                given.append(number)
            if cap is not None and len(given) == cap:
                break
        return given
    while chosen() is None:
        batch = [n for n in numbers if n not in bodies][:FULLTEXT_BATCH]
        answer = _pmc_articles(batch)
        bodies.update((n, answer.get(n)) for n in batch)
        say(f"{what}: PMC asked for {sum(n in bodies for n in numbers)} of {len(numbers)} full texts")
    given = {f"PMC{n}" for n in chosen()}
    return list({r["pmcid"]: r for r in records if r["pmcid"] in given}.values())


def _document(corpus, source, record, sections, licence, topics) -> dict:
    return {"corpus": corpus, "source": source, "title": record["title"], "year": record["year"], "licence": licence,
            "topics": topics, "sections": sections}


# ---------------------------------------------------------------------------------------------------------------------------------
# The textbook: StatPearls on NCBI's Bookshelf


def _textbook(data: Path, terms: list[dict], classes: list[str], say) -> tuple[list[dict], dict]:
    """The StatPearls chapters whose title or summary (the "Continuing Education Activity") names a class, and whose title names
    no other site and no animal, read from the archive of the Bookshelf's Open Access subset; and the archive's record."""
    meta = _statpearls_archive(data / SOURCES / FOLDER / "statpearls", say)
    matcher = _matcher(terms)
    blocked = {tuple(_tokens(word)): word for word in NOT_SKIN + ANIMALS}
    documents, read, unreadable = [], 0, []
    with tarfile.open(data / SOURCES / FOLDER / "statpearls" / meta["file"], mode="r|gz") as archive:
        for member in archive:
            if not (member.isfile() and member.name.endswith(".nxml")):
                continue
            read += 1
            try:
                chapter = _chapter(archive.extractfile(member).read(), Path(member.name).stem)
            except ElementTree.ParseError:
                unreadable.append(member.name)
                continue
            named = _named(_tokens(f"{chapter['title']} {chapter['summary']}"), matcher)[0]
            topics = [label for label in classes if named[label]]
            if topics and not _found(_tokens(chapter["title"]), blocked) and chapter["sections"]:
                documents.append({"corpus": "textbook", "source": chapter["source"], "title": chapter["title"],
                                  "year": chapter["year"], "licence": chapter["licence"], "topics": topics,
                                  "sections": chapter["sections"]})
    if not read:
        raise ValueError(f"{meta['file']} holds no chapter files (.nxml): its form is not the expected one")
    say(f"StatPearls: {read} chapters read, {len(documents)} name a class")
    documents.sort(key=lambda d: d["source"])
    return documents, {**meta, "chapters": read, "unreadable": unreadable}


def _statpearls_archive(folder: Path, say) -> dict:
    """StatPearls' archive from the Bookshelf's Open Access subset (NCBI's route for downloading Bookshelf texts in bulk),
    downloaded once into ``folder`` (a stopped download continues), and checked against its fingerprint every time after."""
    record, pending = folder / "download.json", folder / "download.part.json"
    if record.is_file():
        meta = json.loads(record.read_text(encoding="utf-8"))
        if (folder / meta["file"]).is_file():
            if _sha256(folder / meta["file"]) != meta["sha256"]:
                raise ValueError(f"{(folder / meta['file']).as_posix()} has changed since it was downloaded (its fingerprint "
                                 f"differs): delete it and {record.name} to download it again")
            return meta
    folder.mkdir(parents=True, exist_ok=True)
    if pending.is_file():
        listed = json.loads(pending.read_text(encoding="utf-8"))
    else:
        listed = _listed_archive(_decoded(fetch(OPEN_ACCESS + "file_list.csv", "text/csv")))
        pending.write_text(json.dumps(listed), encoding="utf-8")
    url = OPEN_ACCESS + listed["path"]
    part = folder / (Path(listed["path"]).name + ".part")
    say(f"downloading StatPearls ({listed['path']}) into {folder.as_posix()}/; a stopped download continues when run again")
    _stream(url, part, say)
    meta = {"source": "StatPearls, NCBI Bookshelf Open Access subset", "accession": STATPEARLS, "file": Path(listed["path"]).name,
            "url": url, "listed": listed["row"], "bytes": part.stat().st_size, "sha256": _sha256(part),
            "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    os.replace(part, folder / meta["file"])
    record.write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    pending.unlink()
    return meta


def _decoded(content: bytes) -> str:
    """The file list as text: UTF-8, or else Latin-1 (some of its titles are not UTF-8, such as a "®" written as one byte; the
    paths and accessions it is read for are plain ASCII either way)."""
    try:
        return content.decode("utf-8-sig")
    except UnicodeDecodeError:
        return content.decode("latin-1")


def _listed_archive(text: str) -> dict:
    """StatPearls' row in the Open Access subset's file list: the archive's path and the row as listed."""
    rows = list(csv.reader(text.splitlines()))
    for row in rows[1:]:
        archive = [cell.strip() for cell in row if cell.strip().endswith(".tar.gz")]
        if archive and any(STATPEARLS in cell for cell in row):
            return {"path": archive[0], "row": dict(zip(rows[0], row))}
    raise ValueError(f"the Bookshelf's Open Access file list has no archive for StatPearls ({STATPEARLS}); its columns are: "
                     f"{', '.join(rows[0]) if rows else 'none'}")


def _chapter(content: bytes, stem: str) -> dict:
    """A chapter's NXML file: its Bookshelf accession (NBK...) or else its file's name, title, year (the latest in its metadata),
    licence, summary (its "Continuing Education Activity") and main text as [section title, paragraphs]."""
    root = ElementTree.fromstring(content)
    accession = [e.text.strip() for e in root.iter() if isinstance(e.tag, str) and e.tag.endswith("-id")
                 and re.fullmatch(r"NBK\d+", (e.text or "").strip())]
    title = next((t for t in (root.find(path) for path in (".//book-part-meta/title-group/title", ".//title-group/title",
                                                            ".//title")) if t is not None), None)
    years = [e.text.strip() for meta in root.iter("book-part-meta") for e in meta.iter("year") if (e.text or "").strip().isdigit()]
    licence = root.find(".//license")
    sections, summary = [], ""
    body = root.find(".//body")
    for section in [] if body is None else body.findall("sec"):
        name = " ".join((section.findtext("title") or "").split())
        if name.lower() in LAST:
            break
        blocks = _blocks(section)
        if name.lower() in LEFT_OUT:
            summary = " ".join(blocks)
        elif blocks:
            sections.append([name, blocks])
    return {"source": accession[0] if accession else stem,
            "title": " ".join("".join(title.itertext()).split()) if title is not None else "", "year": max(years, default=""),
            "licence": _licence(licence), "summary": summary, "sections": sections}


def _blocks(section) -> list[str]:
    """The paragraphs of a section and of the sections inside it, in order: each paragraph, and each item of a list that stands
    on its own (titles, figures and tables left out)."""
    found = []
    for child in section:
        if child.tag == "p":
            found.append(" ".join(_text(child).split()))
        elif child.tag == "list":
            found += [" ".join(_text(item).split()) for item in child.findall("list-item")]
        elif child.tag == "sec":
            found += _blocks(child)
    return [block for block in found if block]


def _text(element) -> str:
    """An element's running text, figures and tables left out; paragraphs, lists and list items inside it are kept apart by a
    space, so that their words do not run together."""
    if element.tag in FULLTEXT_SKIP:
        return ""
    parts = [element.text or ""]
    for child in element:
        parts += [f" {_text(child)} " if child.tag in ("p", "list", "list-item") else _text(child), child.tail or ""]
    return "".join(parts)


# ---------------------------------------------------------------------------------------------------------------------------------
# Passages


def _pieces(paragraph: str) -> list[str]:
    """A paragraph as it is, or, when it has more than PASSAGE_WORDS words, cut at its sentence ends into pieces of at most
    PASSAGE_WORDS words (a single longer sentence stays whole)."""
    if len(paragraph.split()) <= PASSAGE_WORDS:
        return [paragraph]
    starts = _sentence_starts(paragraph)
    sentences = [paragraph[a:b].strip() for a, b in zip(starts, starts[1:] + [len(paragraph)])]
    pieces = []
    for sentence in sentences:
        if pieces and len(pieces[-1].split()) + len(sentence.split()) <= PASSAGE_WORDS:
            pieces[-1] += " " + sentence
        else:
            pieces.append(sentence)
    return pieces


def _passages(sections: list) -> list[tuple[str, str]]:
    """A text's passages, as (section title, text): whole paragraphs of one section, joined while they stay within
    PASSAGE_WORDS words; pieces shorter than SHORTEST words are left out."""
    found = []
    for title, paragraphs in sections:
        current = []
        for piece in (piece for paragraph in paragraphs for piece in _pieces(paragraph)):
            if current and sum(len(p.split()) for p in current) + len(piece.split()) > PASSAGE_WORDS:
                found.append((title, "\n\n".join(current)))
                current = []
            current.append(piece)
        if current:
            found.append((title, "\n\n".join(current)))
    return [(title, text) for title, text in found if len(text.split()) >= SHORTEST]


def _passages_of(documents: list[dict], runs: set) -> tuple[list[dict], list[dict], Counter]:
    """The table rows and the passages of every document, in the order of the corpora; leaves out a passage that shares a run of
    SHARED_RUN words with a question, and a passage the same as an earlier one."""
    rows, passages, dropped, seen = [], [], Counter(), set()
    order = {corpus: n for n, corpus in enumerate(CORPORA)}
    for document in sorted(documents, key=lambda d: order[d["corpus"]]):
        number = 0
        for section, text in _passages(document["sections"]):
            if _runs(text) & runs:
                dropped["shares a run of words with a question"] += 1
                continue
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest in seen:
                dropped["the same as an earlier passage"] += 1
                continue
            seen.add(digest)
            number += 1
            source = document["source"].removeprefix("PMID ")
            row = {"passage_id": f"{document['corpus']}-{source}-{number}", "corpus": document["corpus"],
                   "source": document["source"], "title": document["title"], "section": section, "year": document["year"],
                   "licence": document["licence"], "topics": "; ".join(document["topics"]), "words": len(text.split()),
                   "sha256": digest}
            rows.append(row)
            passages.append({**{c: row[c] for c in LIBRARY_COLUMNS if c not in ("words", "sha256")}, "text": text})
    return rows, passages, dropped


def _summary(day, classes, headings, reviews, guidelines, full, documents, rows, dropped, archive) -> list[str]:
    by_corpus = defaultdict(list)
    for document in documents:
        by_corpus[document["corpus"]].append(document)
    names = {"pubmed": "PubMed: abstracts of reviews", "pmc": "PMC: open full texts of reviews",
             "guidelines": "Clinical practice guidelines", "textbook": "Textbook: StatPearls chapters"}
    lines = ["# Library: the reference texts for retrieval", "",
             f"Written by `python -m src.text.library` on {day}. Four corpora, as in RAG² (PubMed, PMC, clinical practice "
             "guidelines, textbooks), from public sources of the US National Library of Medicine, found by fixed rules with no "
             "person choosing a text: the abstracts of PubMed's reviews about each class (its headings as a main subject, on the "
             f"skin, in English, about people; not case reports or guidelines; the newest {REVIEWS_PER_CLASS} per class); the "
             f"open full texts of those reviews in PubMed Central (per class the newest {FULLTEXTS_PER_CLASS} that PMC gives in "
             f"full); PubMed's clinical practice guidelines about each class (the newest {GUIDELINES_PER_CLASS} per class, as open "
             "full text where PMC gives it, else as abstract); and the chapters of StatPearls (NCBI's Bookshelf) whose title or "
             "summary (its \"Continuing Education Activity\") names a class by a name that counts in `class_terms.csv` and whose title names no other site and no "
             f"animal. Every text is cut word for word into passages of whole paragraphs of at most {PASSAGE_WORDS} words. "
             "`library.csv` lists the passages (source, licence, fingerprint); their texts are in `data/text/library/` (not in "
             "Git).", "",
             "## Per class", "",
             "| Class | MeSH headings (main topic) | Reviews (abstracts) | Open full texts of reviews | Guidelines | StatPearls "
             "chapters |", "|---|---|---|---|---|---|"]
    for label in classes:
        lines.append(f"| {label} | {'; '.join(headings[label]) or 'none'} | {len(reviews[label])} | {len(full[label])} | "
                     f"{len(guidelines[label])} | {sum(label in d['topics'] for d in by_corpus['textbook'])} |")
    lines += ["", "## Per corpus", "", "| Corpus | Documents | Passages | Words | Licences (documents) |", "|---|---|---|---|---|"]
    for corpus in CORPORA:
        mine = [r for r in rows if r["corpus"] == corpus]
        licences = Counter(d["licence"] or "not stated" for d in by_corpus[corpus])
        lines.append(f"| {names[corpus]} | {len(by_corpus[corpus])} | {len(mine)} | {sum(r['words'] for r in mine)} | "
                     + "; ".join(f"{licence}: {n}" for licence, n in sorted(licences.items())) + " |")
    lines += [f"| all | {len(documents)} | {len(rows)} | {sum(r['words'] for r in rows)} | |", "",
              "## Checks", "",
              "- Case reports and the dataset's candidate articles are not searched for or are left out, so no case of the "
              "dataset is in the library.",
              f"- Passages left out because they share a run of {SHARED_RUN} words with a question of the dataset: "
              f"{dropped['shares a run of words with a question']}",
              f"- Passages left out because they are the same as an earlier one: {dropped['the same as an earlier passage']}", "",
              f"Query for reviews, with each class's headings as a main subject: {REVIEW_QUERY}.",
              f"Query for guidelines: {GUIDELINE_QUERY}.", "",
              f"Textbook: StatPearls ({STATPEARLS}) from the Bookshelf's Open Access subset, `{archive['file']}`, "
              f"{archive['bytes']} bytes, SHA-256 `{archive['sha256']}` (checked), downloaded {archive['retrieved_utc'][:10]}; "
              f"chapters read: {archive['chapters']}; chapter files that could not be read: {len(archive['unreadable'])}"
              + (f" ({', '.join(archive['unreadable'][:5])})" if archive["unreadable"] else ""), "",
              "## Not done in this step", "",
              "- No person has read the passages. A passage is about its document's class(es) (`topics`), not checked line by line.",
              "- The texts are under their publishers' copyright or licences (in `library.csv`), so they stay on this computer and "
              "are used for research only.",
              "- Retrieval (how passages are chosen for a question, evenly from the four corpora as in RAG²) belongs to the "
              "evaluation, not to the library.", ""]
    return lines


def main(argv: list[str] | None = None) -> None:
    if argv:
        raise SystemExit("usage: python -m src.text.library")
    try:
        done = library(show=print)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    for corpus in CORPORA:
        mine = [r for r in done["rows"] if r["corpus"] == corpus]
        print(f"  {corpus}: {sum(d['corpus'] == corpus for d in done['documents'])} documents, {len(mine)} passages, "
              f"{sum(r['words'] for r in mine)} words")
    print(f"Passages left out: {sum(done['dropped'].values())} ({', '.join(f'{k}: {n}' for k, n in done['dropped'].items()) or 'none'})")
    print(f"Written: {done['table']}, the passages in {done['passages']} (not in Git) and {done['summary']}. Commit "
          f"{done['table']} and {done['summary']}.")


if __name__ == "__main__":
    main(sys.argv[1:])

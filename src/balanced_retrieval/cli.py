"""``python -m balanced_retrieval``: build the four corpora, search them with the rationales, and describe the result.

    fetch     download the documents of one corpus (pubmed, pmc, guidelines or textbooks) as JSON lines
    chunk     cut documents (JSON lines) into passages and write one corpus folder
    embed     write the MedCPT article vectors of one corpus (resumable)
    precision-check  compare 32-bit and 16-bit encoding of a sample (speed, and how much the vectors differ)
    check     list the corpora under a folder, with their sizes next to the paper's
    retrieve  for every question: search all corpora with its rationale, pool, re-rank, keep the best k (resumable)
    report    describe the evidence lists of one ``retrieve`` folder

Exit status: 0 done; 1 the setup is wrong (message on stderr); 2 a mistyped option (usage message); 130 interrupted (run
the same command again to continue).
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from rationale_query.batch import plan
from rationale_query.cli import Streams, tolerate_unencodable
from rationale_query.generator import text_sha256
from rationale_query.search_text import dense_query
from rationale_query.store import (
    ConfigMismatch,
    StoreError,
    load_rationales,
    question_hash,
    read_questions,
    write_atomic,
)

from .chunking import DEFAULT_OVERLAP, DEFAULT_WORDS
from .diagnostics import summarise_lists
from .pool import CORPUS_ORDER
from .records import RECORDS_FILE, RetrievalRecord, RetrievalStore
from .retrieve import RERANK_AGAINST, balanced_retrieve

#: Passages per corpus in the paper's corpora (Table A1), for comparison only.
PAPER_PASSAGES = {"pubmed": 69_700_000, "pmc": 46_300_000, "guidelines": 607_000, "textbooks": 134_000}
SUMMARY_FILE = "summary.json"
RELEVANT = ("top_k", "rerank_against", "corpora", "query_encoder", "cross_encoder")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="balanced_retrieval", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="download the documents of one corpus into a JSON-lines file")
    fetch.add_argument("source", choices=CORPUS_ORDER)
    fetch.add_argument("--out", type=Path, default=None, help="the JSON-lines file to write (a pubmed or pmc download can be resumed)")
    fetch.add_argument("--as-of", help="pubmed: only records published up to this date, YYYY-MM-DD (names the snapshot)")
    fetch.add_argument("--count-only", action="store_true", help="pubmed: only say how many records the scope finds")
    fetch.add_argument("--pubmed-file", type=Path, help="pmc: the pubmed download whose records get their full text")
    fetch.add_argument("--limit", type=int, default=None, help="pubmed, pmc: stop after this many identifiers (a trial run)")
    fetch.add_argument("--file", default=None, help="guidelines: the file of epfl-llm/guidelines to use (default: the open one)")
    fetch.add_argument("--revision", default=None, help="guidelines, textbooks: the revision of the dataset on the hub")
    fetch.add_argument("--quiet", action="store_true", help="no progress lines")

    chunk = sub.add_parser("chunk", help="cut documents into passages and write one corpus folder")
    chunk.add_argument("--corpus", choices=CORPUS_ORDER, required=True)
    chunk.add_argument("--documents", type=Path, required=True, help="JSON lines: one {doc_key, text} per document")
    chunk.add_argument("--root", type=Path, required=True, help="folder that holds one folder per corpus")
    chunk.add_argument("--words", type=int, default=DEFAULT_WORDS, help="passage length in words")
    chunk.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP, help="words shared by neighbouring passages")
    chunk.add_argument("--overwrite", action="store_true", help="replace an existing corpus (its embeddings go too)")

    embed = sub.add_parser("embed", help="write the MedCPT article vectors of one corpus (resumable)")
    embed.add_argument("--root", type=Path, required=True)
    embed.add_argument("--corpus", choices=CORPUS_ORDER, required=True)
    embed.add_argument("--encoder", default="ncbi/MedCPT-Article-Encoder")
    embed.add_argument("--revision", default=None)
    embed.add_argument("--device", default="auto", help="auto, cpu or cuda")
    embed.add_argument("--batch-size", type=int, default=32)
    embed.add_argument("--dtype", choices=("float32", "float16"), default="float32", help="how the vectors are stored")
    embed.add_argument("--compute-dtype", choices=("float32", "float16"), default="float32", help="the arithmetic of the encoder; float16 is faster and is recorded")
    embed.add_argument("--save-every", type=int, default=2048, help="rows between two saves of the progress")
    embed.add_argument("--local-files-only", action="store_true", help="do not contact the Hugging Face hub")
    embed.add_argument("--quiet", action="store_true", help="no progress lines")

    prec = sub.add_parser("precision-check", help="compare 32-bit and 16-bit encoding of a sample of one corpus (speed and difference)")
    prec.add_argument("--root", type=Path, required=True)
    prec.add_argument("--corpus", choices=CORPUS_ORDER, required=True)
    prec.add_argument("--samples", type=int, default=1000, help="passages in the sample, spread evenly over the corpus")
    prec.add_argument("--encoder", default="ncbi/MedCPT-Article-Encoder")
    prec.add_argument("--revision", default=None)
    prec.add_argument("--device", default="auto", help="auto, cpu or cuda")
    prec.add_argument("--batch-size", type=int, default=32)
    prec.add_argument("--local-files-only", action="store_true", help="do not contact the Hugging Face hub")

    check = sub.add_parser("check", help="list the corpora under a folder, with the paper's sizes")
    check.add_argument("--root", type=Path, required=True)

    get = sub.add_parser("retrieve", help="search, pool and re-rank for every question (resumable)")
    get.add_argument("--questions", type=Path, required=True, help="the questions file of the rationales")
    get.add_argument("--rationales", type=Path, required=True, help="the output folder of rationale_query generate")
    get.add_argument("--root", type=Path, required=True, help="folder that holds one folder per corpus")
    get.add_argument("--out", type=Path, required=True, help="output folder (new, or the same as an earlier run)")
    get.add_argument("--top-k", type=int, required=True, help="candidates taken from each corpus and passages kept")
    get.add_argument("--corpora", nargs="+", choices=CORPUS_ORDER, default=list(CORPUS_ORDER))
    get.add_argument("--rerank-against", choices=RERANK_AGAINST, default="question", help="the paper says question")
    get.add_argument("--query-encoder", default="ncbi/MedCPT-Query-Encoder")
    get.add_argument("--cross-encoder", default="ncbi/MedCPT-Cross-Encoder")
    get.add_argument("--device", default="auto", help="auto, cpu or cuda")
    get.add_argument("--batch-size", type=int, default=16, help="pairs per call of the cross-encoder")
    get.add_argument("--limit", type=int, default=None, help="only the first N questions of the file")
    get.add_argument("--local-files-only", action="store_true", help="do not contact the Hugging Face hub")
    get.add_argument("--quiet", action="store_true", help="no progress lines")

    rep = sub.add_parser("report", help="describe the evidence lists of one retrieve folder")
    rep.add_argument("folder", type=Path)
    rep.add_argument("--json", type=Path, default=None, help="also write the numbers to this file")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    article_encoder_factory: Callable[[argparse.Namespace], Any] | None = None,
    query_encoder_factory: Callable[[argparse.Namespace], Any] | None = None,
    scorer_factory: Callable[[argparse.Namespace], Any] | None = None,
    streams: Streams | None = None,
    client_factory: Callable[[], Any] | None = None,
) -> int:
    streams = streams or Streams(sys.stdout, sys.stderr)
    tolerate_unencodable(streams.out, streams.err)
    args = _parser().parse_args(argv)
    try:
        if args.command == "fetch":
            return _fetch(args, streams, client_factory)
        if args.command == "chunk":
            return _chunk(args, streams)
        if args.command == "embed":
            return _embed(args, streams, article_encoder_factory or _default_article_encoder)
        if args.command == "precision-check":
            return _precision_check(args, streams, article_encoder_factory or _default_article_encoder)
        if args.command == "check":
            return _check(args, streams)
        if args.command == "retrieve":
            return _retrieve(args, streams, query_encoder_factory or _default_query_encoder, scorer_factory or _default_scorer)
        return _report(args, streams)
    except KeyboardInterrupt:
        streams.err.write("\ninterrupted; run the same command again to continue\n")
        return 130
    except (StoreError, OSError, ValueError, TypeError, ImportError, IndexError) as exc:
        streams.err.write(f"error: {exc}\n")
        return 1


def _needs_packages(exc: ImportError) -> ImportError:
    return ImportError(f"this command needs numpy, torch and transformers ({exc})")


def _disk() -> Any:
    """The module that reads and writes corpus folders; it needs numpy, which importing this command line does not."""
    try:
        from . import corpus
    except ImportError as exc:
        raise _needs_packages(exc) from exc
    return corpus


def _fetch(args: argparse.Namespace, streams: Streams, client_factory: Callable[[], Any] | None) -> int:
    if args.out is None and not args.count_only:
        raise ValueError("fetch needs --out, the JSON-lines file to write")
    last = 0

    def progress(done: int, total: int) -> None:
        nonlocal last
        if not args.quiet and (done == total or done - last >= 5000):
            last = done
            streams.err.write(f"  {done} of {total} identifiers fetched\n")

    if args.source in ("pubmed", "pmc"):
        from datetime import date

        from .ncbi import Client, count_hits
        from .pmc import fetch_pmc
        from .pubmed import fetch_pubmed
        from .scope import pubmed_scope

        client = client_factory() if client_factory else Client()
        if args.source == "pubmed":
            if not args.as_of:
                raise ValueError("fetch pubmed needs --as-of YYYY-MM-DD, the date that names the snapshot")
            as_of = date.fromisoformat(args.as_of)
            if args.count_only:
                streams.out.write(f"the scope finds {count_hits(client, 'pubmed', pubmed_scope(), as_of):,} records up to {as_of}\n")
                return 0
            state = fetch_pubmed(client, args.out, as_of=as_of, limit=args.limit, progress=progress)
        else:
            if not args.pubmed_file:
                raise ValueError("fetch pmc needs --pubmed-file, the pubmed download whose records get their full text")
            state = fetch_pmc(client, args.out, pubmed_file=args.pubmed_file, limit=args.limit, progress=progress)
        streams.out.write(
            f"{args.source}: {state['documents']} documents from {state['done']} of {state['ids']} identifiers in {args.out}\n"
        )
        for reason, number in sorted(state.get("notes", {}).items()):
            streams.out.write(f"  left out: {number} ({reason})\n")
        if state["done"] < state["ids"]:
            streams.out.write("  not finished; run the same command again to continue\n")
        return 0
    if args.source == "guidelines":
        from .guidelines import download_guidelines

        info = download_guidelines(args.out, file=args.file, revision=args.revision)
    else:
        from .textbooks import download_textbooks

        info = download_textbooks(args.out, revision=args.revision)
    streams.out.write(f"{args.source}: {info['documents']} documents from {info['source']} (revision {info['revision']}) in {args.out}\n")
    return 0


def _chunk(args: argparse.Namespace, streams: Streams) -> int:
    disk = _disk()
    manifest = disk.write_corpus(
        args.root, args.corpus, disk.read_documents(args.documents), words=args.words, overlap=args.overlap, overwrite=args.overwrite
    )
    streams.out.write(
        f"{args.corpus}: {manifest['documents']} documents gave {manifest['passages']} passages of {args.words} words "
        f"(overlap {args.overlap}) in {args.root / args.corpus}\n"
    )
    skipped = manifest["documents_without_words"]
    if skipped:
        streams.out.write(f"  {skipped} documents without any word were skipped\n")
    if manifest["documents_without_key"]:
        streams.out.write(f"  {manifest['documents_without_key']} documents have no doc_key; they never count as duplicates\n")
    streams.out.write(f"  the paper's {args.corpus} corpus has about {PAPER_PASSAGES[args.corpus]:,} passages\n")
    return 0


def _default_article_encoder(args: argparse.Namespace) -> Any:
    try:
        from .embedding import ArticleEncoder

        return ArticleEncoder.from_pretrained(
            args.encoder,
            revision=args.revision,
            device=args.device,
            batch_size=args.batch_size,
            local_files_only=args.local_files_only,
            compute_dtype=getattr(args, "compute_dtype", "float32"),
        )
    except ImportError as exc:
        raise _needs_packages(exc) from exc


def _embed(args: argparse.Namespace, streams: Streams, factory: Callable[[argparse.Namespace], Any]) -> int:
    try:
        from .embedding import embed_corpus
    except ImportError as exc:
        raise _needs_packages(exc) from exc
    corpus = _disk().Corpus(args.root, args.corpus)
    encoder = factory(args)
    device = getattr(encoder, "info", {}).get("device") if isinstance(getattr(encoder, "info", None), dict) else None
    if device and not args.quiet:
        streams.err.write(f"  encoding on the {device}\n")
    last = 0
    first: tuple[float, int] | None = None  # when and where this run's first batch was finished

    def progress(done: int, total: int) -> None:
        nonlocal last, first
        now = time.monotonic()
        if first is None:
            first = (now, done)
        if args.quiet or (done < total and done - last < 10_000):
            return
        last = done
        speed = ""
        if done > first[1] and now > first[0]:
            rate = (done - first[1]) / (now - first[0])
            speed = f" ({rate:.0f} passages a second; about {(total - done) / rate / 60:.0f} minutes left)"
        streams.err.write(f"  embedded {done} of {total} passages{speed}\n")

    state = embed_corpus(corpus, encoder, batch_size=args.batch_size, save_every=args.save_every, dtype=args.dtype, progress=progress)
    arithmetic = "" if args.compute_dtype == "float32" else f", {args.compute_dtype} arithmetic"
    streams.out.write(f"{args.corpus}: {state['rows']} vectors written to {corpus.directory} ({args.dtype}{arithmetic}, not normalised)\n")
    return 0


def _precision_check(args: argparse.Namespace, streams: Streams, factory: Callable[[argparse.Namespace], Any]) -> int:
    try:
        from .embedding import compare_precision
    except ImportError as exc:
        raise _needs_packages(exc) from exc
    if args.samples < 2:
        raise ValueError("--samples must be at least 2")
    corpus = _disk().Corpus(args.root, args.corpus)
    step = max(1, len(corpus) // args.samples)
    rows = list(range(0, len(corpus), step))[: args.samples]
    texts = [passage["text"] for passage in corpus.passages(rows)]
    full = factory(argparse.Namespace(**{**vars(args), "compute_dtype": "float32"}))
    half = factory(argparse.Namespace(**{**vars(args), "compute_dtype": "float16"}))
    result = compare_precision(texts, full, half)
    top = next(key for key in result if key.endswith("_overlap"))
    streams.out.write(
        f"{args.corpus}: {result['texts']} passages encoded twice on the {full.info.get('device')}\n"
        f"  speed: {result['texts'] / result['seconds_full']:.0f} a second in 32-bit, {result['texts'] / result['seconds_half']:.0f} a second in 16-bit\n"
        f"  cosine between the two vectors of a passage: mean {result['cosine_mean']:.6f}, lowest {result['cosine_min']:.6f}\n"
        f"  with {result['queries']} passages as queries: the same {top.removesuffix('_overlap')[3:]} nearest passages {result[top]:.1%} of the time, "
        f"the same nearest passage {result['nearest_same']:.1%} of the time\n"
    )
    return 0


def _check(args: argparse.Namespace, streams: Streams) -> int:
    disk = _disk()
    found = 0
    for name in CORPUS_ORDER:
        if not (args.root / name / disk.MANIFEST_FILE).exists():
            streams.out.write(f"{name}: not built\n")
            continue
        found += 1
        corpus = disk.Corpus(args.root, name)
        manifest = corpus.manifest
        state = corpus.embedding_state()
        if state is None:
            embedded = "not embedded"
        elif state["complete"]:
            embedded = f"embedded ({state['settings']['encoder']}, {state['settings']['dtype']})"
        else:
            embedded = f"embedding stopped at {state['rows_done']} of {state['rows']}"
        chunking = manifest["chunking"]
        streams.out.write(
            f"{name}: {manifest['passages']:,} passages from {manifest['documents']:,} documents "
            f"({chunking['size']} words, overlap {chunking['overlap']}); {embedded}; paper: {PAPER_PASSAGES[name]:,}\n"
        )
    if not found:
        raise disk.CorpusError(f"no corpus under {args.root}")
    return 0


def _default_query_encoder(args: argparse.Namespace) -> Any:
    try:
        from rationale_query.encoder import QueryEncoder

        return QueryEncoder.from_pretrained(args.query_encoder, device=args.device, batch_size=1, local_files_only=args.local_files_only)
    except ImportError as exc:
        raise _needs_packages(exc) from exc


def _default_scorer(args: argparse.Namespace) -> Any:
    try:
        from .scorer import CrossEncoderScorer

        return CrossEncoderScorer.from_pretrained(
            args.cross_encoder, device=args.device, batch_size=args.batch_size, local_files_only=args.local_files_only
        )
    except ImportError as exc:
        raise _needs_packages(exc) from exc


def _stored_query_vectors(folder: Path, texts: list[str], ids: list[str], encoder_name: str, np: Any) -> tuple[Any, dict] | None:
    """The vectors that ``rationale_query encode`` wrote and their description, if made from exactly these texts; else None."""
    vectors_path, meta_path = folder / "query_vectors.npy", folder / "query_vectors.json"
    if not (vectors_path.exists() and meta_path.exists()):
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    same = (
        meta.get("ids") == ids
        and meta.get("encoder") == encoder_name
        and meta.get("max_length") == 512
        and meta.get("rationales_sha256") == text_sha256(json.dumps(texts, ensure_ascii=False))
    )
    return (np.load(vectors_path), meta) if same else None


class _ScoreLog:
    """Passes the scores through and remembers them, so the record can show each kept passage's re-ranking score."""

    def __init__(self, scorer: Callable[..., Sequence[float]]) -> None:
        self.scorer = scorer
        self.by_object: dict[int, float] = {}
        self.pool_size = 0

    def __call__(self, query: str, candidates: Sequence[Any]) -> Sequence[float]:
        scores = list(self.scorer(query, candidates))
        self.by_object = {id(c): float(s) for c, s in zip(candidates, scores, strict=False)}
        self.pool_size = len(candidates)
        return scores


def _lookup_searcher(corpus: Any, table: dict[str, tuple[Any, Any]], k: int) -> Callable[[str, int], list[dict]]:
    def search(query_text: str, asked: int) -> list[dict]:
        if asked != k:
            raise ValueError(f"the searches were made for k = {k}, not {asked}")
        rows, scores = table[query_text]
        passages = corpus.passages([int(row) for row in rows])
        return [dict(p, row=int(r), dense_score=float(s)) for p, r, s in zip(passages, rows, scores, strict=True)]

    return search


def _check_settings(store: RetrievalStore, args: argparse.Namespace) -> None:
    """Refuse other settings for a folder at once, before anything is loaded, even if no question is left to do."""
    if not store.config_path.exists():
        return
    recorded = store.read_config()
    asked = {"top_k": args.top_k, "rerank_against": args.rerank_against, "corpora": sorted(name for name in CORPUS_ORDER if name in args.corpora)}
    before = {"top_k": recorded.get("top_k"), "rerank_against": recorded.get("rerank_against"), "corpora": sorted(recorded.get("corpora", {}))}
    differing = [f"{name}: recorded {before[name]!r}, now {asked[name]!r}" for name in asked if before[name] != asked[name]]
    if differing:
        raise ConfigMismatch(f"{args.out} was made with other settings ({'; '.join(differing)}). Use a new output folder for the new settings.")


def _retrieve(
    args: argparse.Namespace,
    streams: Streams,
    query_factory: Callable[[argparse.Namespace], Any],
    scorer_factory: Callable[[argparse.Namespace], Any],
) -> int:
    try:
        import numpy as np

        from .search import ExactIndex
    except ImportError as exc:
        raise _needs_packages(exc) from exc
    questions = read_questions(args.questions)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1")
        questions = questions[: args.limit]
    k = args.top_k
    if isinstance(k, bool) or k < 1:
        raise ValueError("--top-k must be at least 1")
    rationales = load_rationales(args.rationales, questions)
    texts = {q.id: dense_query(rationales[q.id]) for q in questions}
    store = RetrievalStore(args.out)
    _check_settings(store, args)
    pending = plan(questions, store.load(), args.out).pending
    if not pending:
        streams.out.write(f"nothing to do: all {len(questions)} questions already have their evidence in {args.out}\n")
        return _write_report(args.out, streams, None)

    names = [name for name in CORPUS_ORDER if name in args.corpora]
    corpora = [_disk().Corpus(args.root, name) for name in names]
    matrices = {corpus.name: corpus.embeddings() for corpus in corpora}

    all_ids = [q.id for q in questions]
    all_texts = [texts[i] for i in all_ids]
    stored = _stored_query_vectors(args.rationales, all_texts, all_ids, args.query_encoder, np)
    if stored is not None:
        stored_vectors, meta = stored
        by_id = dict(zip(all_ids, stored_vectors, strict=True))
        streams.err.write("using the query vectors that rationale_query encode wrote\n")
        vectors = np.vstack([by_id[q.id] for q in pending])
        query_info = {key: meta.get(key) for key in ("encoder", "revision", "max_length")}
    else:
        query_encoder = query_factory(args)
        vectors = query_encoder.encode([texts[q.id] for q in pending])
        query_info = {key: query_encoder.info.get(key) for key in ("encoder", "revision", "max_length")}
    scorer = scorer_factory(args)
    cross = {key: scorer.info.get(key) for key in ("cross_encoder", "revision", "max_length")}

    config = {
        "top_k": k,
        "rerank_against": args.rerank_against,
        "corpora": {
            corpus.name: {
                "passages_sha256": corpus.manifest["passages_sha256"],
                "chunking": corpus.manifest["chunking"],
                "embeddings": corpus.embedding_state()["settings"],
            }
            for corpus in corpora
        },
        "query_encoder": query_info,
        "cross_encoder": cross,
        "runtime": {"device": getattr(scorer, "device", None), "batch_size": args.batch_size},
    }
    store.bind(config, RELEVANT)
    for record in store.load(repair=True).values():
        if record.id in texts and record.rationale_sha256 != text_sha256(texts[record.id]):
            raise StoreError(f"{record.id} is in {args.out} with evidence found for another rationale; use a new output folder")

    searchers = {}
    for corpus in corpora:
        rows, scores = ExactIndex(matrices[corpus.name]).search(vectors, k)
        table: dict[str, tuple[Any, Any]] = {}
        for question, found_rows, found_scores in zip(pending, rows, scores, strict=True):
            table.setdefault(texts[question.id], (found_rows, found_scores))
        searchers[corpus.name] = _lookup_searcher(corpus, table, k)
        if not args.quiet:
            streams.err.write(f"searched {corpus.name}: {len(corpus):,} passages\n")

    log = _ScoreLog(scorer)
    for position, question in enumerate(pending, start=1):
        final = balanced_retrieve(
            question=question.text,
            rationale=rationales[question.id],
            searchers=searchers,
            scorer=log,
            top_k=k,
            rerank_against=args.rerank_against,
        )
        evidence = [
            {
                "corpus": item["corpus"],
                "doc_key": item["doc_key"],
                "chunk_index": item["chunk_index"],
                "row": item["row"],
                "dense_score": item["dense_score"],
                "rerank_score": log.by_object[id(item)],
                "text": item["text"],
            }
            for item in final
        ]
        store.append(
            RetrievalRecord(
                id=question.id,
                question_sha256=question_hash(question.text),
                rationale_sha256=text_sha256(texts[question.id]),
                pool_size=log.pool_size,
                evidence=evidence,
            )
        )
        if not args.quiet:
            streams.err.write(f"[{position}/{len(pending)}] {question.id}: pool {log.pool_size}, kept {len(evidence)}\n")
    streams.out.write(f"{len(questions)} questions: {len(questions) - len(pending)} already done, {len(pending)} written\n")
    return _write_report(args.out, streams, None)


def _summary(folder: Path) -> tuple[dict, dict]:
    store = RetrievalStore(folder)
    if not (folder / RECORDS_FILE).exists():
        raise StoreError(f"no {RECORDS_FILE} in {folder}")
    records = list(store.load().values())
    config = store.read_config() if (folder / "config.json").exists() else {}
    summary = summarise_lists([record.evidence for record in records])
    pools = [record.pool_size for record in records]
    if pools:
        summary["mean_pool_size"] = round(sum(pools) / len(pools), 3)
    return summary, config


def _report_text(summary: dict, config: dict) -> str:
    lines = [f"questions with evidence: {summary['questions']}"]
    if "mean_corpus_share" in summary:
        lines.append(f"mean pool size {summary.get('mean_pool_size')}, mean list length {summary['mean_list_length']}")
        shares = ", ".join(f"{name} {summary['mean_corpus_share'][name]:.3f}" for name in CORPUS_ORDER)
        lines.append(f"mean share of the final list: {shares}")
        lines.append(
            f"mean duplicate rate across corpora: {summary['mean_duplicate_rate']:.3f}; "
            f"passages without a document key: {summary['share_without_document_key']:.3f}"
        )
    if summary.get("empty_lists"):
        lines.append(f"{summary['empty_lists']} questions have an empty list")
    if config:
        lines.append(
            f"settings: top_k {config.get('top_k')}, re-ranker reads the {config.get('rerank_against')}, "
            f"corpora {', '.join(config.get('corpora', {}))}, cross-encoder {config.get('cross_encoder', {}).get('cross_encoder')}"
        )
    return "\n".join(lines) + "\n"


def _write_report(folder: Path, streams: Streams, json_path: Path | None) -> int:
    summary, config = _summary(folder)
    streams.out.write(_report_text(summary, config))
    write_atomic(folder / SUMMARY_FILE, json.dumps({"summary": summary, "config": config}, indent=2, ensure_ascii=False) + "\n")
    if json_path is not None:
        write_atomic(json_path, json.dumps({"summary": summary, "config": config}, indent=2, ensure_ascii=False) + "\n")
    return 0


def _report(args: argparse.Namespace, streams: Streams) -> int:
    summary, config = _summary(args.folder)
    buffer = io.StringIO()
    buffer.write(_report_text(summary, config))
    streams.out.write(buffer.getvalue())
    if args.json is not None:
        write_atomic(args.json, json.dumps({"summary": summary, "config": config}, indent=2, ensure_ascii=False) + "\n")
    return 0

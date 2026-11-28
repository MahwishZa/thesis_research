"""
The one entry point of the project. Run it from the repository folder:  python -m src.main <command> [options]
Every command has --help. The commands, in the order of the research pipeline:

  The dataset
    preprocess          step 1: clean, de-duplicate and split the Kaggle download (laptop: the raw images)
    views               step 2: the three views and their question files
    features            the feature sufficiency and shortcut check of the cleaned data
    status              where the pipeline stands: done / not run / problem for every step (reads only; the default command)

  The knowledge library (for the RAG systems)
    library             every step below in order, from corpus to index, in one command (internet and laptop; --from STEP to resume)
    corpus              the text library: PubMed, PMC, clinical guidelines and Dermoscopedia (internet; laptop)
    audit               checks of both libraries: real sources and unchanged text (--online asks NCBI), near-duplicates, reference copies, tags
    visual              the figure list of the image library: skin-image figures and captions of open-access PMC papers
    visual-images       the images of those figures (internet; laptop; --probe N first)
    visual-calibrate    how well the look-alike tests separate same-lesion from different-lesion Kaggle photos
    references          the reference texts of the 8 diseases for the semantic-similarity score (evaluation material)

  The indexes
    index               the text index (MedCPT article encoder; laptop)
    visual-index        the image index, without any image that looks like a Kaggle image (laptop; MONET)

  The systems
    llm-only            the LLM-only system: the question alone
    standard-rag        the standard RAG system
    model-probe         times the steps of one question of a local model (load, reading the picture, each new token): where the time goes
    retrieval-eval      the retrieval-only comparison on the validation split (no model call, no cost)

  The evaluation
    evaluate            the report of one system's predictions on one question file
    compare             several runs on the same cases (McNemar's exact test, Holm's correction)
    collect             a report for every finished run and the table results/comparisons/summary.csv
    pilot-check         the gates of the validation pilot (readable answers, above chance, evidence line, quotes) for every complete validation run
    similarity-check    is the semantic similarity informative? (finished runs and the reference texts)
"""
import argparse
import filecmp
import importlib
import json
import os
import shutil
import sys
import tempfile

from src.data.views import TEST_PER_CLASS, VAL_PER_CLASS, build_views, images_not_in_table, moved_files, question_file, view_table
from src.pipelines.base import run_folder
from src.utils import (
    ALL_CASES,
    CLASSES,
    CORPUS_FOLDERS,
    IMAGE_LIBRARY,
    INDEX_DIR,
    MULTIMODAL_LIBRARY,
    PROCESSED_CASES,
    PROCESSED_IMAGES,
    RAW,
    REFERENCE_TEXTS,
    SPLITS,
    TEXT_LIBRARY,
    VIEWS,
    file_sha256,
    read_csv,
    text_matches,
)


def count_lines(path):
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def step_raw(data):
    audit = os.path.join(data, PROCESSED_CASES, "raw_files_audit.csv")
    folders = [os.path.join(data, RAW, d) for d in ("Train", "Test")]
    if not all(os.path.isdir(f) for f in folders):
        return NOT_RUN, "the Kaggle download is not in data/raw/ (Train/, Test/)"
    n = sum(f.lower().endswith(".jpg") for d in folders for _, _, fs in os.walk(d) for f in fs)
    expected = len(read_csv(audit)) if os.path.isfile(audit) else None
    if expected is not None and n != expected:
        return PROBLEM, f"{n} JPG files, but the audit lists {expected}"
    return DONE, f"{n} JPG files"


def step_clean(data):
    missing = [f for f in PROCESSED_FILES if not os.path.isfile(os.path.join(data, PROCESSED_CASES, f))]
    if len(missing) == len(PROCESSED_FILES):
        return NOT_RUN, "no table in data/processed/cases/"
    if missing:
        return PROBLEM, f"missing: {', '.join(missing)}"
    cases = read_csv(os.path.join(data, ALL_CASES))
    for split, per_class in (("test", TEST_PER_CLASS), ("val", VAL_PER_CLASS)):
        counts = {c: sum(r["split"] == split and r["label"] == c for r in cases) for c in CLASSES}
        if set(counts.values()) != {per_class}:
            return PROBLEM, (f"the {split} split is not balanced to {per_class} per class ("
                             + ", ".join(f"{c} {n}" for c, n in counts.items()) + "): rerun the preprocess command, then the views command")
    images = os.path.join(data, PROCESSED_IMAGES)
    if not os.path.isdir(images):
        return NOT_RUN, f"{len(cases)} cases; the cleaned images are not in data/{PROCESSED_IMAGES}/ (laptop only)"
    absent = [c["isic_id"] for c in cases if not os.path.isfile(os.path.join(data, c["image_path"]))]
    extra = images_not_in_table(data, [c["isic_id"] for c in cases])
    notes = []
    if absent:
        notes.append(f"{len(absent)} images missing, rerun the preprocess command (e.g. {', '.join(absent[:3])})")
    if extra:
        notes.append(f"{len(extra)} images not in the case table, left from an earlier run; delete them from "
                     f"data/{PROCESSED_IMAGES}/: {', '.join(extra[:10])}" + (" ..." if len(extra) > 10 else ""))
    if notes:
        return PROBLEM, "; ".join(notes)
    return DONE, f"{len(cases)} cases, {TEST_PER_CLASS} test and {VAL_PER_CLASS} validation images per class, every image present"


def step_views(data):
    names = [view_table(v) for v in VIEWS] + [question_file(v, s) for v in VIEWS for s in SPLITS]
    if not os.path.isfile(os.path.join(data, ALL_CASES)):
        return NOT_RUN, "needs step 1"
    missing = [n for n in names if not os.path.isfile(os.path.join(data, n))]
    if len(missing) == len(names):
        return NOT_RUN, "no view built"
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, PROCESSED_CASES))
        shutil.copy(os.path.join(data, ALL_CASES), os.path.join(d, ALL_CASES))
        build_views(d)
        stale = [n for n in names if n in missing or not filecmp.cmp(os.path.join(d, n), os.path.join(data, n), shallow=False)]
    if stale:
        return PROBLEM, f"{len(stale)} view files differ from a fresh build (e.g. {stale[0]})"
    return DONE, f"{len(names)} files, the same as a fresh build"


def step_feature_check(results):
    path = os.path.join(results, "data_quality", "feature_check_results.json")
    return (DONE, "results present") if os.path.isfile(path) else (NOT_RUN, "no results")


def step_corpus(data):
    folder = os.path.join(data, TEXT_LIBRARY)
    files = {f: os.path.join(folder, f) for f in ("passages.jsonl", "manifest.csv", "corpus_card.json")}
    present = [f for f, p in files.items() if os.path.isfile(p)]
    if not present:
        return NOT_RUN, "the corpus has not been built"
    if "manifest.csv" in present and "passages.jsonl" not in present:
        return NOT_RUN, f"manifest only ({len(read_csv(files['manifest.csv']))} passages); passages.jsonl is on the laptop"
    if len(present) < 3:
        return PROBLEM, f"only {', '.join(present)}"
    n_passages, n_manifest = count_lines(files["passages.jsonl"]), len(read_csv(files["manifest.csv"]))
    if n_passages != n_manifest:
        return PROBLEM, f"{n_passages} passages but {n_manifest} manifest rows: rebuild"
    with open(files["corpus_card.json"], encoding="utf-8") as f:
        empty = json.load(f).get("empty_topic_boxes", [])
    return DONE, f"{n_passages} passages" + (f"; topics with no passage: {', '.join(empty)}" if empty else "")


def step_references(data):
    path = os.path.join(data, REFERENCE_TEXTS)
    if not os.path.isfile(path):
        return NOT_RUN, "no reference texts"
    with open(path, encoding="utf-8") as f:
        refs = json.load(f)
    old = [c for c, r in refs.items() if "texts" not in r]
    if old:
        return PROBLEM, "the file has the old form (one text per disease): rerun the references command"
    missing = [c for c in CLASSES if c not in refs]
    empty = [c for c, r in refs.items() if not r["texts"]]
    if missing or empty:
        return PROBLEM, "; ".join(([f"missing {missing}"] if missing else []) + ([f"no text: {empty}"] if empty else []))
    counts = ", ".join(f"{c} {len(refs[c]['texts'])}" for c in CLASSES)
    return DONE, f"{sum(len(r['texts']) for r in refs.values())} reference texts ({counts})"


def step_index(data):
    folder = os.path.join(data, INDEX_DIR)
    meta_path = os.path.join(folder, "index.json")
    if not os.path.isfile(meta_path):
        return NOT_RUN, "no index: run the index command on the laptop (needs passages.jsonl)"
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    if not os.path.isfile(os.path.join(folder, "embeddings.f32")):
        return PROBLEM, "index.json without embeddings.f32: build the index again"
    if os.path.getsize(os.path.join(folder, "embeddings.f32")) != 4 * meta["n_passages"] * meta["dim"]:
        return PROBLEM, "embeddings.f32 has the wrong size: build the index again"
    passages = os.path.join(data, TEXT_LIBRARY, "passages.jsonl")
    if os.path.isfile(passages) and file_sha256(passages) != meta.get("passages_sha256"):
        return PROBLEM, "passages.jsonl changed after the index was built: run the index command again"
    if meta.get("captions_file"):
        captions = os.path.normpath(os.path.join(data, TEXT_LIBRARY, meta["captions_file"]))
        if not os.path.isfile(captions) or file_sha256(captions) != meta.get("captions_sha256"):
            return PROBLEM, "the image library changed after the index was built: run the index command again"
    note = f"{meta['n_passages']} passages, {meta['dim']} numbers each ({meta['encoder']})"
    if meta.get("captions_file"):
        note += f"; with {meta.get('n_captions_indexed', 0)} caption records"
    note += "; reference pages left out" if meta.get("exclude_reference_pages") else "; all passages"
    return DONE, note + ("; reference copies left out" if meta.get("exclude_reference_overlap") else "")


def step_libraries(data):
    """The library of each view (data/corpus/<view>/): text_only and image_only hold the records, multimodal names the two it combines
    and how their searches are merged (it must say what the standard RAG does)."""
    from src.pipelines.standard_rag import FUSION_LIST
    path = os.path.join(data, MULTIMODAL_LIBRARY, "library.json")
    if not os.path.isfile(path):
        return NOT_RUN, f"no {MULTIMODAL_LIBRARY}/library.json"
    with open(path, encoding="utf-8") as f:
        library = json.load(f)
    if sorted(library.get("sources", [])) != ["image_only", "text_only"]:
        return PROBLEM, f"the multimodal library must combine text_only and image_only, not {library.get('sources')}"
    if library.get("fusion") != {"method": "reciprocal rank fusion", "k": 60, "list": FUSION_LIST}:
        return PROBLEM, f"library.json says {library.get('fusion')}, the standard RAG fuses with k = 60 and the top {FUSION_LIST}"
    missing = [v for v in ("text_only", "image_only") if not os.path.isdir(os.path.join(data, CORPUS_FOLDERS[v]))]
    if missing:
        return PROBLEM, "no library folder for " + ", ".join(missing)
    card_path = os.path.join(data, IMAGE_LIBRARY, "visual_card.json")
    images = ""
    if os.path.isfile(card_path):
        with open(card_path, encoding="utf-8") as f:
            images = f"; image_only: {json.load(f).get('n_indexed_images', 0)} indexed images"
    records = os.path.join(data, IMAGE_LIBRARY, "items_final.jsonl")                  # laptop only: the records the image index was made from
    meta_path = os.path.join(data, IMAGE_LIBRARY, "index", "index.json")
    if os.path.isfile(records):
        if not os.path.isfile(meta_path):
            return PROBLEM, "no image index in image_only/index/: run the visual-index command"
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("passages_sha256") != file_sha256(records):
            return PROBLEM, "the image records changed after the image index was built: run the visual-index command again"
        images += f" ({meta['n_passages']} records in the index, {meta['encoder']})"
    return DONE, f"text_only and image_only hold the records, multimodal combines them (rank fusion, k = 60, top {FUSION_LIST}){images}"


SYSTEMS = ("llm_only", "standard_rag", "baseline_rag", "proposed_rag")     # the four approaches, each with three view folders


def runs_on_disk(results, system, view):
    """[(model, split, finished)] of the runs in results/<system>/<view>/ (a folder <model>_<split>, or <model>_<split>.partial)."""
    folder = os.path.join(results, system, view)
    if not os.path.isdir(folder):
        return []
    runs = []
    for name in sorted(os.listdir(folder)):
        base = name[:-len(".partial")] if name.endswith(".partial") else name
        model, _, split = base.rpartition("_")
        if os.path.isdir(os.path.join(folder, name)) and model and split in SPLITS:
            runs.append((model, split, name == base))
    return runs


def step_runs(data, results, systems=SYSTEMS):
    """One line per run found on disk (finished or not). A system and view without any run get one 'not run' line: which models run
    in which system is not decided in advance (the researcher has no budget, and the RAG backbone is still to be chosen), so no
    line asks for a run of a particular model. The trial folders of the standard RAG (standard_rag_<mode>) are listed too."""
    trials = sorted(d for d in os.listdir(results) if d.startswith("standard_rag_") and os.path.isdir(os.path.join(results, d))) if os.path.isdir(results) else []
    rows = []
    for system in [*systems, *trials]:
        for view in VIEWS:
            found = runs_on_disk(results, system, view)
            if not found:
                rows.append((f"{system} {view}", NOT_RUN, "no run yet"))
            for model, split, finished in found:
                name = f"{system} {model} {view} {split}"
                final, _ = run_folder(results, model, view, split, system)
                if not finished:
                    with open(os.path.join(final + ".partial", "run.json"), encoding="utf-8") as f:
                        info = json.load(f)
                    progress = f"{info['answered']}/{info['of']} ({info['status']}); run the same command again to continue"
                    # a validation run (git-ignored, a trial) takes hours and is normally under way: it is a step still to finish, not a problem
                    rows.append((name, NOT_RUN, f"validation run unfinished: {progress}") if split == "val" else (name, PROBLEM, f"unfinished: {progress}"))
                else:
                    rows.append((name, *check_run(data, final, view, split)))
    return rows


def check_run(data, folder, view, split):
    with open(os.path.join(folder, "run.json"), encoding="utf-8") as f:
        info = json.load(f)
    qfile = os.path.join(data, question_file(view, split))
    recorded = info["settings"].get("questions_sha256")
    if not os.path.isfile(qfile):
        return PROBLEM, f"the question file {question_file(view, split)} is not there, so the run cannot be matched to its questions"
    if not text_matches(qfile, recorded):
        moved = moved_files(data).get(question_file(view, split).replace("\\", "/"), {})
        if not (moved.get("previous_sha256") == recorded and text_matches(qfile, moved.get("sha256"))):
            return PROBLEM, "the question file changed after this run: the run does not match the current questions"
    if split == "val":       # a validation run is judged by pilot-check and never reported (researcher's rule): no report.json is expected
        return DONE, "complete (validation run: judged by pilot-check)"
    report = os.path.join(folder, "report.json")
    if not os.path.isfile(report):
        return PROBLEM, "complete, but not evaluated yet (no report.json)"
    return DONE, "complete and evaluated"


def step_retrieval_checks(data, results):
    """The retrieval-eval result of each view (results/standard_rag/<view>/retrieval_validation.json) must belong to the libraries
    that exist now: a result made with an older text or image library says nothing about the current one."""
    from src.pipelines.base import APPROVED_RETRIEVAL
    current = {"text_index_passages_sha256": None, "image_items_sha256": None}
    for key, path in (("text_index_passages_sha256", os.path.join(data, INDEX_DIR, "index.json")),
                      ("image_items_sha256", os.path.join(data, IMAGE_LIBRARY, "index", "index.json"))):
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                current[key] = json.load(f).get("passages_sha256")
    notes, missing, old = [], [], []
    for view in VIEWS:
        path = os.path.join(results, "standard_rag", view, "retrieval_validation.json")
        if not os.path.isfile(path):
            missing.append(view)
            continue
        with open(path, encoding="utf-8") as f:
            saved = json.load(f)
        if "libraries" not in saved:
            old.append(view)
            continue
        if saved.get("approved") != APPROVED_RETRIEVAL:
            return PROBLEM, f"{view}: made when {saved.get('approved')} was the approved search, now it is {APPROVED_RETRIEVAL}: run retrieval-eval again"
        for key, value in saved["libraries"].items():
            if value is not None and current[key] is not None and value != current[key]:
                return PROBLEM, f"{view}: made with another {'text' if key.startswith('text') else 'image'} library: run retrieval-eval --view {view} again"
        notes.append(view)
    if len(missing) == len(VIEWS):
        return NOT_RUN, "no result yet (retrieval-eval, validation split, no cost)"
    if missing or old:
        return NOT_RUN, "; ".join(([f"no result for {', '.join(missing)}"] if missing else [])
                                  + ([f"{', '.join(old)}: made before the libraries were recorded, run retrieval-eval again"] if old else []))
    return DONE, f"{', '.join(notes)}: made with the current libraries"


def check(data="data", results="results"):
    steps = [("1 raw Kaggle download", step_raw(data)), ("1 cleaning (preprocess)", step_clean(data)),
             ("2 views (views)", step_views(data)), ("- feature check (features)", step_feature_check(results)),
             ("3 knowledge corpus (corpus)", step_corpus(data)),
             ("3b reference texts (references)", step_references(data)),
             ("4 retrieval index (index)", step_index(data)),
             ("4b libraries per view (text_only, image_only, multimodal)", step_libraries(data)),
             ("5 retrieval checks (retrieval-eval)", step_retrieval_checks(data, results))]
    return [(name, *r) for name, r in steps] + step_runs(data, results)


def status_main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.main status", description="Where the research pipeline stands (reads only).")
    ap.add_argument("--data", default="data")
    ap.add_argument("--results", default="results")
    a = ap.parse_args(argv)
    rows = check(a.data, a.results)
    width = max(len(r[0]) for r in rows)
    mark = {DONE: "[x]", NOT_RUN: "[ ]", PROBLEM: "[!]"}
    for name, status, note in rows:
        print(f"{mark[status]} {name.ljust(width)}  {status:8s} {note}")
    problems = sum(r[1] == PROBLEM for r in rows)
    print(f"\n{sum(r[1] == DONE for r in rows)} done, {sum(r[1] == NOT_RUN for r in rows)} not run, {problems} with a problem")
    return 1 if problems else 0


# ---- the whole library in one command -----------------------------------------------------------------------------------------------
LIBRARY_STEPS = ("corpus", "audit", "visual", "visual-images", "visual-index", "index")   # the order the libraries and their indexes are built in


def library_steps(extra_papers=0, dermoscopedia=0, online=False):
    """[(command, its options)] of the whole library build. The text index is built with the approved settings (reference pages and
    reference copies left out); visual-calibrate is a report about the look-alike filter, not a step of the build, so it is not here."""
    visual = (["--extra-papers", str(extra_papers)] if extra_papers else []) + (["--dermoscopedia", str(dermoscopedia)] if dermoscopedia else [])
    options = {"corpus": [], "audit": ["--online"] if online else [], "visual": visual, "visual-images": [], "visual-index": [],
               "index": ["--exclude-reference-pages", "--exclude-reference-overlap"]}
    return [(step, options[step]) for step in LIBRARY_STEPS]


def library_main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.main library", description="Builds both libraries and their indexes, step by step, "
                                 "stopping at the first step that fails. Default folders only (data/corpus/...).")
    ap.add_argument("--extra-papers", type=int, default=0, help="visual: up to N more PMC papers per diagnosis")
    ap.add_argument("--dermoscopedia", type=int, default=0, help="visual: up to N Dermoscopedia pages per diagnosis")
    ap.add_argument("--online", action="store_true", help="audit: also ask NCBI and Dermoscopedia whether every document exists")
    ap.add_argument("--from", dest="first", default=LIBRARY_STEPS[0], choices=LIBRARY_STEPS, help="start at this step (to resume after a failure)")
    a = ap.parse_args(argv)
    steps = library_steps(a.extra_papers, a.dermoscopedia, a.online)
    for step, options in steps[LIBRARY_STEPS.index(a.first):]:
        print(f"== {step} {' '.join(options)}".rstrip(), flush=True)
        try:
            result = command(step)(options)
        except (Exception, SystemExit) as e:                           # a failed step must say which one, and how to continue from it
            print(f"stopped at {step}: {e}\nfix the cause and run: python -m src.main library --from {step} <the same options>", file=sys.stderr)
            return 1
        if isinstance(result, int) and result:
            print(f"stopped at {step} (exit code {result})", file=sys.stderr)
            return result
    print("done; now run: python -m src.main status")
    return 0


# ---- the entry point ------------------------------------------------------------------------------------------------
# command -> (module, function); a module is imported only when its command runs, so a command never needs another one's packages
COMMANDS = {
    "preprocess": ("src.data.preprocessing", "preprocess_main"),
    "views": ("src.data.views", "views_main"),
    "features": ("src.data.preprocessing", "features_main"),
    "status": ("src.main", "status_main"),
    "library": ("src.main", "library_main"),
    "corpus": ("src.retrieval.corpus", "corpus_main"),
    "audit": ("src.retrieval.corpus", "audit_main"),
    "visual": ("src.retrieval.images", "visual_main"),
    "visual-images": ("src.retrieval.images", "visual_images_main"),
    "visual-calibrate": ("src.retrieval.images", "visual_calibrate_main"),
    "references": ("src.evaluation.references", "references_main"),
    "index": ("src.retrieval.indexer", "index_main"),
    "visual-index": ("src.retrieval.indexer", "visual_index_main"),
    "llm-only": ("src.pipelines.llm_only", "llm_only_main"),
    "model-probe": ("src.pipelines.llm_only", "probe_main"),
    "standard-rag": ("src.pipelines.standard_rag", "standard_rag_main"),
    "retrieval-eval": ("src.pipelines.standard_rag", "retrieval_eval_main"),
    "evaluate": ("src.evaluation.evaluator", "report_main"),
    "compare": ("src.evaluation.evaluator", "compare_main"),
    "collect": ("src.evaluation.evaluator", "collect_main"),
    "pilot-check": ("src.evaluation.evaluator", "pilot_main"),
    "similarity-check": ("src.evaluation.evaluator", "similarity_check_main"),
}


def command(name):
    """The function that runs a command."""
    module, function = COMMANDS[name]
    return getattr(importlib.import_module(module), function)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv:
        argv = ["status"]
    if argv[0] not in COMMANDS:
        print(__doc__.strip())
        return 0 if argv in (["-h"], ["--help"]) else 2
    result = command(argv[0])(argv[1:])
    return result if isinstance(result, int) else 0      # a command's result (e.g. a folder or a list) is not an exit code


if __name__ == "__main__":
    sys.exit(main())

DONE, NOT_RUN, PROBLEM = "done", "not run", "problem"
PROCESSED_FILES = ("all_cases.csv", "raw_files_audit.csv", "near_duplicates.csv", "classes.json", "normalization.json",
                   "class_weights.json", "train_oversampled.csv", "train_undersampled.csv")

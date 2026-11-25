"""
The one entry point of the project. Run it from the repository folder:  python -m src.main <command> [options]
Every command has --help. The commands, in the order of the research pipeline:

  The dataset
    preprocess          step 1: clean, de-duplicate and split the Kaggle download (laptop: the raw images)
    views               step 2: the three views and their question files
    features            the feature sufficiency and shortcut check of the cleaned data
    status              where the pipeline stands: done / not run / problem for every step (reads only; the default command)

  The knowledge library (for the RAG systems)
    corpus              the text library: PubMed, PMC, clinical guidelines and Dermoscopedia (internet; laptop)
    audit               checks of the corpus text: near-duplicates, copies of the reference texts, tags
    visual              the figure list of the image library: skin-image figures and captions of open-access PMC papers
    visual-images       the images of those figures (internet; laptop; --probe N first)
    visual-calibrate    how well the look-alike tests separate same-lesion from different-lesion Kaggle photos
    references          the reference texts of the 8 diseases for the semantic-similarity score (evaluation material)

  The indexes
    index               the text index (MedCPT article encoder; laptop)
    visual-index        the image index, without any image that looks like a Kaggle image (laptop; BiomedCLIP)

  The systems
    llm-only            the LLM-only system: the question alone
    standard-rag        the standard RAG system
    retrieval-eval      the retrieval-only comparison on the validation split (no model call, no cost)

  The evaluation
    evaluate            the report of one system's predictions on one question file
    compare             several runs on the same cases (McNemar's exact test, Holm's correction)
    collect             a report for every finished run and the table results/comparisons/summary.csv
    similarity-check    is the semantic similarity informative? (finished runs and the reference texts)
"""
import importlib
import sys

# command -> (module, function); a module is imported only when its command runs, so a command never needs another one's packages
COMMANDS = {
    "preprocess": ("src.data.preprocessing", "preprocess_main"),
    "views": ("src.data.views", "views_main"),
    "features": ("src.data.quality", "features_main"),
    "status": ("src.utils.status", "status_main"),
    "corpus": ("src.knowledge.corpus", "corpus_main"),
    "audit": ("src.knowledge.corpus", "audit_main"),
    "visual": ("src.knowledge.images", "visual_main"),
    "visual-images": ("src.knowledge.images", "visual_images_main"),
    "visual-calibrate": ("src.knowledge.lookalike", "visual_calibrate_main"),
    "references": ("src.knowledge.references", "references_main"),
    "index": ("src.retrieval.indexer", "index_main"),
    "visual-index": ("src.retrieval.indexer", "visual_index_main"),
    "llm-only": ("src.pipelines.llm_only", "llm_only_main"),
    "standard-rag": ("src.pipelines.standard_rag", "standard_rag_main"),
    "retrieval-eval": ("src.pipelines.standard_rag", "retrieval_eval_main"),
    "evaluate": ("src.evaluation.evaluator", "report_main"),
    "compare": ("src.evaluation.evaluator", "compare_main"),
    "collect": ("src.evaluation.evaluator", "collect_main"),
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

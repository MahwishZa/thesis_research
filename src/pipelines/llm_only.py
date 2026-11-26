"""The LLM-only system (python -m src.main llm-only): the question alone, no retrieval."""
import argparse
import os

from src.models.llm import MODELS, make_client, require_free_or_allowed
from src.pipelines.base import INSTRUCTIONS, run, run_folder
from src.utils import SPLITS, VIEWS, get_logger, load_env


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.main llm-only", description="LLM-only baseline: one model, one view, one split.")
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--view", required=True, choices=VIEWS)
    ap.add_argument("--split", default="val", choices=SPLITS, help="val to try things out; test only for the final runs")
    ap.add_argument("--style", default="rationale", choices=sorted(INSTRUCTIONS))
    ap.add_argument("--limit", type=int, default=0, help="ask at most this many questions now (a trial)")
    ap.add_argument("--sleep", type=float, default=0.0, help="seconds to wait between questions")
    ap.add_argument("--allow-paid", action="store_true", help="allow a model that costs credit (the free model needs no flag)")
    ap.add_argument("--dry-run", action="store_true", help="show the prompts of the first questions; no API call, no key needed")
    ap.add_argument("--env-file", default=".env", help="the file with the API keys")
    ap.add_argument("--data", default="data")
    ap.add_argument("--results", default="results")
    return ap.parse_args(argv)


def llm_only_main(argv=None):
    args = parse_args(argv)
    final, _ = run_folder(args.results, args.model, args.view, args.split)
    name = f"llm_only.{args.model}.{args.view}.{args.split}"
    log, client = get_logger(name), None
    if not args.dry_run and not os.path.isdir(final):
        unusable = load_env(args.env_file)
        if unusable:
            log.warning(f"{args.env_file}: lines {unusable} are not NAME=value (or have no value) and were skipped")
        require_free_or_allowed(args.model, args.allow_paid)
        client = make_client(args.model, log)                       # stops here when the key is missing
    return run(args, client, log)

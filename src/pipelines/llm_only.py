"""The LLM-only system (python -m src.main llm-only): the question alone, no retrieval."""
import argparse
import os

from src.models.llm import make_client, require_free_or_allowed
from src.pipelines.base import INSTRUCTIONS, add_run_arguments, run, run_folder
from src.utils import get_logger, load_env


def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.main llm-only", description="LLM-only baseline: one model, one view, one split.")
    add_run_arguments(ap)
    ap.add_argument("--style", default="rationale", choices=sorted(INSTRUCTIONS))
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

"""The LLM-only system (python -m src.main llm-only): the question alone, no retrieval."""
import argparse
import os
import time
from collections import Counter

from src.data.views import question_file
from src.models.llm import image_b64, make_client, package_version, require_free_or_allowed
from src.pipelines.base import INSTRUCTIONS, SYSTEM, add_run_arguments, asking_order, run, run_folder, user_text
from src.utils import get_logger, load_env, read_jsonl


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


# ---- model-probe: where the time of a local model goes (python -m src.main model-probe) -------------------------------------------------
# Loads the model, takes the first validation question of a view (the same prompt as a run) and times the steps one by one, so that a slow model can
# be understood before a long run: the load, the input preparation, the first new token (the model reads the picture and the prompt: "prefill"), and
# the time of each further token. Also says on which devices the layers sit. It writes nothing and asks no paid service.
def synchronize():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def timed(step):
    start = time.time()
    result = step()
    synchronize()
    return result, time.time() - start


def device_summary(model):
    """How many named blocks of the model sit on each device (None when the model was not split)."""
    placed = getattr(model, "hf_device_map", None)
    return dict(Counter(str(device) for device in placed.values())) if placed else None


def probe_main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.main model-probe", description="Times the steps of one local-model question (no run, nothing written).")
    ap.add_argument("--model", required=True)
    ap.add_argument("--view", default="image_only", choices=("text_only", "image_only", "multimodal"))
    ap.add_argument("--data", default="data")
    ap.add_argument("--env-file", default=".env")
    ap.add_argument("--tokens", type=int, default=17, help="new tokens of the second timing (the per-token time is read from the difference)")
    a = ap.parse_args(argv)
    require_free_or_allowed(a.model, False)                      # the probe is for local, free models only
    load_env(a.env_file)
    questions = asking_order(read_jsonl(os.path.join(a.data, question_file(a.view, "val"))))
    q = questions[0]
    image = image_b64(a.data, q["image_path"]) if a.view != "text_only" else None
    client, load_s = timed(lambda: make_client(a.model))
    inputs, prepare_s = timed(lambda: client.build_inputs(SYSTEM, user_text(q, "rationale", a.view), image))
    n_in = int(inputs["input_ids"].shape[-1])
    first, first_s = timed(lambda: client.generate(inputs, 1))
    more, more_s = timed(lambda: client.generate(inputs, a.tokens))
    per_token = (more_s - first_s) / max(1, len(more) - len(first))
    report = {"model": a.model, "question": q["id"], "view": a.view, "prompt_tokens": n_in, "devices": device_summary(client.model),
              "torch": package_version("torch"),
              "load_seconds": round(load_s, 1), "prepare_seconds": round(prepare_s, 2), "first_token_seconds": round(first_s, 2),
              f"{len(more)}_tokens_seconds": round(more_s, 2), "seconds_per_further_token": round(per_token, 3)}
    try:
        import torch
        report["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            report["gpu"] = torch.cuda.get_device_name(0)
            report["gpu_memory_gb_now"] = round(torch.cuda.memory_allocated() / 1e9, 2)
            report["gpu_memory_gb_peak"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    except ImportError:
        pass
    try:
        import psutil
        report["computer_memory_used_gb"] = round(psutil.virtual_memory().used / 1e9, 1)
        report["computer_memory_total_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        pass
    for key, value in report.items():
        print(f"{key}: {value}")
    print("reading: first_token_seconds is the model reading the picture and the prompt; seconds_per_further_token is the speed of the answer; "
          "an answer of 100 tokens takes first_token_seconds + 100 x seconds_per_further_token.")
    return report

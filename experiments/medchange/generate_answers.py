"""Generate one answer per (item, arm) with a local GGUF model (llama.cpp, CPU).

    python -m experiments.medchange.generate_answers --split dev --arms B0 B1 \\
        --model-path models\\Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --limit 3

Greedy decoding (temperature 0), identical prompt template and budget for every arm,
frozen pools only. Resumable: one JSON line per (item, arm); finished pairs are skipped
and a truncated last line (killed mid-write) is ignored. Items are interleaved across
arms so an interrupted run leaves a balanced partial set. Records carry the arm-settings
hash, prompt hash and wall-clock seconds (the G1 speed benchmark).

The generator itself (model file hash, context size, token limit, seed, ...) is recorded once
per answers file in ``<answers>.config.json`` (``generator.py``). A resumed run whose result-relevant
settings differ from the recorded ones is refused, so one answers file never mixes configurations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Callable

from . import arms as A
from .files import file_sha256, load_jsonl
from .generator import check_config, config_path, llama_generator, llama_version, run_config
from .prompts import SYSTEM, build_prompt, parse_verdict

HERE = Path(__file__).resolve().parent
MAX_NEW_TOKENS = 160


def run(items: list[dict], pools: dict[str, dict], arm_list: list[str],
        out: Path, generate: Callable[[str, str], str], *, clock=time.time,
        on_progress: Callable[[int, int], None] = lambda a, b: None) -> int:
    done = {(r["item_id"], r["arm"]) for r in load_jsonl(out)}
    todo = [(it, arm) for it in items for arm in arm_list if (it["item_id"], arm) not in done]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8", newline="\n") as h:
        for k, (it, arm) in enumerate(todo, 1):
            admitted = A.admit(arm, pools[it["item_id"]]["candidates"])
            prompt = build_prompt(it["question"], admitted)
            t0 = clock()
            text = generate(SYSTEM, prompt)
            h.write(json.dumps({
                "item_id": it["item_id"], "arm": arm, "settings": A.settings_hash(),
                "admitted": [c["pmid"] for c in admitted],
                "admitted_upper": [c["upper"] for c in admitted],
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()[:16],
                "text": text, "verdict": parse_verdict(text),
                "seconds": round(clock() - t0, 1)}) + "\n")
            h.flush()
            on_progress(k, len(todo))
    return len(todo)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--split", default="dev", choices=("dev", "confirm", "ad"))
    ap.add_argument("--arms", nargs="+", default=["B0", "B1"], choices=list(A.ARMS),
                    help="B0: no retrieval; B1: standard retrieval (the top 5 of the frozen pool)")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None, help="first N items only")
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--n-threads", type=int, default=None)
    ap.add_argument("--n-gpu-layers", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    args = ap.parse_args(argv)

    d = HERE / "data"
    items = [r for r in load_jsonl(d / "benchmark.jsonl")
             if r["split"] == args.split and not r["likely_label_noise"]][:args.limit]
    pools = {r["item_id"]: r for r in load_jsonl(d / f"frozen_{args.split}.jsonl")}
    missing = [i["item_id"] for i in items if i["item_id"] not in pools]
    if missing:
        print(f"no frozen pool for {len(missing)} items (run freeze_candidates): {missing[:3]}",
              file=sys.stderr)
        return 2
    if not Path(args.model_path).is_file():
        print(f"model file not found: {args.model_path}", file=sys.stderr)
        return 2
    out = Path(args.out or d / f"answers_{args.split}.jsonl")
    if out.exists() and not config_path(out).exists():
        print(f"note: {out.name} has no recorded configuration (made before it was recorded); "
              "recording the current one.", file=sys.stderr)
    print("hashing the model file ...", flush=True)
    differs = check_config(out, run_config(
        model_path=args.model_path, model_sha256=file_sha256(args.model_path), n_ctx=args.n_ctx,
        max_new_tokens=args.max_new_tokens, n_threads=args.n_threads,
        n_gpu_layers=args.n_gpu_layers, llama_version=llama_version()))
    if differs:
        print(f"refusing to extend {out.name}: its recorded generator configuration differs in "
              f"{', '.join(differs)} (see {config_path(out).name}). Use a different --out, or restore "
              "the original model and settings.", file=sys.stderr)
        return 2
    gen = llama_generator(args.model_path, args.n_ctx, args.n_threads, args.n_gpu_layers,
                          args.max_new_tokens)
    t0 = time.time()
    n = run(items, pools, args.arms, out, gen,
            on_progress=lambda k, t: print(f"  {k}/{t} ({time.time() - t0:.0f}s)", flush=True)
            if k % 5 == 0 or k == t else None)
    print(f"generated {n} answers -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

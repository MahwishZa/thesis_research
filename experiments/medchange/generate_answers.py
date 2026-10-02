"""Generate one answer per (item, arm) with a local GGUF model (llama.cpp, CPU).

    python -m experiments.medchange.generate_answers --split dev --arms B0 B1 \\
        --model-path models\\Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --limit 3

Greedy decoding (temperature 0), identical prompt template and budget for every arm,
frozen pools only. Resumable: one JSON line per (item, arm); finished pairs are skipped
and a truncated last line (killed mid-write) is ignored. Items are interleaved across
arms so an interrupted run leaves a balanced partial set. Records carry the arm-settings
hash, prompt hash and wall-clock seconds (the G1 speed benchmark).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Callable, Optional

from . import arms as A
from .prompts import SYSTEM, build_prompt, parse_verdict

HERE = Path(__file__).resolve().parent
MAX_NEW_TOKENS = 160


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if i == len(lines) - 1:
                    break
                raise
    return rows


def merge_helpfulness(pool: list[dict], rec: Optional[dict]) -> list[dict]:
    """Attach cached helpfulness scores, refusing a pool/score mismatch."""
    if rec is None:
        return pool
    if rec["pmids"] != [c["pmid"] for c in pool]:
        raise ValueError("helpfulness scores were computed for a different pool")
    return [dict(c, helpful=s) for c, s in zip(pool, rec["helpful"])]


def run(items: list[dict], pools: dict[str, dict], helpful: dict[str, dict], arm_list: list[str],
        out: Path, generate: Callable[[str, str], str], *, clock=time.time,
        on_progress: Callable[[int, int], None] = lambda a, b: None) -> int:
    done = {(r["item_id"], r["arm"]) for r in load_jsonl(out)}
    todo = [(it, arm) for it in items for arm in arm_list if (it["item_id"], arm) not in done]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8", newline="\n") as h:
        for k, (it, arm) in enumerate(todo, 1):
            pool = merge_helpfulness(pools[it["item_id"]]["candidates"], helpful.get(it["item_id"]))
            admitted = A.admit(arm, pool, it["newest"]["date"], item_id=it["item_id"])
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


def llama_generator(model_path: str, n_ctx: int, n_threads: Optional[int], n_gpu_layers: int,
                    max_new_tokens: int) -> Callable[[str, str], str]:
    from llama_cpp import Llama
    llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads,
                n_gpu_layers=n_gpu_layers, verbose=False, seed=42)

    def generate(system: str, user: str) -> str:
        r = llm.create_chat_completion(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_new_tokens, temperature=0.0)
        return r["choices"][0]["message"]["content"]
    return generate


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--split", default="dev", choices=("dev", "confirm"))
    ap.add_argument("--arms", nargs="+", default=list(A.ARMS), choices=list(A.ARMS))
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
    helpful = {r["item_id"]: r for r in load_jsonl(d / f"helpfulness_{args.split}.jsonl")}
    needs = any(a in ("B2", "P", "C1") for a in args.arms)
    if needs and any(i["item_id"] not in helpful for i in items):
        print("B2/P/C1 need helpfulness scores for every item: run "
              "experiments.medchange.helpfulness first", file=sys.stderr)
        return 2
    out = Path(args.out or d / f"answers_{args.split}.jsonl")
    gen = llama_generator(args.model_path, args.n_ctx, args.n_threads, args.n_gpu_layers,
                          args.max_new_tokens)
    t0 = time.time()
    n = run(items, pools, helpful, args.arms, out, gen,
            on_progress=lambda k, t: print(f"  {k}/{t} ({time.time() - t0:.0f}s)", flush=True)
            if k % 5 == 0 or k == t else None)
    print(f"generated {n} answers -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

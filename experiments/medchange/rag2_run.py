"""Run the realigned systems on one split (local machine; llama.cpp and MedCPT on CPU; no network).

    python -m experiments.medchange.rag2_run rationale --split dev --model-path models\\Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
    python -m experiments.medchange.rag2_run lists --split dev
    python -m experiments.medchange.rag2_run filter --split dev --model-path models\\Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
    python -m experiments.medchange.rag2_run answers --split dev --arms R2 R2C R2V R2V-ND --model-path models\\Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
    python -m experiments.medchange.rag2_run judge --split dev --model-path models\\Qwen2.5-7B-Instruct-Q4_K_M.gguf

Steps and outputs (``data/``, gitignored; ``rag2_pipeline`` copies the shareable ones to ``results/``):

* ``rationale``  the generator's closed-book rationale per question -> ``rag2_rationales_<split>.jsonl``
* ``lists``      R2's candidate lists and the two retrieval ablations, from the cached as-of PubMed records
                 (``pubmed_g0/``, ``abstracts.jsonl``) -> ``rag2_lists_<split>.jsonl``
* ``filter``     the zero-shot LLM filter's P(yes) per listed abstract -> ``rag2_filter_<split>.jsonl``
* ``answers``    R2 and its variants, R2C, R2V, R2V-ND -> ``rag2_answers_<split>.jsonl``
* ``judge``      optional: an independent model's directness judgement of the abstracts B1 and R2 admitted
                 -> ``rag2_directness_<split>.jsonl``

Every step appends one JSON line per finished unit, skips finished ones on a rerun, ignores a truncated last
line, and refuses to extend a file written under a different configuration (model file, prompts, settings).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

from . import rag2 as R
from .generate_answers import (SEED, check_config, config_path, file_sha256, llama_generator, llama_version,
                               load_jsonl)
from .prompts import build_prompt, parse_verdict
from .abstracts import study_snippet

HERE = Path(__file__).resolve().parent
RATIONALE_N_CTX = 1024
ANSWER_RELEVANT = ("model_sha256", "n_ctx", "max_new_tokens", "temperature", "seed", "settings_sha256", "prompts")


def out_path(data: Path, step: str, split: str) -> Path:
    return data / f"rag2_{step}_{split}.jsonl"


def load_items(data: Path, split: str, limit: Optional[int] = None) -> list[dict]:
    items = [r for r in load_jsonl(data / "benchmark.jsonl")
             if r["split"] == split and not r["likely_label_noise"]]
    return items[:limit] if limit else items


def _printer(every: int = 5) -> Callable[[int, int], None]:
    started = time.time()
    return lambda k, t: print(f"  {k}/{t} ({time.time() - started:.0f}s)", flush=True) \
        if k % every == 0 or k == t else None


def _refuse(out: Path, differs: Sequence[str]) -> int:
    print(f"refusing to extend {out.name}: its recorded configuration differs in {', '.join(differs)} "
          f"(see {config_path(out).name}). Use a different --out, or restore the original model and settings.",
          file=sys.stderr)
    return 2


# --------------------------------------------------------------------------------------
# Step 1: rationales
# --------------------------------------------------------------------------------------

def run_rationales(items: Sequence[dict], generate: Callable[[str, str], str], out: Path, *, clock=time.time,
                   on_progress: Callable[[int, int], None] = lambda a, b: None) -> int:
    done = {r["item_id"] for r in load_jsonl(out)}
    todo = [it for it in items if it["item_id"] not in done]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8", newline="\n") as h:
        for k, it in enumerate(todo, 1):
            system, user = R.build_rationale_prompt(it["question"])
            started = clock()
            text = generate(system, user)
            h.write(json.dumps({"item_id": it["item_id"], "rationale": (text or "").strip(),
                                "prompt_sha256": R.sha256(user)[:16],
                                "seconds": round(clock() - started, 1)}) + "\n")
            h.flush()
            on_progress(k, len(todo))
    return len(todo)


def rationale_config(model_path: str, model_sha256: str) -> dict:
    return {"model_file": Path(model_path).name, "model_sha256": model_sha256, "n_ctx": RATIONALE_N_CTX,
            "max_new_tokens": R.RATIONALE_MAX_NEW_TOKENS, "temperature": 0.0, "seed": SEED,
            "system_sha256": R.sha256(R.RATIONALE_SYSTEM), "template_sha256": R.sha256(R.RATIONALE_TEMPLATE)}


# --------------------------------------------------------------------------------------
# Step 2: candidate lists
# --------------------------------------------------------------------------------------

def run_lists(items: Sequence[dict], probes: dict, abstracts: dict, rationales: dict, out: Path, *,
              query_encoder, article_encoder, reranker,
              on_progress: Callable[[int, int], None] = lambda a, b: None) -> int:
    done = {r["item_id"] for r in load_jsonl(out)}
    todo = [it for it in items if it["item_id"] not in done]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8", newline="\n") as h:
        for k, it in enumerate(todo, 1):
            rec = R.build_lists(it, probes[it["item_id"]]["records"], abstracts, rationales[it["item_id"]],
                                query_encoder=query_encoder, article_encoder=article_encoder, reranker=reranker)
            h.write(json.dumps(rec) + "\n")
            h.flush()
            on_progress(k, len(todo))
    return len(todo)


def lists_config(encoders: dict) -> dict:
    return {"encoders": encoders, "settings_sha256": R.settings_hash(), "quota": R.QUOTA, "keep": R.RERANK_KEEP}


# --------------------------------------------------------------------------------------
# Step 3: filter (and the optional directness judge): one Yes/No token, its probability
# --------------------------------------------------------------------------------------

class LlamaYesNo:
    """A GGUF model answering Yes/No; P(yes) from the first output token's log-probabilities."""

    def __init__(self, model_path, n_ctx: int = R.FILTER_N_CTX, n_threads: Optional[int] = None,
                 n_gpu_layers: int = 0, build: Callable[[str, str], tuple[str, str]] = R.build_filter_messages):
        from llama_cpp import Llama
        self.build = build
        self.n_ctx = n_ctx
        self.model_sha256 = file_sha256(model_path)
        self.name = f"llama-logprobs:{Path(model_path).name}"
        self._llm = Llama(model_path=str(model_path), n_ctx=n_ctx, n_threads=n_threads,
                          n_gpu_layers=n_gpu_layers, logits_all=True, verbose=False, seed=SEED)

    def score(self, question: str, study: str) -> tuple[Optional[float], float]:
        system, user = self.build(question, study)
        reply = self._llm.create_chat_completion(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=1, temperature=0.0, logprobs=True, top_logprobs=20)
        return R.yes_probability(reply["choices"][0]["logprobs"]["content"][0]["top_logprobs"])


def judge_one(scorer, question: str, cand: dict, clock=time.time) -> dict:
    study = study_snippet(cand)
    started = clock()
    p_yes, mass = scorer.score(question, study)
    valid = p_yes is not None and mass >= R.MIN_YES_NO_MASS
    return {"pmid": cand["pmid"], "p_yes": round(p_yes, 6) if p_yes is not None else None,
            "mass": round(mass, 4), "valid": bool(valid), "seconds": round(clock() - started, 2),
            "snippet_sha256": R.sha256(study)[:16]}


def run_judgements(tasks: Sequence[tuple[dict, dict]], scorer, out: Path, *, clock=time.time,
                   on_progress: Callable[[int, int], None] = lambda a, b: None) -> int:
    """``tasks``: (item, candidate) pairs; one record per (item, pmid), finished ones skipped."""
    done = {(r["item_id"], r["pmid"]) for r in load_jsonl(out)}
    todo, seen = [], set(done)
    for item, cand in tasks:
        key = (item["item_id"], cand["pmid"])
        if key not in seen:
            seen.add(key)
            todo.append((item, cand))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8", newline="\n") as h:
        for k, (item, cand) in enumerate(todo, 1):
            rec = judge_one(scorer, item["question"], cand, clock)
            rec["item_id"] = item["item_id"]
            h.write(json.dumps(rec) + "\n")
            h.flush()
            on_progress(k, len(todo))
    return len(todo)


def filter_tasks(items: Sequence[dict], lists: dict, variants: Sequence[str]) -> list[tuple[dict, dict]]:
    return [(it, c) for it in items for v in variants for c in lists[it["item_id"]]["lists"][v]]


def judge_config(scorer, template: str, system: str) -> dict:
    return {"backend": scorer.name, "model_sha256": scorer.model_sha256, "n_ctx": scorer.n_ctx,
            "system_sha256": R.sha256(system), "template_sha256": R.sha256(template),
            "snippet": R.SETTINGS["filter_snippet"], "temperature": 0.0, "seed": SEED}


def scores_by_item(records: Sequence[dict]) -> dict:
    out: dict = {}
    for r in records:
        out.setdefault(r["item_id"], {})[r["pmid"]] = r
    return out


# --------------------------------------------------------------------------------------
# Step 4: answers
# --------------------------------------------------------------------------------------

#: Words kept per abstract when a prompt does not fit the context window. This is not a design setting: it is
#: used only in place of a crash (llama.cpp raises an error for a prompt longer than the context), it never
#: applies to a prompt that fits, and every answer written under it carries ``"context_truncated": true``.
OVERFLOW_ABSTRACT_WORDS = 200


def is_context_overflow(exc: Exception) -> bool:
    return isinstance(exc, ValueError) and "exceed context window" in str(exc).lower()


def shorten_abstracts(passages: Sequence[dict], words: int = OVERFLOW_ABSTRACT_WORDS) -> list[dict]:
    return [dict(p, abstract=" ".join((p.get("abstract") or "").split()[:words])) for p in passages]


def generate_fitting(generate: Callable[[str, str], str], build: Callable[[Sequence[dict]], tuple[str, str]],
                     passages: Sequence[dict]) -> tuple[str, str, bool]:
    """(text, the user prompt that was used, whether the abstracts had to be shortened)."""
    system, user = build(passages)
    try:
        return generate(system, user), user, False
    except ValueError as exc:
        if not is_context_overflow(exc):
            raise
    system, user = build(shorten_abstracts(passages))
    return generate(system, user), user, True


def _evidence_fields(admitted: Sequence[dict]) -> dict:
    return {"admitted": [c["pmid"] for c in admitted], "admitted_lower": [c["lower"] for c in admitted],
            "admitted_upper": [c["upper"] for c in admitted], "admitted_stratum": [c["stratum"] for c in admitted]}


def answer_one(item: dict, arm: str, lists: dict, scores: dict, existing: dict,
               generate: Callable[[str, str], str], clock=time.time) -> dict:
    question = item["question"]
    admitted = R.admit("R2" if arm in R.CRITERIA_ARMS else arm, lists, scores)
    base = {"item_id": item["item_id"], "arm": arm, "settings": R.settings_hash(), **_evidence_fields(admitted)}
    started = clock()
    if arm not in R.CRITERIA_ARMS:
        text, prompt, cut = generate_fitting(generate, lambda ps: (R.ANSWER_SYSTEM, build_prompt(question, ps)),
                                             admitted)
        return dict(base, prompt_sha256=R.sha256(prompt)[:16], text=text, verdict=parse_verdict(text),
                    seconds=round(clock() - started, 1), **({"context_truncated": True} if cut else {}))
    draft = existing.get((item["item_id"], "R2"))
    if draft is None:
        raise RuntimeError(f"{item['item_id']}: R2 must be answered before {arm}")
    extra = {"draft_verdict": draft["verdict"]}
    if not admitted:
        return dict(base, **extra, prompt_sha256=None, text=draft["text"], verdict=draft["verdict"],
                    valid=True, changed=False, fallback="no evidence", seconds=0.0)
    text, user, cut = generate_fitting(
        generate, lambda ps: R.build_criteria_prompt(question, ps, draft=None if arm == "R2C" else draft["text"],
                                                     dated=arm != "R2V-ND"), admitted)
    parsed = R.parse_final_verdict(text)
    verdict = parsed if (parsed is not None or arm == "R2C") else draft["verdict"]
    return dict(base, **extra, prompt_sha256=R.sha256(user)[:16], text=text, verdict=verdict,
                valid=parsed is not None, changed=verdict != draft["verdict"], fallback=None,
                seconds=round(clock() - started, 1), **({"context_truncated": True} if cut else {}))


def run_answers(items: Sequence[dict], lists: dict, scores: dict, arms: Sequence[str], out: Path,
                generate: Callable[[str, str], str], *, clock=time.time,
                on_progress: Callable[[int, int], None] = lambda a, b: None) -> int:
    """All requested arms of one item before the next item (R2 first: the criteria arms use its answer)."""
    existing = {(r["item_id"], r["arm"]): r for r in load_jsonl(out)}
    order = [a for a in R.ARMS if a in arms]
    todo = [(it, a) for it in items for a in order if (it["item_id"], a) not in existing]
    for it, a in todo:
        if a in R.CRITERIA_ARMS and "R2" not in order and (it["item_id"], "R2") not in existing:
            raise RuntimeError(f"{it['item_id']}: answer R2 before {a} (add R2 to --arms)")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8", newline="\n") as h:
        for k, (it, a) in enumerate(todo, 1):
            rec = answer_one(it, a, lists[it["item_id"]]["lists"], scores.get(it["item_id"], {}), existing,
                             generate, clock)
            existing[(it["item_id"], a)] = rec
            h.write(json.dumps(rec) + "\n")
            h.flush()
            on_progress(k, len(todo))
    return len(todo)


def answers_config(model_path: str, model_sha256: str, n_threads: Optional[int]) -> dict:
    return {"model_file": Path(model_path).name, "model_sha256": model_sha256, "n_ctx": R.ANSWER_N_CTX,
            "max_new_tokens": R.ANSWER_MAX_NEW_TOKENS, "temperature": 0.0, "seed": SEED,
            "settings_sha256": R.settings_hash(), "prompts": R.prompt_hashes(), "n_threads": n_threads,
            "llama_cpp_python": llama_version()}


# --------------------------------------------------------------------------------------
# Optional: directness of what B1 and R2 admitted, judged by an independent model
# --------------------------------------------------------------------------------------

def directness_tasks(items: Sequence[dict], frozen: dict, b1: dict, lists: dict, r2: dict) -> list[tuple[dict, dict]]:
    tasks = []
    for it in items:
        iid = it["item_id"]
        pool = {c["pmid"]: c for c in frozen.get(iid, {}).get("candidates", [])}
        listed = {c["pmid"]: c for v in R.LISTS for c in lists.get(iid, {}).get("lists", {}).get(v, [])}
        for pmid in (b1.get(iid) or {}).get("admitted", []):
            if pmid in pool:
                tasks.append((it, pool[pmid]))
        for pmid in (r2.get(iid) or {}).get("admitted", []):
            if pmid in listed:
                tasks.append((it, listed[pmid]))
    return tasks


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------

def _model_ok(path: Optional[str]) -> bool:
    if not path or not Path(path).is_file():
        print(f"model file not found: {path}", file=sys.stderr)
        return False
    return True


def cmd_rationale(a, data: Path) -> int:
    if not _model_ok(a.model_path):
        return 2
    items = load_items(data, a.split, a.limit)
    out = Path(a.out or out_path(data, "rationales", a.split))
    print("hashing the model file ...", flush=True)
    differs = check_config(out, rationale_config(a.model_path, file_sha256(a.model_path)),
                           ("model_sha256", "n_ctx", "max_new_tokens", "temperature", "seed", "system_sha256",
                            "template_sha256"))
    if differs:
        return _refuse(out, differs)
    gen = llama_generator(a.model_path, RATIONALE_N_CTX, a.n_threads, 0, R.RATIONALE_MAX_NEW_TOKENS)
    n = run_rationales(items, gen, out, on_progress=_printer())
    print(f"wrote {n} rationales -> {out}")
    return 0


def cmd_lists(a, data: Path) -> int:
    items = load_items(data, a.split, a.limit)
    rationales = {r["item_id"]: r["rationale"] for r in load_jsonl(out_path(data, "rationales", a.split))}
    probes = {}
    for it in items:
        f = data / "pubmed_g0" / f"{it['item_id']}.json"
        if not f.is_file():
            print(f"missing as-of PubMed records for {it['item_id']} ({f})", file=sys.stderr)
            return 2
        probes[it["item_id"]] = json.loads(f.read_text(encoding="utf-8"))
    missing = [it["item_id"] for it in items if it["item_id"] not in rationales]
    if missing:
        print(f"no rationale for {len(missing)} items (run the rationale step first): {missing[:3]}",
              file=sys.stderr)
        return 2
    abstracts = {r["pmid"]: r for r in load_jsonl(data / "abstracts.jsonl")}
    from experiments.medchange.encoders import (MedCPTReranker, medcpt_article_encoder,
                                                       medcpt_query_encoder)
    qe, ae, rr = (medcpt_query_encoder(device=a.device), medcpt_article_encoder(device=a.device),
                  MedCPTReranker(device=a.device))
    out = Path(a.out or out_path(data, "lists", a.split))
    differs = check_config(out, lists_config({"query": qe.name, "article": ae.name, "reranker": rr.name}),
                           ("encoders", "settings_sha256", "quota", "keep"))
    if differs:
        return _refuse(out, differs)
    n = run_lists(items, probes, abstracts, rationales, out, query_encoder=qe, article_encoder=ae, reranker=rr,
                  on_progress=_printer())
    print(f"wrote candidate lists for {n} items -> {out}")
    return 0


def _lists(data: Path, split: str, items: Sequence[dict]) -> Optional[dict]:
    lists = {r["item_id"]: r for r in load_jsonl(out_path(data, "lists", split))}
    missing = [it["item_id"] for it in items if it["item_id"] not in lists]
    if missing:
        print(f"no candidate lists for {len(missing)} items (run the lists step first): {missing[:3]}",
              file=sys.stderr)
        return None
    return lists


def cmd_filter(a, data: Path) -> int:
    if not _model_ok(a.model_path):
        return 2
    items = load_items(data, a.split, a.limit)
    lists = _lists(data, a.split, items)
    if lists is None:
        return 2
    scorer = LlamaYesNo(a.model_path, R.FILTER_N_CTX, a.n_threads)
    out = Path(a.out or out_path(data, "filter", a.split))
    differs = check_config(out, judge_config(scorer, R.FILTER_TEMPLATE, R.FILTER_SYSTEM),
                           ("backend", "model_sha256", "n_ctx", "system_sha256", "template_sha256", "snippet"))
    if differs:
        return _refuse(out, differs)
    n = run_judgements(filter_tasks(items, lists, a.variants), scorer, out, on_progress=_printer(10))
    print(f"judged {n} abstracts -> {out}")
    return 0


def cmd_answers(a, data: Path) -> int:
    if not _model_ok(a.model_path):
        return 2
    items = load_items(data, a.split, a.limit)
    lists = _lists(data, a.split, items)
    if lists is None:
        return 2
    scores = scores_by_item(load_jsonl(out_path(data, "filter", a.split)))
    needed = {R.LIST_OF[arm] for arm in a.arms if arm != "R2-NF"}
    lacking = [it["item_id"] for it in items for v in needed for c in lists[it["item_id"]]["lists"][v]
               if c["pmid"] not in scores.get(it["item_id"], {})]
    if lacking:
        print(f"{len(lacking)} listed abstracts have no filter judgement (run the filter step with --variants "
              f"{' '.join(sorted(needed))}): {lacking[:3]}", file=sys.stderr)
        return 2
    out = Path(a.out or out_path(data, "answers", a.split))
    print("hashing the model file ...", flush=True)
    differs = check_config(out, answers_config(a.model_path, file_sha256(a.model_path), a.n_threads),
                           ANSWER_RELEVANT)
    if differs:
        return _refuse(out, differs)
    gen = llama_generator(a.model_path, R.ANSWER_N_CTX, a.n_threads, 0, R.ANSWER_MAX_NEW_TOKENS)
    n = run_answers(items, lists, scores, a.arms, out, gen, on_progress=_printer())
    print(f"generated {n} answers -> {out}")
    return 0


def cmd_judge(a, data: Path) -> int:
    if not _model_ok(a.model_path):
        return 2
    items = load_items(data, a.split, a.limit)
    lists = _lists(data, a.split, items)
    if lists is None:
        return 2
    frozen = {r["item_id"]: r for r in load_jsonl(data / f"frozen_{a.split}.jsonl")}
    b1 = {r["item_id"]: r for r in load_jsonl(data / f"answers_{a.split}.jsonl") if r["arm"] == "B1"}
    r2 = {r["item_id"]: r for r in load_jsonl(out_path(data, "answers", a.split)) if r["arm"] == "R2"}
    scorer = LlamaYesNo(a.model_path, R.FILTER_N_CTX, a.n_threads, build=R.build_judge_messages)
    out = Path(a.out or out_path(data, "directness", a.split))
    differs = check_config(out, judge_config(scorer, R.JUDGE_TEMPLATE, R.JUDGE_SYSTEM),
                           ("backend", "model_sha256", "n_ctx", "system_sha256", "template_sha256", "snippet"))
    if differs:
        return _refuse(out, differs)
    n = run_judgements(directness_tasks(items, frozen, b1, lists, r2), scorer, out, on_progress=_printer(10))
    print(f"judged {n} admitted abstracts -> {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="step", required=True)
    for name in ("rationale", "lists", "filter", "answers", "judge"):
        p = sub.add_parser(name)
        p.add_argument("--split", default="dev", choices=("dev", "confirm", "ad", "fresh"))
        p.add_argument("--data-dir", default=str(HERE / "data"))
        p.add_argument("--limit", type=int, default=None, help="first N items only (timing test)")
        p.add_argument("--out", default=None)
        if name == "lists":
            p.add_argument("--device", default="cpu")
        else:
            p.add_argument("--model-path", required=True)
            p.add_argument("--n-threads", type=int, default=None)
        if name == "filter":
            p.add_argument("--variants", nargs="+", default=["R2"], choices=list(R.LISTS))
        if name == "answers":
            p.add_argument("--arms", nargs="+", default=["R2", "R2C", "R2V", "R2V-ND"], choices=list(R.ARMS))
    a = ap.parse_args(argv)
    data = Path(a.data_dir)
    if not (data / "benchmark.jsonl").is_file():
        print(f"no benchmark at {data / 'benchmark.jsonl'} (run build_benchmark first)", file=sys.stderr)
        return 2
    return {"rationale": cmd_rationale, "lists": cmd_lists, "filter": cmd_filter, "answers": cmd_answers,
            "judge": cmd_judge}[a.step](a, data)


if __name__ == "__main__":
    sys.exit(main())

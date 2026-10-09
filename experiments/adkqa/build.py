"""Builder of the Alzheimer's-specific question set (``docs/protocol.md`` §8).

    python -m experiments.adkqa.build prepare [--api-key KEY]                       fetch the frozen sources, split, count
    python -m experiments.adkqa.build diagnose                                      why records lack a conclusion; how the split fell
    python -m experiments.adkqa.build draft  --split dev --model-path <Qwen gguf>   draft questions (the drafting model)
    python -m experiments.adkqa.build verify --split dev --model-path <Phi gguf>    independent verifier (--role audit: Qwen)
    python -m experiments.adkqa.build assemble --split dev                          keep rules, question file, manifest, trial figures
    python -m experiments.adkqa.build pools  --split dev [--api-key KEY]            candidate-pool sizes (gate 1)
    python -m experiments.adkqa.build gate1  [--b0-accuracy X] [--repeat-manifest F]  evaluate gate 1 and record it

Drafting of the test split is refused until gate 1 has passed. The generator under test is never used here. Text (abstracts,
drafts, questions) goes to ``experiments/adkqa/data/`` (not tracked); the tracked results hold counts, hashes and offsets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence

from experiments.medchange import label_audit as LA
from experiments.medchange.generate_answers import check_config, file_sha256, llama_generator, llama_version, load_jsonl
from experiments.medchange.pubmed_asof import EUtils, build_term, day_before, query_terms

from . import records as R
from . import spec

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"
LABELS = spec.VERDICTS

DRAFT_SYSTEM = "You are a careful assistant that writes yes/no/maybe research questions from systematic-review abstracts."
DRAFT_TEMPLATE = (
    "Below are the title and abstract of a systematic review, meta-analysis or guideline about Alzheimer's disease.\n\n"
    "Write ONE question: choose a template and copy two short phrases (x and y) EXACTLY as they appear in the title or abstract "
    "(1 to 8 words each, together at most 6 content words).\nTemplates:\n"
    "effect: " + spec.TEMPLATES["effect"] + " (x: an intervention; y: the outcome)\n"
    "association: " + spec.TEMPLATES["association"] + " (x: a factor, exposure or biomarker; y: the outcome)\n"
    "test: " + spec.TEMPLATES["test"] + " (x: a test or tool; y: the diagnostic purpose)\n\n"
    "Then give the verdict that the AUTHORS' CONCLUSION gives for the question: SUPPORTED if it at least partially supports the "
    "claim; REFUTED if it at least partially does not (no effect, no association, poor accuracy); NOT ENOUGH INFORMATION only if "
    "the authors state that the evidence is insufficient, not merely uncertain. If the abstract states no claim of this kind (for "
    "example only prevalence), answer none.\n\n"
    "Answer in exactly four lines:\nTEMPLATE: <effect|association|test|none>\nX: <phrase>\nY: <phrase>\n"
    "VERDICT: <SUPPORTED|REFUTED|NOT ENOUGH INFORMATION|none>\n\nTITLE: <<TITLE>>\nABSTRACT: <<ABSTRACT>>")
_LINE = {k: re.compile(rf"^\s*{k}\s*:\s*(.*?)\s*$", re.IGNORECASE | re.MULTILINE) for k in ("TEMPLATE", "X", "Y", "VERDICT")}


def draft_prompt(row: dict) -> str:
    return DRAFT_TEMPLATE.replace("<<TITLE>>", row["title"]).replace("<<ABSTRACT>>", row["abstract"])


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def norm(s: str) -> str:
    return " ".join(s.lower().split())


def parse_draft(raw: str) -> Optional[dict]:
    got = {k: (m.group(1).strip().strip("\"'*") if (m := rx.search(raw or "")) else None) for k, rx in _LINE.items()}
    if got["TEMPLATE"] is None or got["VERDICT"] is None:
        return None
    verdict = re.sub(r"\s+", " ", got["VERDICT"].upper())
    return {"template": got["TEMPLATE"].lower(), "x": got["X"] or "", "y": got["Y"] or "",
            "verdict": verdict if verdict in LABELS else ("NONE" if verdict.startswith("NONE") else None)}


def check_draft(row: dict, draft: Optional[dict]) -> tuple[str, Optional[str]]:
    """(status, rule label): the first automatic check a draft fails, or ``rule_ok`` with the rule-based reading."""
    if draft is None or draft["verdict"] is None:
        return "draft_unparsed", None
    if draft["template"] == "none" or draft["verdict"] == "NONE":
        return "no_claim", None
    if draft["template"] not in spec.TEMPLATES:
        return "bad_template", None
    hay = norm(row["title"] + " " + row["abstract"])
    if not all(norm(draft[k]) and norm(draft[k]) in hay for k in ("x", "y")):
        return "span_not_in_abstract", None
    if spec.question_checks(draft["template"], draft["x"], draft["y"]):
        return "question_rules", None
    rule = spec.read_conclusion(row["conclusion"])
    if rule is None:
        return "rule_no_verdict", None
    if rule != draft["verdict"]:
        return "rule_disagrees", rule
    return "rule_ok", rule


def question_of(draft: dict) -> str:
    return spec.fill(draft["template"], draft["x"], draft["y"])


def _config(model_path: str, n_ctx: int, system: str, template: str) -> dict:
    return {"model_sha256": file_sha256(model_path), "n_ctx": n_ctx, "system_sha256": sha(system),
            "template_sha256": sha(template), "llama_cpp_python": llama_version()}


def run_draft(rows: Sequence[dict], gen: Callable[[str, str], str], out: Path, progress=lambda k, n: None) -> int:
    done = {r["pmid"] for r in load_jsonl(out)}
    todo = [r for r in rows if r["pmid"] not in done and r["status"] == "eligible"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8", newline="\n") as fh:
        for k, row in enumerate(todo, 1):
            raw = gen(DRAFT_SYSTEM, draft_prompt(row))
            fh.write(json.dumps({"pmid": row["pmid"], "raw": raw, "draft": parse_draft(raw)}, ensure_ascii=False) + "\n")
            fh.flush()
            progress(k, len(todo))
    return len(todo)


def run_verify(rows: Sequence[dict], drafts: dict, gen: Callable[[str, str], str], out: Path, progress=lambda k, n: None) -> int:
    """The verifier reads the question, the objectives and the quoted conclusion only (the audit prompt, unchanged)."""
    done = {r["pmid"] for r in load_jsonl(out)}
    todo = []
    for row in rows:
        d = drafts.get(row["pmid"])
        if row["pmid"] in done or d is None:
            continue
        if check_draft(row, d["draft"])[0] == "rule_ok":
            todo.append((row, d["draft"]))
    with open(out, "a", encoding="utf-8", newline="\n") as fh:
        for k, (row, d) in enumerate(todo, 1):
            raw = gen(LA.SYSTEM, LA.build_prompt(question_of(d), row["objectives"], row["conclusion"]))
            fh.write(json.dumps({"pmid": row["pmid"], "label": spec.parse_verdict(raw), "raw": (raw or "")[:60]}) + "\n")
            fh.flush()
            progress(k, len(todo))
    return len(todo)


def decide(row: dict, drafts: dict, verified: dict) -> dict:
    """Final status of one source record and, when kept, its question."""
    if row["status"] != "eligible":
        return {"status": row["status"]}
    d = drafts.get(row["pmid"])
    if d is None:
        return {"status": "not_drafted"}
    status, rule = check_draft(row, d["draft"])
    if status != "rule_ok":
        return {"status": status}
    v = verified.get(row["pmid"])
    if v is None:
        return {"status": "not_verified"}
    if v["label"] is None:
        return {"status": "verifier_unparsed"}
    if v["label"] != d["draft"]["verdict"]:
        return {"status": "verifier_disagrees", "verifier": v["label"]}
    dr = d["draft"]
    return {"status": "kept", "template": dr["template"], "x": dr["x"], "y": dr["y"], "question": question_of(dr),
            "label": dr["verdict"], "rule_label": rule, "verifier_label": v["label"]}


def select(rows: Sequence[dict], split: str) -> list[dict]:
    """The records of a split in the order they are drafted: the first DRAFT_N development records, or the whole test pool."""
    pool = [r for r in rows if r["split"] == split]
    by = {r["pmid"]: r for r in pool}
    order = [by[p] for p in spec.draft_order(by)]
    return order[:spec.DRAFT_N] if split == "dev" else order


def claim_only_accuracy(questions: Sequence[str], labels: Sequence[str], folds: int = 5) -> dict:
    """5-fold multinomial naive Bayes on the question words alone: how much does the claim reveal about the verdict?"""
    import numpy as np
    words = lambda q: re.findall(r"[a-z][a-z0-9\-]+", q.lower())
    n = len(questions)
    vocab = {w: i for i, w in enumerate(sorted({w for q in questions for w in map(str, words(q))}))}
    X = np.zeros((n, max(len(vocab), 1)))
    for i, q in enumerate(questions):
        for w in words(q):
            X[i, vocab[w]] += 1
    classes = sorted(set(labels))
    y = np.array([classes.index(l) for l in labels])
    correct = 0
    for f in range(folds):
        test = np.arange(n) % folds == f
        train = ~test
        if not test.any() or not train.any():
            continue
        scores = []
        for c in range(len(classes)):
            counts = X[train & (y == c)].sum(axis=0) + 1.0
            prior = math.log(((train & (y == c)).sum() + 1) / (train.sum() + len(classes)))
            scores.append(X[test] @ np.log(counts / counts.sum()) + prior)
        correct += int((np.argmax(np.stack(scores, axis=1), axis=1) == y[test]).sum())
    majority = max(list(labels).count(c) for c in classes) / n if n else 0.0
    return {"accuracy": round(correct / n, 4) if n else None, "majority": round(majority, 4)}


def kappa(pairs: Sequence[tuple[str, str]]) -> Optional[float]:
    return LA.cohens_kappa(list(pairs))


def manifest_hash(items: Sequence[dict]) -> str:
    lines = sorted(json.dumps([i["item_id"], i["source_pmid"], i["date"], i["question"], i["label"]]) for i in items)
    return sha("\n".join(lines))


def _coverage(statuses: dict) -> Optional[float]:
    """Share of well-formed drafts (template, spans and question rules passed) for which the cue rules gave a verdict."""
    gave = sum(statuses.get(k, 0) for k in ("kept", "verifier_disagrees", "verifier_unparsed", "rule_disagrees", "not_verified"))
    total = gave + statuses.get("rule_no_verdict", 0)
    return round(gave / total, 4) if total else None


def assemble(rows: Sequence[dict], drafts: dict, verified: dict, split: str, audit: Optional[dict] = None) -> dict:
    """Apply the keep rules to a split. Returns the items (text; not tracked) and the figures (tracked)."""
    order = select(rows, split)
    decisions = {r["pmid"]: decide(r, drafts, verified) for r in order}
    kept = [r for r in order if decisions[r["pmid"]]["status"] == "kept"]
    if split == "dev":
        kept = kept[:spec.DEV_N]
    elif len(kept) > spec.TEST_MAX:
        kept = sorted(kept, key=lambda r: spec._unit(spec.SPLIT_SEED, "keep", r["pmid"]))[:spec.TEST_MAX]
        kept = [r for r in order if r in kept]
    prefix = "AK-D" if split == "dev" else "AK-T"
    items = []
    for k, r in enumerate(kept, 1):
        d = decisions[r["pmid"]]
        items.append({"item_id": f"{prefix}-{k:04d}", "source_pmid": r["pmid"], "date": r["date"], "split": split,
                      "area": r["area"], "cluster": r["cluster"], "template": d["template"], "x": d["x"], "y": d["y"],
                      "question": d["question"], "label": d["label"], "conclusion_span": r["conclusion_span"],
                      "conclusion_sha256": r["conclusion_sha256"], "rule_label": d["rule_label"],
                      "verifier_label": d["verifier_label"]})
    statuses: dict[str, int] = {}
    for d in decisions.values():
        statuses[d["status"]] = statuses.get(d["status"], 0) + 1
    drafted = [r for r in order if decisions[r["pmid"]]["status"] not in ("not_drafted", "not_verified")]
    reached_verifier = [r for r in order if decisions[r["pmid"]]["status"] in ("kept", "verifier_disagrees", "verifier_unparsed")]
    parsed = [r for r in reached_verifier if decisions[r["pmid"]]["status"] != "verifier_unparsed"]
    agree = [r for r in parsed if decisions[r["pmid"]]["status"] == "kept"]
    labels = [i["label"] for i in items]
    rep = {"split": split, "records_in_scope": len(order), "statuses": statuses, "kept": len(items),
           "survival": round(sum(d["status"] == "kept" for d in decisions.values()) / len(order), 4) if order else None,
           "all_drafted": len(drafted) == len(order),
           "verifier_agreement": round(len(agree) / len(parsed), 4) if parsed else None,
           "verifier_unparsed": len(reached_verifier) - len(parsed),
           "rule_coverage": _coverage(statuses),
           "verdict_share": {c: round(labels.count(c) / len(labels), 4) if labels else None for c in LABELS},
           "per_area": {a: sum(1 for i in items if i["area"] == a) for a in sorted({r["area"] for r in rows})},
           "claim_only": claim_only_accuracy([i["question"] for i in items], labels) if len(items) >= 10 else None,
           "manifest_hash": manifest_hash(items), "settings": {"seed": spec.SPLIT_SEED}}
    if audit is not None:
        pairs = [(i["label"], audit[i["source_pmid"]]["label"]) for i in items if audit.get(i["source_pmid"], {}).get("label")]
        rep["audit"] = {"n": len(pairs), "agreement": round(sum(a == b for a, b in pairs) / len(pairs), 4) if pairs else None,
                        "kappa": kappa(pairs)}
    return {"items": items, "report": rep}


def pool_sizes(items: Sequence[dict], eu: EUtils, retmax: int = 100) -> dict:
    """Candidate pool of each question as the frozen retrieval would find it (AND of the query terms, abstracts only, strictly
    earlier than the source, the source itself removed). Counts are capped at ``retmax``."""
    sizes = []
    for it in items:
        term = build_term(query_terms(it["question"]), mode="and") + " AND hasabstract[text]"
        count, ids = eu.search(term, day_before(it["date"]), retmax)
        ids = [i for i in ids if i != it["source_pmid"]]
        sizes.append(len(ids) if count <= retmax else retmax)
    return {"n": len(sizes), "median": statistics.median(sizes) if sizes else None,
            "empty_share": round(sum(1 for s in sizes if s == 0) / len(sizes), 4) if sizes else None,
            "min": min(sizes) if sizes else None, "retmax": retmax}


def evaluate_gate1(dev: dict, pools: Optional[dict], b0_accuracy: Optional[float], repeat_hash: Optional[str]) -> dict:
    """Gate 1 of protocol §8. A check that cannot be made yet is None; the gate passes only when every check is True."""
    c = {}
    c["survival_at_least_40pct"] = None if dev["survival"] is None else dev["survival"] >= 0.40 and dev["kept"] >= spec.DEV_N
    c["verifier_agreement_at_least_85pct"] = None if dev["verifier_agreement"] is None else dev["verifier_agreement"] >= 0.85
    co = dev.get("claim_only")
    c["claim_only_at_most_majority_plus_5pp"] = None if not co else co["accuracy"] <= co["majority"] + 0.05
    shares = dev["verdict_share"]
    c["each_verdict_at_least_25pct"] = None if None in shares.values() else all(v >= 0.25 for v in shares.values())
    c["same_seed_same_hash"] = None if repeat_hash is None else repeat_hash == dev["manifest_hash"]
    c["pool_median_at_least_15_and_empty_at_most_5pct"] = (
        None if not pools else pools["median"] >= 15 and pools["empty_share"] <= 0.05)
    constant = max(shares.values()) if None not in shares.values() else None
    c["b0_between_constant_plus_5pp_and_80pct"] = (
        None if b0_accuracy is None or constant is None else constant + 0.05 <= b0_accuracy <= 0.80)
    return {"checks": c, "passed": all(v is True for v in c.values()), "pending": sorted(k for k, v in c.items() if v is None),
            "failed": sorted(k for k, v in c.items() if v is False)}


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def _read_rows(data: Path) -> list[dict]:
    rows = load_jsonl(data / "sources.jsonl")
    if not rows:
        raise SystemExit("no prepared sources: run `prepare` first")
    return rows


def main(argv=None, *, eu: Optional[EUtils] = None, generator: Optional[Callable] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("cmd", choices=("prepare", "diagnose", "draft", "verify", "assemble", "pools", "gate1"))
    ap.add_argument("--split", default="dev", choices=("dev", "test"))
    ap.add_argument("--model-path")
    ap.add_argument("--role", default="verifier", choices=("verifier", "audit"))
    ap.add_argument("--data-dir", default=str(DATA))
    ap.add_argument("--results-dir", default=str(RESULTS))
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--n-threads", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="first N records of the split only (for a quick look)")
    ap.add_argument("--b0-accuracy", type=float, default=None)
    ap.add_argument("--repeat-manifest", default=None, help="manifest of an independent second build, for the same-hash check")
    a = ap.parse_args(argv)
    data, results = Path(a.data_dir), Path(a.results_dir)

    if a.cmd == "prepare":
        counts = json.loads(R.SOURCE_FILE.read_text(encoding="utf-8"))
        areas = R.area_assignment(counts)
        records = R.fetch_records(eu or EUtils(a.api_key), sorted(areas))
        rows = R.prepare(records, areas)
        R.write_jsonl(data / "sources.jsonl", rows)
        rep = R.pools(rows)
        _write_json(results / "adkqa_pools.json", rep)
        print(json.dumps(rep, indent=2))
        return 0

    rows = _read_rows(data)
    if a.cmd == "diagnose":
        rep = R.diagnose(rows)
        _write_json(results / "adkqa_diagnose.json", rep)
        print(json.dumps(rep, indent=2))
        return 0
    scope = select(rows, a.split)[:a.limit]
    if a.cmd in ("draft", "verify"):
        if not a.model_path or not Path(a.model_path).is_file():
            print("--model-path must name a model file", file=sys.stderr)
            return 2
    if a.cmd == "draft":
        if a.split == "test":
            g = results / "adkqa_gate1.json"
            if not g.is_file() or not json.loads(g.read_text(encoding="utf-8")).get("passed"):
                print("the test split is not drafted until gate 1 has passed (adkqa_gate1.json)", file=sys.stderr)
                return 2
        out = data / f"drafts_{a.split}.jsonl"
        differs = check_config(out, _config(a.model_path, a.n_ctx, DRAFT_SYSTEM, DRAFT_TEMPLATE),
                               ("model_sha256", "n_ctx", "system_sha256", "template_sha256"))
        if differs:
            print(f"refusing to extend {out.name}: configuration differs in {', '.join(differs)}", file=sys.stderr)
            return 2
        gen = generator or llama_generator(a.model_path, a.n_ctx, a.n_threads, 0, 120)
        n = run_draft(scope, gen, out, lambda k, t: print(f"  {k}/{t}", flush=True) if k % 10 == 0 or k == t else None)
        print(f"drafted {n} records -> {out}")
        return 0
    if a.cmd == "verify":
        drafts = {r["pmid"]: r for r in load_jsonl(data / f"drafts_{a.split}.jsonl")}
        out = data / f"{a.role}_{a.split}.jsonl"
        differs = check_config(out, _config(a.model_path, a.n_ctx, LA.SYSTEM, LA.TEMPLATE),
                               ("model_sha256", "n_ctx", "system_sha256", "template_sha256"))
        if differs:
            print(f"refusing to extend {out.name}: configuration differs in {', '.join(differs)}", file=sys.stderr)
            return 2
        gen = generator or llama_generator(a.model_path, a.n_ctx, a.n_threads, 0, 12)
        n = run_verify(scope, drafts, gen, out, lambda k, t: print(f"  {k}/{t}", flush=True) if k % 10 == 0 or k == t else None)
        print(f"{a.role}: labelled {n} drafts -> {out}")
        return 0

    drafts = {r["pmid"]: r for r in load_jsonl(data / f"drafts_{a.split}.jsonl")}
    verified = {r["pmid"]: r for r in load_jsonl(data / f"verifier_{a.split}.jsonl")}
    if a.cmd == "assemble":
        audit = {r["pmid"]: r for r in load_jsonl(data / f"audit_{a.split}.jsonl")} or None
        built = assemble(rows, drafts, verified, a.split, audit)
        R.write_jsonl(data / f"adkqa_{a.split}.jsonl", built["items"])
        rep = dict(built["report"])
        _write_json(results / f"adkqa_build_{a.split}.json", rep)
        _write_json(results / f"adkqa_manifest_{a.split}.json", {
            "manifest_hash": rep["manifest_hash"], "items": [{k: i[k] for k in (
                "item_id", "source_pmid", "date", "area", "cluster", "template", "label", "conclusion_span", "conclusion_sha256")}
                for i in built["items"]]})
        print(json.dumps(rep, indent=2))
        return 0
    if a.cmd == "pools":
        items = load_jsonl(data / f"adkqa_{a.split}.jsonl")
        rep = pool_sizes(items, eu or EUtils(a.api_key))
        _write_json(results / f"adkqa_pools_{a.split}.json", rep)
        print(json.dumps(rep, indent=2))
        return 0
    if a.cmd == "gate1":
        dev = json.loads((results / "adkqa_build_dev.json").read_text(encoding="utf-8"))
        pp = results / "adkqa_pools_dev.json"
        pools = json.loads(pp.read_text(encoding="utf-8")) if pp.is_file() else None
        rep_hash = None
        if a.repeat_manifest:
            rep_hash = json.loads(Path(a.repeat_manifest).read_text(encoding="utf-8"))["manifest_hash"]
        g = evaluate_gate1(dev, pools, a.b0_accuracy, rep_hash)
        _write_json(results / "adkqa_gate1.json", g)
        print(json.dumps(g, indent=2))
        return 0 if g["passed"] else 3
    return 2


if __name__ == "__main__":
    sys.exit(main())

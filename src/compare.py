#!/usr/bin/env python3
"""
Compares several runs on the same cases: each run's accuracy, and for every pair McNemar's exact test on the questions exactly one
of the two answers right, with Holm's correction for the number of pairs (Holm 1979), since a page of many comparisons would
otherwise show some "significant" differences by chance.

A run is name=questions_file,predictions_file. The question files may differ (e.g. two views of the same cases, whose options are
the same per case); every run must hold the same case ids. An answer is read with its own question's options; a missing or invalid
answer is wrong.

Usage:
  python src/compare.py --out results/llm_only/compare_text_only_test.json \
      --run claude=data/views/text_only/test.jsonl,results/llm_only/claude-sonnet-5-5/text_only_test/predictions.jsonl \
      --run gpt-4o=data/views/text_only/test.jsonl,results/llm_only/gpt-4o/text_only_test/predictions.jsonl
"""
import argparse
import itertools

from common import get_logger, write_json
from evaluate import file_record, gold_class, holm, load, mcnemar_exact, predicted_class


def correct_by_id(questions, predictions):
    unknown = set(predictions) - set(questions)
    if unknown:
        raise ValueError(f"{len(unknown)} predictions are not in the question file, e.g. {sorted(unknown)[:3]}")
    return {i: predicted_class(q, predictions.get(i)) == gold_class(q) for i, q in questions.items()}


def compare(runs):
    """runs: {name: {id: right (bool)}}, all on the same ids. Returns the accuracy of each run and the tests of every pair."""
    ids = None
    for name, right in runs.items():
        if ids is None:
            ids = sorted(right)
        elif sorted(right) != ids:
            raise ValueError(f"run {name!r} holds other cases than the first run")
    accuracy = {name: round(sum(right.values()) / len(right), 4) for name, right in runs.items()}
    pairs = []
    for a, b in itertools.combinations(runs, 2):
        truth = [True] * len(ids)                    # McNemar on right/wrong: "gold" is True, a run's answer is its rightness
        test = mcnemar_exact(truth, [runs[a][i] for i in ids], [runs[b][i] for i in ids])
        pairs.append({"a": a, "b": b, "accuracy_a": accuracy[a], "accuracy_b": accuracy[b], **test})
    for pair, adjusted in zip(pairs, holm([p["p_value"] for p in pairs])):
        pair["p_holm"] = round(adjusted, 6)
    return {"n_questions": len(ids), "accuracy": accuracy, "pairs": pairs}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compares several runs on the same cases (McNemar + Holm).")
    ap.add_argument("--run", action="append", required=True, help="name=questions_file,predictions_file (two or more)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if len(a.run) < 2:
        ap.error("give at least two --run")
    runs, inputs = {}, {}
    for spec in a.run:
        name, _, files = spec.partition("=")
        qpath, _, ppath = files.partition(",")
        if not (name and qpath and ppath) or name in runs:
            ap.error(f"--run must be a new name=questions_file,predictions_file (got {spec!r})")
        runs[name] = correct_by_id(load(qpath), load(ppath))
        inputs[name] = {"questions": file_record(qpath), "predictions": file_record(ppath)}
    rep = {"inputs": inputs, **compare(runs)}
    write_json(a.out, rep)
    log = get_logger("compare")
    for p in rep["pairs"]:
        log.info(f"{p['a']} {p['accuracy_a']} vs {p['b']} {p['accuracy_b']}: only a right {p['a_right_b_wrong']}, only b right "
                 f"{p['a_wrong_b_right']}, p = {p['p_value']}, Holm p = {p['p_holm']}")
    return rep


if __name__ == "__main__":
    main()

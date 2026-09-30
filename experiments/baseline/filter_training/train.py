"""CLI: fine-tune Flan-T5-large as the RAG² admission filter.

Trains on labels from ``build_labels.py`` (general-medical MedQA - never the
thesis's Alzheimer's questions, specification SS10.6). Adds
``[HELPFUL]``/``[NOT_HELPFUL]`` as special tokens and resizes the embedding
matrix, exactly as the released ``classifier/model/token_add.ipynb`` does
(specification SS10.3) - ``FlanT5RAG2Filter`` refuses a checkpoint that
skips this.

    python -m experiments.baseline.filter_training.train \\
        --labels experiments/baseline/filter_training/labels/medqa_filter_labels.json \\
        --output-dir checkpoints/rag2_filter \\
        --epochs 3

``--epochs``: the paper trains 40 (``FilterTrainingConfig``'s reference
value); what fits this machine depends on the labelled-set size and measured
step time, so it must be chosen and recorded, not defaulted.

Local (laptop, CPU) recipe - every switch below is reported as a deviation in
the checkpoint record:

    --cpu --precision fp32 --optimizer adafactor --gradient-checkpointing

``--calibrate-steps N`` runs N optimizer steps, prints seconds/step and peak
memory, and writes NO checkpoint - use it to choose ``--epochs``. A killed
run is continued with ``--resume`` (refused if the settings changed).
Validation is scored with the deployed two-way rule (see ``metrics.py``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .config import LABEL_TOKENS, CheckpointRecord, ConfigError, FilterTrainingConfig
from .labeling import HELPFUL, NOT_HELPFUL
from .metrics import classification_metrics

FINGERPRINT_FILE = "train_fingerprint.json"


def training_fingerprint(
    config: FilterTrainingConfig, *, labels_sha256: str, val_fraction: float,
) -> dict:
    """Everything that must be identical for ``--resume`` to be valid."""
    fp = config.to_dict()
    fp.pop("deviations_from_paper", None)
    fp["labels_sha256"] = labels_sha256
    fp["val_fraction"] = val_fraction
    return fp


def check_resume_fingerprint(expected: dict, saved: dict) -> None:
    diffs = [k for k in sorted(set(expected) | set(saved))
             if expected.get(k) != saved.get(k)]
    if diffs:
        raise ConfigError(
            "cannot resume: settings differ from the interrupted run in "
            + ", ".join(f"{k} ({saved.get(k)!r} -> {expected.get(k)!r})"
                        for k in diffs)
        )


#: Written by the HF Trainer for a fully saved checkpoint. A kill (power
#: loss, closed lid) mid-save leaves a checkpoint-N directory missing some of
#: these; ``get_last_checkpoint`` picks it anyway and resume then crashes.
_CHECKPOINT_REQUIRED = ("model.safetensors", "optimizer.pt", "scheduler.pt",
                        "rng_state.pth", "trainer_state.json")


def latest_complete_checkpoint(output_dir: str | Path) -> Path | None:
    """Newest ``checkpoint-N`` holding every required file, else None."""
    root = Path(output_dir)
    if not root.is_dir():
        return None
    numbered = []
    for d in root.iterdir():
        name = d.name
        if d.is_dir() and name.startswith("checkpoint-") \
                and name[len("checkpoint-"):].isdigit():
            numbered.append((int(name[len("checkpoint-"):]), d))
    for _n, d in sorted(numbered, reverse=True):
        if all((d / f).is_file() and (d / f).stat().st_size > 0
               for f in _CHECKPOINT_REQUIRED):
            return d
        print(f"ignoring incomplete checkpoint (interrupted save): {d.name}")
    return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_labelled_examples(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def split_train_val(
    records: list[dict], *, val_fraction: float = 0.1, seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """A plain data split for training THIS classifier.

    Unrelated to, and must not be confused with, the thesis's own
    Alzheimer's validation/test split
    (``experiments/shared/questions/splits.json``), which exists to fit
    lambda/theta/H (roadmap step 5) - a completely different purpose from
    splitting MedQA labels to train a filter (roadmap step 4). Naming
    keeps them apart: this function's outputs are "filter train"/"filter
    val", never "the validation split".
    """
    if not 0.0 < val_fraction < 1.0:
        raise ConfigError(f"val_fraction must be in (0, 1), got {val_fraction}")
    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)
    n_val = max(1, round(len(shuffled) * val_fraction))
    if n_val >= len(shuffled):
        raise ConfigError(
            f"val_fraction={val_fraction} leaves no training examples for "
            f"{len(shuffled)} records"
        )
    return shuffled[n_val:], shuffled[:n_val]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--labels", required=True,
                    help="training-label JSON from build_labels.py")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--base-model", default=None,
                    help="override the base checkpoint (recorded as a "
                         "deviation; used for offline tests)")
    ap.add_argument("--optimizer", choices=("adamw", "adafactor"), default="adamw")
    ap.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="force CPU training")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run in --output-dir")
    ap.add_argument("--early-stopping-patience", type=int, default=0,
                    help="stop after N epochs without balanced-accuracy gain "
                         "(0 = off)")
    ap.add_argument("--calibrate-steps", type=int, default=0,
                    help="run N optimizer steps, report speed/memory, write "
                         "no checkpoint")
    args = ap.parse_args(argv)

    out_dir = Path(args.output_dir)
    calibrating = args.calibrate_steps > 0
    if not calibrating and out_dir.exists() and any(out_dir.iterdir()) \
            and not args.resume:
        print(f"refusing to write into a non-empty directory: {out_dir} "
              "(pass --resume to continue an interrupted run)", file=sys.stderr)
        return 2
    if args.resume and calibrating:
        print("--resume and --calibrate-steps cannot be combined",
              file=sys.stderr)
        return 2

    extra = {"base_model": args.base_model} if args.base_model else {}
    config = FilterTrainingConfig(
        **extra, epochs=args.epochs, seed=args.seed, optimizer=args.optimizer,
        mixed_precision=args.precision,
        gradient_checkpointing=args.gradient_checkpointing, use_cpu=args.cpu,
    )
    try:
        config.validate()
    except ConfigError as exc:
        print(f"invalid training config: {exc}", file=sys.stderr)
        return 2
    print("training config (paper value alongside any deviation):")
    print(json.dumps(config.to_dict(), indent=2))

    try:
        import numpy as np
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            DataCollatorForSeq2Seq,
            EarlyStoppingCallback,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )
    except ImportError as exc:
        print(f"train.py requires torch, transformers, datasets: {exc}",
              file=sys.stderr)
        return 2

    labels_path = Path(args.labels)
    records = load_labelled_examples(labels_path)
    if not records:
        print("no labelled examples to train on", file=sys.stderr)
        return 2

    fingerprint = training_fingerprint(
        config, labels_sha256=_sha256(labels_path),
        val_fraction=args.val_fraction)
    last_checkpoint = None
    if args.resume:
        fp_path = out_dir / FINGERPRINT_FILE
        if not fp_path.exists():
            print(f"cannot resume: {fp_path} not found", file=sys.stderr)
            return 2
        try:
            check_resume_fingerprint(
                fingerprint, json.loads(fp_path.read_text(encoding="utf-8")))
        except ConfigError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        last_checkpoint = latest_complete_checkpoint(out_dir)
        last_checkpoint = str(last_checkpoint) if last_checkpoint else None
        print(f"resuming from: {last_checkpoint or 'no checkpoint yet (start)'}")

    train_records, val_records = split_train_val(
        records, val_fraction=args.val_fraction, seed=args.seed
    )
    print(f"filter-train examples: {len(train_records)}  "
          f"filter-val examples: {len(val_records)}")

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    n_added = tokenizer.add_tokens(list(LABEL_TOKENS))
    print(f"added {n_added} new label token(s): {LABEL_TOKENS}")
    if n_added != len(LABEL_TOKENS):
        print(
            "WARNING: expected to add both label tokens as new; one may "
            "already exist in this tokenizer's vocabulary. Verify "
            "FlanT5RAG2Filter's single-token check passes on this "
            "checkpoint before trusting it.",
            file=sys.stderr,
        )

    # Fail fast: each label must be exactly ONE token (+EOS), or the
    # deployed rule (first-decoder-position logits) is not what is trained.
    helpful_id, not_helpful_id = tokenizer.convert_tokens_to_ids(list(LABEL_TOKENS))
    for token, tid in zip(LABEL_TOKENS, (helpful_id, not_helpful_id)):
        ids = tokenizer(token)["input_ids"]
        if ids != [tid, tokenizer.eos_token_id]:
            print(f"label {token} tokenizes to {ids}, expected "
                  f"[{tid}, {tokenizer.eos_token_id}]", file=sys.stderr)
            return 2
    if helpful_id == not_helpful_id:
        print("label tokens share an id", file=sys.stderr)
        return 2

    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model)
    model.resize_token_embeddings(len(tokenizer))
    if config.gradient_checkpointing:
        model.config.use_cache = False

    def to_hf_dataset(recs: list[dict]) -> "Dataset":
        return Dataset.from_dict({
            "question": [r["question"] for r in recs],
            "answer": [r["answer"] for r in recs],
        })

    def preprocess(batch):
        model_inputs = tokenizer(
            batch["question"], truncation=True, max_length=config.max_seq_length,
        )
        labels = tokenizer(batch["answer"], truncation=True, max_length=8)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_ds = to_hf_dataset(train_records).map(preprocess, batched=True)
    val_ds = to_hf_dataset(val_records).map(preprocess, batched=True)

    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    def preprocess_logits(logits, labels):
        """The deployed rule, on teacher-forced logits: decoder position 0
        (input = decoder-start) scores the first label token; two-way
        comparison of [HELPFUL] vs [NOT_HELPFUL]. Returns 1 = helpful. Keeps
        eval memory tiny (no batch x len x vocab accumulation)."""
        if isinstance(logits, tuple):
            logits = logits[0]
        two = logits[:, 0, [helpful_id, not_helpful_id]]
        return (two[:, 0] >= two[:, 1]).long()

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predicted = np.asarray(predictions).reshape(-1) == 1
        gold = np.asarray(labels)[:, 0] == helpful_id
        return classification_metrics(predicted.tolist(), gold.tolist())

    use_bf16 = config.mixed_precision == "bf16"
    use_fp16 = config.mixed_precision == "fp16"
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_batch_size,
        per_device_eval_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.epochs,
        max_steps=args.calibrate_steps if calibrating else -1,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        optim=config.optimizer,
        gradient_checkpointing=config.gradient_checkpointing,
        use_cpu=config.use_cpu,
        bf16=use_bf16, fp16=use_fp16,
        predict_with_generate=False,
        eval_strategy="no" if calibrating else "epoch",
        save_strategy="no" if calibrating else "epoch",
        save_total_limit=2,
        load_best_model_at_end=not calibrating,
        metric_for_best_model="balanced_accuracy",
        greater_is_better=True,
        logging_steps=1 if calibrating else 10,
        seed=config.seed,
        report_to=[],
    )

    callbacks = []
    if args.early_stopping_patience > 0 and not calibrating:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience))

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits,
        callbacks=callbacks,
    )

    if not calibrating:
        out_dir.mkdir(parents=True, exist_ok=True)
        fp_path = out_dir / FINGERPRINT_FILE
        if not fp_path.exists():
            fp_path.write_text(json.dumps(fingerprint, indent=2, sort_keys=True),
                               encoding="utf-8")

    t0 = time.time()
    trainer.train(resume_from_checkpoint=last_checkpoint)
    elapsed = time.time() - t0

    if calibrating:
        n = args.calibrate_steps
        per_step = elapsed / n
        steps_per_epoch = max(1, len(train_ds) // config.effective_batch_size)
        print(f"\nCALIBRATION: {n} optimizer steps in {elapsed:.0f} s "
              f"-> {per_step:.1f} s/step")
        print(f"  ~{steps_per_epoch} optimizer steps/epoch "
              f"-> ~{per_step * steps_per_epoch / 3600:.2f} h/epoch "
              "(evaluation extra)")
        try:
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            print(f"  peak RSS ~{peak:.0f} MB")
        except ImportError:  # Windows: watch Task Manager / ram_monitor.csv
            print("  peak memory: read Task Manager (no resource module)")
        print("no checkpoint written (calibration mode)")
        return 0

    m = trainer.evaluate()
    val_accuracy = float(m["eval_accuracy"])
    balanced = float(m["eval_balanced_accuracy"])
    majority = float(m["eval_majority_baseline"])
    print(f"validation (deployed two-way rule): accuracy {val_accuracy:.4f}  "
          f"balanced {balanced:.4f}  majority baseline {majority:.4f}  "
          f"recall helpful {m['eval_recall_helpful']:.3f}  "
          f"recall not-helpful {m['eval_recall_not_helpful']:.3f}")

    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    label_dist: dict[str, int] = {}
    for r in records:
        label_dist[r["answer"]] = label_dist.get(r["answer"], 0) + 1

    epochs_run = max(1, int(round(trainer.state.epoch or config.epochs)))
    record = CheckpointRecord(
        checkpoint_path=str(final_dir),
        base_model=config.base_model,
        n_training_examples=len(train_records),
        n_validation_examples=len(val_records),
        validation_accuracy=val_accuracy,
        epochs_run=epochs_run,
        label_distribution=label_dist,
        trained_on="MedQA (general-medical), never the thesis's Alzheimer's "
                   "questions - see experiments/baseline/filter_training/medqa_data.py",
        notes=f"deviations from paper: {config.deviations()}; "
              f"best epoch by balanced accuracy restored; "
              f"recall_helpful={m['eval_recall_helpful']:.3f}, "
              f"recall_not_helpful={m['eval_recall_not_helpful']:.3f}",
        balanced_accuracy=balanced,
        majority_baseline=majority,
    )
    record.validate()
    (out_dir / "checkpoint_record.json").write_text(
        json.dumps(asdict(record), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"usable as baseline filter (beats majority baseline {majority:.3f} "
          f"and balanced accuracy > 0.5): {record.is_usable()}")
    print(f"checkpoint saved to {final_dir}")
    print(f"checkpoint record saved to {out_dir / 'checkpoint_record.json'}")
    if not record.is_usable():
        print(
            "WARNING: this checkpoint does not beat the majority-class "
            "baseline and/or has balanced accuracy <= 0.5. It has not "
            "learned the label function and must not be passed to "
            "--rag2-checkpoint as a thesis baseline "
            "(specification SS10.5 / FlanT5RAG2Filter's own contract).",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

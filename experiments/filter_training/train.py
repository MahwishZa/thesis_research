"""CLI: fine-tune Flan-T5-large as the RAG² admission filter.

Trains on labels from ``build_labels.py`` (general-medical MedQA - never the
thesis's Alzheimer's questions, specification SS10.6). Adds
``[HELPFUL]``/``[NOT_HELPFUL]`` as special tokens and resizes the embedding
matrix, exactly as the released ``classifier/model/token_add.ipynb`` does
(specification SS10.3) - ``FlanT5RAG2Filter`` refuses a checkpoint that
skips this.

    python -m experiments.filter_training.train \\
        --labels experiments/filter_training/labels/medqa_filter_labels.json \\
        --output-dir checkpoints/rag2_filter \\
        --epochs 3

``--epochs``: the paper trains 40 (``FilterTrainingConfig``'s reference
value); what fits a free Colab session depends on the labelled-set size and
must be chosen and recorded, not defaulted - this script requires it
explicitly rather than guessing a number for you.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

from .config import LABEL_TOKENS, CheckpointRecord, ConfigError, FilterTrainingConfig
from .labeling import HELPFUL, NOT_HELPFUL


def load_labelled_examples(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def split_train_val(
    records: list[dict], *, val_fraction: float = 0.1, seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """A plain data split for training THIS classifier.

    Unrelated to, and must not be confused with, the thesis's own
    Alzheimer's validation/test split
    (``experiments/questions/splits.json``), which exists to fit
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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", required=True,
                    help="training-label JSON from build_labels.py")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    out_dir = Path(args.output_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"refusing to write into a non-empty directory: {out_dir}",
              file=sys.stderr)
        return 2

    config = FilterTrainingConfig(epochs=args.epochs, seed=args.seed)
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
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )
    except ImportError as exc:
        print(f"train.py requires torch, transformers, datasets: {exc}",
              file=sys.stderr)
        return 2

    records = load_labelled_examples(args.labels)
    if not records:
        print("no labelled examples to train on", file=sys.stderr)
        return 2

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

    model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model)
    model.resize_token_embeddings(len(tokenizer))

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

    def compute_metrics(eval_pred):
        """Fraction of eval examples whose first generated token equals
        the first label token. Single-token labels ([HELPFUL]/
        [NOT_HELPFUL] are each exactly one token, by construction above),
        so the first generated token is the whole prediction - this has
        not been verified end-to-end in this environment (no torch/GPU
        here); sanity-check it on a small run before trusting it at scale.
        """
        predictions, labels = eval_pred
        pred_ids = predictions[:, 0] if predictions.ndim > 1 else predictions
        label_ids = labels[:, 0] if labels.ndim > 1 else labels
        return {"accuracy": float(np.mean(pred_ids == label_ids))}

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_batch_size,
        per_device_eval_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.epochs,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        bf16=(config.mixed_precision == "bf16"),
        fp16=(config.mixed_precision == "fp16"),
        predict_with_generate=True,
        generation_max_length=4,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        seed=config.seed,
        report_to=[],
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()
    val_accuracy = float(eval_metrics.get("eval_accuracy", 0.0))
    print(f"validation accuracy: {val_accuracy:.4f}")

    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    label_dist: dict[str, int] = {}
    for r in records:
        label_dist[r["answer"]] = label_dist.get(r["answer"], 0) + 1

    record = CheckpointRecord(
        checkpoint_path=str(final_dir),
        base_model=config.base_model,
        n_training_examples=len(train_records),
        n_validation_examples=len(val_records),
        validation_accuracy=val_accuracy,
        epochs_run=config.epochs,
        label_distribution=label_dist,
        trained_on="MedQA (general-medical), never the thesis's Alzheimer's "
                    "questions - see experiments/filter_training/medqa_data.py",
        notes=f"deviations from paper: {config.deviations()}",
    )
    record.validate()
    (out_dir / "checkpoint_record.json").write_text(
        json.dumps(asdict(record), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"usable as baseline filter (accuracy > 0.5 floor): "
          f"{record.is_usable()}")
    print(f"checkpoint saved to {final_dir}")
    print(f"checkpoint record saved to {out_dir / 'checkpoint_record.json'}")
    if not record.is_usable():
        print(
            "WARNING: validation accuracy is at or below chance (0.5). "
            "This checkpoint has not learned the label function and must "
            "not be passed to --rag2-checkpoint as a thesis baseline "
            "(specification SS10.5 / FlanT5RAG2Filter's own contract).",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""The fine-tuning of the filter, as the authors run it, plus a memory test and an evaluation (docs/methodology.md, decision D22).

What the authors' ``run_classifier.py`` and launch script fix, and what is kept here:

* AdamW, learning rate 3e-5, weight decay 0, a linear schedule from the learning rate to zero over all steps with no warmup;
* 40 epochs, batches of 16 shuffled afresh each epoch, the last batch of an epoch smaller if the data do not divide;
* inputs of at most 512 tokens, an input that is longer cut into windows that overlap by 128 tokens, every window carrying the
  label of its example; padding to the longest of a batch, labels padded with -100;
* the target is the label token followed by the end token, with the ordinary sequence-to-sequence cross-entropy, dropout as the
  model has it; a gradient-accumulation count of 1 (a larger count is an open hardware measure, see the methodology).

Added, and not part of the method: the settings are bound to the output folder, a stopped run continues from ``last/`` (the model,
the optimizer and the scheduler, saved after every epoch), the order of the batches and the dropout of an epoch depend on the
seed and the epoch only so that a resumed run is the same run, the validation accuracy is recorded after every epoch in
``training_log.jsonl``, and ``best/`` holds the model with the highest validation accuracy (the earliest if two tie). The
paper "selected a few candidate models from the validation set" (p.13); this is the mechanical form of that choice.

torch and transformers are imported inside the functions that need them.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rationale_query.store import RationaleStore, write_atomic

from .filter_eval import accuracy_report, auc, helpful_scores
from .filter_model import DOC_STRIDE, MAX_ANSWER_LENGTH, MAX_SEQ_LENGTH, FilterModel, FilterModelError, label_token_ids, tokenize_windows
from .labels import TOKEN_HELPFUL, TOKEN_NOT_HELPFUL
from .training_data import TrainingDataError, read_training_file

LOG_FILE = "training_log.jsonl"
LAST, BEST, FINAL = "last", "best", "final"
IGNORE = -100  # the label of a padding position, ignored by the loss

#: The settings that decide what a training run produces. A folder whose recorded values differ is refused.
RELEVANT = (
    "settings", "train_sha256", "validation_sha256", "base_files_sha256", "torch", "transformers",
)


class TrainingError(RuntimeError):
    """The training cannot go on as asked."""


@dataclass(frozen=True)
class TrainSettings:
    """Every choice of a run. The defaults are the authors'; ``seed`` and ``eval_batch_size`` are not theirs."""

    model_dir: str
    train_file: str
    out_dir: str
    validation_file: str | None = None
    epochs: int = 40
    batch_size: int = 16
    learning_rate: float = 3e-5
    weight_decay: float = 0.0
    max_seq_length: int = MAX_SEQ_LENGTH
    doc_stride: int = DOC_STRIDE
    max_answer_length: int = MAX_ANSWER_LENGTH
    gradient_accumulation_steps: int = 1
    seed: int = 42
    eval_batch_size: int = 8
    save_every: int = 0

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size", "gradient_accumulation_steps", "eval_batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a whole number from 1, got {value!r}")
        for name in ("seed", "save_every"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a whole number from 0, got {value!r}")
        if not isinstance(self.learning_rate, float | int) or isinstance(self.learning_rate, bool) or not 0 < self.learning_rate < 1:
            raise ValueError(f"learning_rate must be above 0 and below 1, got {self.learning_rate!r}")
        if not isinstance(self.weight_decay, float | int) or isinstance(self.weight_decay, bool) or not 0 <= self.weight_decay < 1:
            raise ValueError(f"weight_decay must be from 0 and below 1, got {self.weight_decay!r}")
        if isinstance(self.max_answer_length, bool) or not isinstance(self.max_answer_length, int) or self.max_answer_length < 2:
            raise ValueError("max_answer_length must be at least 2 (the label token and the end token)")
        if isinstance(self.max_seq_length, bool) or not isinstance(self.max_seq_length, int) or self.max_seq_length < 8:
            raise ValueError("max_seq_length must be a whole number of at least 8")
        if isinstance(self.doc_stride, bool) or not isinstance(self.doc_stride, int) or not 0 <= self.doc_stride < self.max_seq_length - 2:
            raise ValueError("doc_stride must be a whole number below max_seq_length - 2")
        for name in ("model_dir", "train_file", "out_dir"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be non-empty text")


# ------------------------------------------------------------------------------------------------ pure arithmetic


def steps_per_epoch(features: int, batch_size: int, accumulation: int = 1) -> int:
    """Optimizer steps in one epoch: batches per epoch, divided by the accumulation count, rounded up (as in the authors' script)."""
    return math.ceil(math.ceil(features / batch_size) / accumulation)


def total_steps(features: int, batch_size: int, epochs: int, accumulation: int = 1) -> int:
    return epochs * steps_per_epoch(features, batch_size, accumulation)


def linear_factor(step: int, total: int) -> float:
    """The multiplier of the learning rate at an optimizer step: 1 at step 0, falling in a straight line to 0 at the last step."""
    return max(0.0, (total - step) / max(1, total))


def epoch_order(count: int, seed: int, epoch: int) -> list[int]:
    """The order of the features in one epoch. It depends on the seed and the epoch number only, so a resumed run repeats it."""
    order = list(range(count))
    random.Random(f"{seed}:{epoch}").shuffle(order)
    return order


def batches(order: Sequence[int], size: int) -> list[list[int]]:
    return [list(order[start : start + size]) for start in range(0, len(order), size)]


def epoch_seed(seed: int, epoch: int) -> int:
    """A 31-bit number for torch's generator in one epoch (dropout), from the seed and the epoch."""
    return int.from_bytes(hashlib.sha256(f"{seed}:{epoch}:torch".encode()).digest()[:4], "big") % 2**31


# ------------------------------------------------------------------------------------------------ examples and features


@dataclass(frozen=True)
class Feature:
    """One training window: token ids of the input, the ids of the target, and the example it came from."""

    input_ids: list[int]
    labels: list[int]
    example: int


def read_examples(path: str | Path) -> list[dict[str, str]]:
    """The records of a training or validation file, checked: the four fields, a label token, a non-empty question, unique ids."""
    records = read_training_file(path)
    seen: set[str] = set()
    for position, record in enumerate(records):
        if not isinstance(record, dict) or not all(isinstance(record.get(name), str) and record[name] for name in ("id", "answer", "question")):
            raise TrainingDataError(f"{path}, record {position}: needs the text fields id, answer and question")
        if record["answer"] not in (TOKEN_HELPFUL, TOKEN_NOT_HELPFUL):
            raise TrainingDataError(f"{path}, record {position}: the answer must be {TOKEN_HELPFUL} or {TOKEN_NOT_HELPFUL}, got {record['answer']!r}")
        if record["id"] in seen:
            raise TrainingDataError(f"{path}: the id {record['id']} appears twice")
        seen.add(record["id"])
    if not records:
        raise TrainingDataError(f"{path} holds no examples")
    return records


def build_features(tokenizer: Any, examples: Sequence[Mapping[str, str]], max_seq_length: int, doc_stride: int, max_answer_length: int) -> list[Feature]:
    """Windows of every example, each with the target of its example (the label token and the end token)."""
    helpful, not_helpful = label_token_ids(tokenizer)
    windows, owners = tokenize_windows(tokenizer, [example["question"] for example in examples], max_seq_length, doc_stride)
    targets = tokenizer(text_target=[example["answer"] for example in examples], max_length=max_answer_length, truncation=True)["input_ids"]
    expected = {TOKEN_HELPFUL: helpful, TOKEN_NOT_HELPFUL: not_helpful}
    for example, target in zip(examples, targets, strict=True):
        if list(target) != [expected[example["answer"]], tokenizer.eos_token_id]:
            raise FilterModelError(f"the target of {example['id']} is {list(target)}, expected the label token and the end token")
    return [Feature(ids, list(targets[owner]), owner) for ids, owner in zip(windows, owners, strict=True)]


def collate(features: Sequence[Feature], pad_id: int) -> dict[str, Any]:
    """Padded tensors for a batch: inputs padded to the longest, labels padded with -100 (ignored by the loss)."""
    import torch

    width = max(len(f.input_ids) for f in features)
    label_width = max(len(f.labels) for f in features)
    ids = torch.full((len(features), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(features), width), dtype=torch.long)
    labels = torch.full((len(features), label_width), IGNORE, dtype=torch.long)
    for row, feature in enumerate(features):
        ids[row, : len(feature.input_ids)] = torch.tensor(feature.input_ids)
        mask[row, : len(feature.input_ids)] = 1
        labels[row, : len(feature.labels)] = torch.tensor(feature.labels)
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


# ------------------------------------------------------------------------------------------------ files


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def base_files_sha256(model_dir: str | Path) -> dict[str, str]:
    """The hash of every weight and tokenizer file of the prepared base model, so a changed base is noticed."""
    folder = Path(model_dir)
    names = sorted(p.name for p in folder.iterdir() if p.is_file() and p.suffix in (".safetensors", ".bin", ".json", ".model", ".txt"))
    return {name: file_sha256(folder / name) for name in names}


def _swap_in(temporary: Path, final: Path) -> None:
    """Replace ``final`` by ``temporary`` so that a crash leaves the old folder or the new one, never half of either."""
    old = final.with_name(final.name + ".old")
    if old.exists():
        shutil.rmtree(old)
    if final.exists():
        os.replace(final, old)
    os.replace(temporary, final)
    if old.exists():
        shutil.rmtree(old)


def recover(out_dir: str | Path) -> None:
    """After a crash between the two renames of ``_swap_in``, put the old ``last`` folder back."""
    for name in (LAST, BEST):
        final = Path(out_dir) / name
        old = final.with_name(final.name + ".old")
        if old.exists() and not final.exists():
            os.replace(old, final)
        temporary = final.with_name(final.name + ".tmp")
        if temporary.exists():
            shutil.rmtree(temporary)


def _save(folder: Path, model: Any, tokenizer: Any, extra: Callable[[Path], None] | None = None) -> None:
    temporary = folder.with_name(folder.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    model.save_pretrained(temporary)
    tokenizer.save_pretrained(temporary)
    if extra is not None:
        extra(temporary)
    _swap_in(temporary, folder)


# ------------------------------------------------------------------------------------------------ evaluation


def evaluate_examples(filter_model: FilterModel, examples: Sequence[Mapping[str, str]], *, batch_size: int = 8) -> dict[str, Any]:
    """Predictions and the report for labelled examples: accuracy overall and per class, the majority rate, the AUC if both classes occur."""
    logits = filter_model.logits([example["question"] for example in examples], batch_size=batch_size)
    by_id = {example["id"]: pair for example, pair in zip(examples, logits, strict=True)}
    gold = {example["id"]: example["answer"] for example in examples}
    helpful = {key: pair[0] >= pair[1] for key, pair in by_id.items()}
    predictions = {key: TOKEN_HELPFUL if flag else TOKEN_NOT_HELPFUL for key, flag in helpful.items()}
    report: dict[str, Any] = accuracy_report(predictions, gold)
    classes = set(gold.values())
    report["auc"] = round(auc(helpful_scores(by_id), gold), 4) if len(classes) == 2 else None
    return {
        "report": report,
        "predictions": {example["id"]: {"prediction": predictions[example["id"]], "answer": example["answer"], "dataset_name": example.get("dataset_name")} for example in examples},
    }


# ------------------------------------------------------------------------------------------------ training


def _device(requested: str | None) -> str:
    import torch

    return requested or ("cuda" if torch.cuda.is_available() else "cpu")


def _config(settings: TrainSettings) -> dict[str, Any]:
    import torch
    import transformers

    values = asdict(settings)
    for name in ("out_dir",):
        values.pop(name)
    return {
        "settings": values,
        "train_sha256": file_sha256(settings.train_file),
        "validation_sha256": file_sha256(settings.validation_file) if settings.validation_file else None,
        "base_files_sha256": base_files_sha256(settings.model_dir),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def _optimizer(model: Any, settings: TrainSettings) -> Any:
    import torch

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")  # as in the authors' script
    decay = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
    plain = [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": settings.weight_decay}, {"params": plain, "weight_decay": 0.0}], lr=settings.learning_rate
    )


def train(
    settings: TrainSettings, *, device: str | None = None, progress: Callable[[dict[str, Any]], None] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Fine-tune the prepared model in ``settings.model_dir``; returns a summary. A stopped run continues when called again."""
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    out = Path(settings.out_dir)
    recover(out)
    store = RationaleStore(out)
    store.bind(_config(settings), RELEVANT)
    device = _device(device)
    resume_from = out / LAST if (out / LAST / "state.pt").is_file() else None
    source = resume_from or Path(settings.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(str(source), local_files_only=True).to(device)
    label_token_ids(tokenizer)
    examples = read_examples(settings.train_file)
    features = build_features(tokenizer, examples, settings.max_seq_length, settings.doc_stride, settings.max_answer_length)
    validation = read_examples(settings.validation_file) if settings.validation_file else None
    per_epoch = steps_per_epoch(len(features), settings.batch_size, settings.gradient_accumulation_steps)
    total = total_steps(len(features), settings.batch_size, settings.epochs, settings.gradient_accumulation_steps)
    optimizer = _optimizer(model, settings)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: linear_factor(step, total))
    start_epoch, best = 0, None
    if resume_from is not None:
        state = torch.load(resume_from / "state.pt", map_location=device, weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, best = state["epoch"] + 1, state.get("best")
    log_path = out / LOG_FILE
    started = clock()
    pad = tokenizer.pad_token_id
    for epoch in range(start_epoch, settings.epochs):
        torch.manual_seed(epoch_seed(settings.seed, epoch))
        model.train()
        order = batches(epoch_order(len(features), settings.seed, epoch), settings.batch_size)
        epoch_loss, pending, begun = 0.0, 0, clock()
        for number, indices in enumerate(order, start=1):
            batch = {key: value.to(device) for key, value in collate([features[i] for i in indices], pad).items()}
            loss = model(**batch).loss
            (loss / settings.gradient_accumulation_steps).backward()
            epoch_loss += float(loss.detach().cpu())
            pending += 1
            if pending == settings.gradient_accumulation_steps or number == len(order):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                pending = 0
            if progress is not None:
                progress({"event": "step", "epoch": epoch + 1, "batch": number, "batches": len(order), "loss": float(loss.detach().cpu())})
        entry: dict[str, Any] = {
            "epoch": epoch + 1, "mean_loss": round(epoch_loss / len(order), 6), "seconds": round(clock() - begun, 2),
            "learning_rate": scheduler.get_last_lr()[0], "windows": len(features),
        }
        if validation is not None:
            result = evaluate_examples(FilterModel(model, tokenizer, device, max_seq_length=settings.max_seq_length, doc_stride=settings.doc_stride), validation, batch_size=settings.eval_batch_size)
            model.train()
            entry["validation"] = {key: result["report"][key] for key in ("n", "accuracy", "accuracy_per_class", "majority_rate", "auc")}
            accuracy = result["report"]["accuracy"]
            if best is None or accuracy > best["accuracy"]:
                best = {"epoch": epoch + 1, "accuracy": accuracy}
                _save(out / BEST, model, tokenizer, lambda folder: (folder / "metrics.json").write_text(json.dumps({**best, **entry}, indent=2) + "\n", encoding="utf-8"))
        if settings.save_every and (epoch + 1) % settings.save_every == 0:
            _save(out / f"epoch_{epoch + 1}", model, tokenizer)
        _save(out / LAST, model, tokenizer, lambda folder: torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "epoch": epoch, "best": best}, folder / "state.pt"))
        with open(log_path, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if progress is not None:
            progress({"event": "epoch", **entry})
    _save(out / FINAL, model, tokenizer)
    return {
        "epochs": settings.epochs, "windows": len(features), "examples": len(examples), "steps_per_epoch": per_epoch, "total_steps": total,
        "seconds": round(clock() - started, 2), "best": best, "device": device, "resumed_from_epoch": start_epoch if resume_from else None,
    }


def memory_test(settings: TrainSettings, *, device: str | None = None, clock: Callable[[], float] = time.perf_counter) -> dict[str, Any]:
    """One optimizer step on the longest ``batch_size`` windows of the training file, to see whether a run fits and how long a step takes.

    Nothing is saved. The longest windows are used because they need the most memory; the time per step is then an upper bound for
    most steps, and the estimate of an epoch is rounded up accordingly. An out-of-memory error is the result, not a crash.
    """
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = _device(device)
    tokenizer = AutoTokenizer.from_pretrained(settings.model_dir, local_files_only=True)
    examples = read_examples(settings.train_file)
    features = build_features(tokenizer, examples, settings.max_seq_length, settings.doc_stride, settings.max_answer_length)
    worst = sorted(features, key=lambda f: len(f.input_ids), reverse=True)[: settings.batch_size]
    result: dict[str, Any] = {
        "device": device, "batch_size": len(worst), "longest_input_tokens": len(worst[0].input_ids), "windows": len(features),
        "steps_per_epoch": steps_per_epoch(len(features), settings.batch_size, settings.gradient_accumulation_steps),
        "gradient_accumulation_steps": settings.gradient_accumulation_steps,
    }
    if len(worst) < settings.batch_size:
        result["note"] = f"the file has only {len(worst)} windows, fewer than the batch size {settings.batch_size}"
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(settings.model_dir, local_files_only=True).to(device)
        optimizer = _optimizer(model, settings)
        model.train()
        batch = {key: value.to(device) for key, value in collate(worst, tokenizer.pad_token_id).items()}
        begun = clock()
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        seconds = clock() - begun
    except torch.cuda.OutOfMemoryError as exc:
        result.update({"fits": False, "error": f"out of memory: {str(exc).splitlines()[0]}", "peak_gpu_bytes": torch.cuda.max_memory_allocated()})
        return result
    result.update(
        {
            "fits": True, "seconds_per_step": round(seconds, 4), "loss": round(float(loss.detach().cpu()), 4),
            "estimated_seconds_per_epoch": round(seconds * math.ceil(len(features) / settings.batch_size), 3),
            "parameters": sum(p.numel() for p in model.parameters()),
        }
    )
    if device.startswith("cuda"):
        result["peak_gpu_bytes"] = torch.cuda.max_memory_allocated()
        result["total_gpu_bytes"] = torch.cuda.get_device_properties(device).total_memory
    else:
        try:
            import resource

            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            result["peak_process_bytes"] = peak * 1024 if os.uname().sysname == "Linux" else peak
        except (ImportError, AttributeError):
            result["peak_process_bytes"] = None
    return result


def write_report(path: str | Path, report: Mapping[str, Any]) -> Path:
    """Write a JSON report; refuses to replace a file."""
    target = Path(path)
    if target.exists():
        raise TrainingError(f"refusing to overwrite {target}")
    write_atomic(target, json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return target


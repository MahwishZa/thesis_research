"""The filter model: Flan-T5 with the two label tokens, and what it says about a snippet (docs/methodology.md, section 5).

The authors add the tokens ``[HELPFUL]`` and ``[NOT_HELPFUL]`` to a Flan-T5 tokenizer, resize the embedding, and fine-tune the
model to produce one of them (``classifier/model/token_add.ipynb``; the notebook loads ``AutoModelForCausalLM``, which does not
fit a T5, so ``AutoModelForSeq2SeqLM`` is used as in their training script). At prediction time they take the two logits of the
label tokens at the first generated position and the larger wins (``run_classifier.py``). The first decoder step needs only the
encoder and one decoder position, so this module runs exactly that step and not a whole generation; the tests check that the
numbers equal ``generate(...).scores[0]``.

An input longer than the encoder's 512 tokens is cut into windows of 512 tokens that overlap by 128, as in training, where every
window carries the label of its example. The authors' evaluation never says how the windows of one input are combined (their
check that predictions and examples agree in number would stop at the first overflowing input). **Working default:** the two
logits are averaged over the windows of an input and the decision is taken on the averages. The share of such inputs is reported.

numpy-free; torch and transformers are imported only where they are used, so importing this module is light.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .inference import decide_helpful
from .labels import TOKEN_HELPFUL, TOKEN_NOT_HELPFUL

LABEL_TOKENS = (TOKEN_HELPFUL, TOKEN_NOT_HELPFUL)
MAX_SEQ_LENGTH = 512  # the launch script of the authors
DOC_STRIDE = 128  # likewise
MAX_ANSWER_LENGTH = 30  # the authors' default; "[HELPFUL]" and the end token need two


class FilterModelError(RuntimeError):
    """The model or the tokenizer cannot be used as a filter."""


def effective_length(tokenizer: Any, max_seq_length: int = MAX_SEQ_LENGTH) -> int:
    """The window length: ``max_seq_length``, or the tokenizer's own limit if that is smaller (as in the authors' script)."""
    if isinstance(max_seq_length, bool) or not isinstance(max_seq_length, int) or max_seq_length < 8:
        raise ValueError(f"max_seq_length must be a whole number of at least 8, got {max_seq_length!r}")
    limit = getattr(tokenizer, "model_max_length", None)
    return min(max_seq_length, limit) if isinstance(limit, int) and 0 < limit < 10**9 else max_seq_length


def check_stride(max_length: int, doc_stride: int) -> None:
    if isinstance(doc_stride, bool) or not isinstance(doc_stride, int) or not 0 <= doc_stride < max_length - 2:
        raise ValueError(f"doc_stride must be a whole number from 0 to {max_length - 3}, got {doc_stride!r}")


def tokenize_windows(
    tokenizer: Any, texts: Sequence[str], max_seq_length: int = MAX_SEQ_LENGTH, doc_stride: int = DOC_STRIDE
) -> tuple[list[list[int]], list[int]]:
    """Token ids of every window of every text, and the index of the text each window belongs to.

    A text that fits is one window. A longer one gives windows of ``max_seq_length`` tokens (the end token included) that
    overlap by ``doc_stride`` tokens; the text is stripped first, as in the authors' preprocessing.
    """
    if not texts:
        return [], []
    for position, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"text {position} is empty or not text")
    length = effective_length(tokenizer, max_seq_length)
    check_stride(length, doc_stride)
    encoded = tokenizer([text.strip() for text in texts], truncation=True, max_length=length, stride=doc_stride, return_overflowing_tokens=True)
    return [list(ids) for ids in encoded["input_ids"]], list(encoded["overflow_to_sample_mapping"])


def mean_over_windows(rows: Sequence[tuple[float, float]], owners: Sequence[int], count: int) -> list[tuple[float, float]]:
    """One pair of logits per text: the mean over the windows that belong to it. Every text must have at least one window."""
    sums = [[0.0, 0.0] for _ in range(count)]
    seen = [0] * count
    for (first, second), owner in zip(rows, owners, strict=True):
        sums[owner][0] += first
        sums[owner][1] += second
        seen[owner] += 1
    if 0 in seen:
        raise FilterModelError(f"text {seen.index(0)} has no window")
    return [(a / n, b / n) for (a, b), n in zip(sums, seen, strict=True)]


def label_token_ids(tokenizer: Any) -> tuple[int, int]:
    """The ids of the two label tokens. Both must be known to the tokenizer and different from the unknown token."""
    ids = []
    for token in LABEL_TOKENS:
        number = tokenizer.convert_tokens_to_ids(token)
        if number is None or number == getattr(tokenizer, "unk_token_id", None) or number < 0:
            raise FilterModelError(f"the tokenizer does not know {token}; run prepare-model on the base model first")
        ids.append(int(number))
    if ids[0] == ids[1]:
        raise FilterModelError("the two label tokens have the same id")
    return ids[0], ids[1]


def prepare_model(source: str | Path, out_dir: str | Path, *, local_files_only: bool = False) -> dict[str, Any]:
    """Add the two label tokens to a base model and its tokenizer and save both into a new folder.

    ``source`` is a model name of the hub (``google/flan-t5-base``) or a local folder. The embedding is resized to the length of
    the tokenizer, as in the authors' notebook. A folder that is not empty is refused, and so is a base model that already knows
    one of the tokens.
    """
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    target = Path(out_dir)
    if target.exists() and any(target.iterdir()):
        raise FilterModelError(f"{target} is not empty; choose a new folder")
    tokenizer = AutoTokenizer.from_pretrained(str(source), local_files_only=local_files_only)
    model = AutoModelForSeq2SeqLM.from_pretrained(str(source), local_files_only=local_files_only)
    known = [token for token in LABEL_TOKENS if tokenizer.convert_tokens_to_ids(token) not in (None, tokenizer.unk_token_id)]
    if known:
        raise FilterModelError(f"{source} already knows {', '.join(known)}; it is not a base model")
    before = len(tokenizer)
    rows_before = model.get_input_embeddings().weight.shape[0]
    added = tokenizer.add_tokens(list(LABEL_TOKENS))
    if added != len(LABEL_TOKENS):
        raise FilterModelError(f"{added} tokens were added instead of {len(LABEL_TOKENS)}")
    model.resize_token_embeddings(len(tokenizer))
    ids = label_token_ids(tokenizer)
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target)
    tokenizer.save_pretrained(target)
    return {
        "source": str(source),
        "folder": str(target),
        "model_class": type(model).__name__,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "tokenizer_length": [before, len(tokenizer)],
        "embedding_rows": [rows_before, model.get_input_embeddings().weight.shape[0]],
        "label_token_ids": {TOKEN_HELPFUL: ids[0], TOKEN_NOT_HELPFUL: ids[1]},
    }


class FilterModel:
    """A prepared (and usually fine-tuned) Flan-T5 that says, for filter inputs, how much it favours each label token."""

    def __init__(self, model: Any, tokenizer: Any, device: str, *, max_seq_length: int = MAX_SEQ_LENGTH, doc_stride: int = DOC_STRIDE) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_seq_length = max_seq_length
        self.doc_stride = doc_stride
        self.helpful_id, self.not_helpful_id = label_token_ids(tokenizer)
        rows = model.get_input_embeddings().weight.shape[0]
        if max(self.helpful_id, self.not_helpful_id) >= rows:
            raise FilterModelError("a label token has no row in the model's embedding; the model was not prepared with these tokens")
        start = getattr(model.config, "decoder_start_token_id", None)
        if start is None:
            raise FilterModelError("the model has no decoder_start_token_id")
        self.decoder_start = int(start)
        self.windowed_inputs = 0  # inputs that needed more than one window, since this object was made
        self.inputs_seen = 0
        self.model.eval()

    @classmethod
    def load(
        cls, path: str | Path, *, device: str | None = None, max_seq_length: int = MAX_SEQ_LENGTH, doc_stride: int = DOC_STRIDE,
        local_files_only: bool = True,
    ) -> FilterModel:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=local_files_only)
        model = AutoModelForSeq2SeqLM.from_pretrained(str(path), local_files_only=local_files_only)
        chosen = device or ("cuda" if torch.cuda.is_available() else "cpu")
        return cls(model.to(chosen), tokenizer, chosen, max_seq_length=max_seq_length, doc_stride=doc_stride)

    def logits(self, texts: Sequence[str], *, batch_size: int = 8) -> list[tuple[float, float]]:
        """``(logit of [HELPFUL], logit of [NOT_HELPFUL])`` for each text, at the first decoder position, averaged over windows."""
        import torch

        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a whole number from 1")
        windows, owners = tokenize_windows(self.tokenizer, texts, self.max_seq_length, self.doc_stride)
        if not windows:
            return []
        self.inputs_seen += len(texts)
        counts = [0] * len(texts)
        for owner in owners:
            counts[owner] += 1
        self.windowed_inputs += sum(1 for count in counts if count > 1)
        order = sorted(range(len(windows)), key=lambda index: len(windows[index]))
        rows: list[tuple[float, float] | None] = [None] * len(windows)
        pad = self.tokenizer.pad_token_id
        with torch.no_grad():
            for start in range(0, len(order), batch_size):
                chunk = order[start : start + batch_size]
                width = max(len(windows[i]) for i in chunk)
                ids = torch.full((len(chunk), width), pad, dtype=torch.long)
                mask = torch.zeros((len(chunk), width), dtype=torch.long)
                for row, index in enumerate(chunk):
                    ids[row, : len(windows[index])] = torch.tensor(windows[index])
                    mask[row, : len(windows[index])] = 1
                decoder = torch.full((len(chunk), 1), self.decoder_start, dtype=torch.long)
                out = self.model(input_ids=ids.to(self.device), attention_mask=mask.to(self.device), decoder_input_ids=decoder.to(self.device))
                step = out.logits[:, 0, :].float().cpu()
                for row, index in enumerate(chunk):
                    rows[index] = (float(step[row, self.helpful_id]), float(step[row, self.not_helpful_id]))
        finished = [row for row in rows if row is not None]
        if len(finished) != len(rows):
            raise FilterModelError("a window was not scored")
        return mean_over_windows(finished, owners, len(texts))

    def judge_many(self, texts: Sequence[str], *, batch_size: int = 8) -> list[bool]:
        """Helpful or not for each filter input; an exact tie counts as helpful (the authors' ``argmax`` picks index 0)."""
        return [decide_helpful(logit_helpful=a, logit_not_helpful=b) for a, b in self.logits(texts, batch_size=batch_size)]

    def judge(self, text: str) -> bool:
        """One filter input (for ``rationale_filter.inference.judge_snippets``)."""
        return self.judge_many([text], batch_size=1)[0]

    def windowed_share(self) -> float | None:
        """The share of the inputs seen so far that needed more than one window; None before the first input."""
        return round(self.windowed_inputs / self.inputs_seen, 6) if self.inputs_seen else None

    def info(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "parameters": sum(parameter.numel() for parameter in self.model.parameters()),
            "max_seq_length": effective_length(self.tokenizer, self.max_seq_length),
            "doc_stride": self.doc_stride,
            "label_token_ids": {TOKEN_HELPFUL: self.helpful_id, TOKEN_NOT_HELPFUL: self.not_helpful_id},
            "overflow_rule": "the logits are averaged over the windows of one input",
        }


"""RAG² evidence-admission filter.

The RAG² filtering stage classifies each (question, evidence) pair as
``[HELPFUL]`` or ``[NOT_HELPFUL]`` with a Flan-T5 sequence-to-sequence model.

Fidelity notes, verified against the official release
(https://github.com/dmis-lab/RAG2):

* **Prompt template.** Taken verbatim from the released training artifact
  ``classifier/data/medqa/llama3_cot/5%-train.json``:

      Given the following evidence, determine whether it helps answer the
      provided question.

      Evidence: {evidence}

      Question: {question}

  A filter trained on this template is sensitive to deviation from it, so the
  wording is reproduced exactly rather than paraphrased.

* **The question carries its answer options.** In the released data the
  question field is the full item text with its options appended
  (``... A) ... B) ... C) ... D) ...``). Callers must pass the question in
  the same form the filter was trained on; this module does not synthesise
  options.

* **Decision rule.** ``run_classifier.py`` takes a two-way softmax over the
  ``[HELPFUL]`` / ``[NOT_HELPFUL]`` label-token logits at the first decoder
  position and applies argmax. That is reproduced here.

* **Label tokens.** The official setup ADDS both labels as new tokens
  (``classifier/model/token_add.ipynb``) and resizes the embedding matrix, so
  each maps to exactly one id. A stock Flan-T5 checkpoint will not satisfy
  this, and the constructor says so rather than silently scoring multi-token
  labels.

The original trained checkpoint is not distributed and is never fabricated
here: the caller supplies a path explicitly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from ..interfaces.evidence import Candidate


HELPFUL = "[HELPFUL]"
NOT_HELPFUL = "[NOT_HELPFUL]"

#: Verbatim from the official RAG² training artifact. Do not paraphrase.
RAG2_FILTER_TEMPLATE = (
    "Given the following evidence, determine whether it helps answer "
    "the provided question.\n\n"
    "Evidence: {evidence}\n\n"
    "Question: {question}"
)


@dataclass(frozen=True)
class AdmissionDecision:
    """Decision for one question/evidence pair."""

    evidence_id: str
    label: str
    probability_helpful: Optional[float] = None

    #: True when the encoder input hit the length limit. Silent truncation
    #: changes what the filter saw, so it is recorded rather than hidden.
    input_truncated: bool = False

    @property
    def helpful(self) -> bool:
        return self.label == HELPFUL


class AdmissionFilter(ABC):
    """Interface for evidence-admission filters."""

    @abstractmethod
    def predict(
        self,
        question: str,
        candidate: Candidate,
    ) -> AdmissionDecision:
        raise NotImplementedError


class MockRAG2Filter(AdmissionFilter):
    """Deterministic development/test double.

    This class must never be used to produce thesis performance results.
    """

    def __init__(self, labels: dict[str, str]) -> None:

        invalid = set(labels.values()) - {HELPFUL, NOT_HELPFUL}

        if invalid:
            raise ValueError(f"Invalid labels: {sorted(invalid)}")

        self._labels = dict(labels)

    def predict(
        self,
        question: str,
        candidate: Candidate,
    ) -> AdmissionDecision:

        label = self._labels.get(
            candidate.evidence.evidence_id,
            NOT_HELPFUL,
        )

        return AdmissionDecision(
            evidence_id=candidate.evidence.evidence_id,
            label=label,
        )


class FlanT5RAG2Filter(AdmissionFilter):
    """Adapter for an explicitly supplied RAG²-compatible checkpoint."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: Optional[str] = None,
        max_input_length: int = 512,
        template: str = RAG2_FILTER_TEMPLATE,
    ) -> None:

        try:
            import torch
            from transformers import (
                AutoModelForSeq2SeqLM,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise ImportError(
                "FlanT5RAG2Filter requires torch and transformers."
            ) from exc

        self._torch = torch
        self._template = template

        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path
        )

        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path
        )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif "cuda" in device and not torch.cuda.is_available():
            # A clear, immediate error instead of torch's own
            # "AssertionError: Torch not compiled with CUDA enabled" eight
            # frames deep in Module._apply - hit for real (2026-09-21) after
            # a pip install silently replaced a Colab/Kaggle notebook's
            # preinstalled CUDA-enabled torch with a CPU-only build.
            raise ImportError(
                f"requested device={device!r}, but torch.cuda.is_available() "
                f"is False (torch {torch.__version__}). This usually means "
                "a `pip install` after the notebook started replaced the "
                "platform's preinstalled CUDA-enabled torch with a CPU-only "
                "build - check `torch.__version__` for a '+cpu' suffix. Fix: "
                "reinstall torch from the CUDA wheel index matching this "
                "machine's CUDA version (check `!nvidia-smi`), e.g. `pip "
                "install --index-url https://download.pytorch.org/whl/cu121 "
                "torch --force-reinstall`, then verify with "
                "`torch.cuda.is_available()` before retrying."
            )

        self._device = torch.device(device)

        self._model.to(self._device)
        self._model.eval()

        self._max_input_length = max_input_length

        helpful_ids = self._tokenizer.encode(
            HELPFUL,
            add_special_tokens=False,
        )

        not_helpful_ids = self._tokenizer.encode(
            NOT_HELPFUL,
            add_special_tokens=False,
        )

        for name, ids in (
            (HELPFUL, helpful_ids),
            (NOT_HELPFUL, not_helpful_ids),
        ):
            if len(ids) != 1:
                raise ValueError(
                    f"{name!r} maps to {len(ids)} tokens, expected 1. "
                    "The RAG² filter checkpoint must have both label "
                    "tokens added to its tokenizer and its embedding "
                    "matrix resized (see classifier/model/token_add.ipynb "
                    "in the official release). A stock Flan-T5 checkpoint "
                    "will not satisfy this."
                )

        self._helpful_id = helpful_ids[0]
        self._not_helpful_id = not_helpful_ids[0]

    def format_input(self, question: str, evidence_text: str) -> str:
        """Build the classifier input from the official template."""
        return self._template.format(
            evidence=evidence_text,
            question=question,
        )

    def predict(
        self,
        question: str,
        candidate: Candidate,
    ) -> AdmissionDecision:

        text = self.format_input(question, candidate.evidence.text)

        untruncated = self._tokenizer(
            text,
            return_tensors=None,
            truncation=False,
        )["input_ids"]

        truncated = len(untruncated) > self._max_input_length

        encoded = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_input_length,
        )

        encoded = {
            key: value.to(self._device)
            for key, value in encoded.items()
        }

        decoder_start_id = self._model.config.decoder_start_token_id

        if decoder_start_id is None:
            decoder_start_id = self._model.config.pad_token_id

        decoder_input_ids = self._torch.tensor(
            [[decoder_start_id]],
            device=self._device,
        )

        with self._torch.no_grad():
            outputs = self._model(
                **encoded,
                decoder_input_ids=decoder_input_ids,
            )

        # Two-way softmax over the label-token logits at the first decoder
        # position, then argmax - the official decision rule.
        logits = outputs.logits[:, 0, :]

        label_logits = logits[
            :,
            [self._helpful_id, self._not_helpful_id],
        ]

        probabilities = self._torch.softmax(label_logits, dim=-1)[0]

        helpful_probability = float(probabilities[0].item())
        not_helpful_probability = float(probabilities[1].item())

        label = (
            HELPFUL
            if helpful_probability >= not_helpful_probability
            else NOT_HELPFUL
        )

        return AdmissionDecision(
            evidence_id=candidate.evidence.evidence_id,
            label=label,
            probability_helpful=helpful_probability,
            input_truncated=truncated,
        )

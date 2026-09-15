"""RAG² evidence-admission filter.

The original RAG² filtering stage uses a Flan-T5 sequence-to-sequence
model to classify each question/evidence pair as [HELPFUL] or
[NOT_HELPFUL].

The original trained checkpoint is not assumed to be available.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from ..interfaces.evidence import Candidate


HELPFUL = "[HELPFUL]"
NOT_HELPFUL = "[NOT_HELPFUL]"


@dataclass(frozen=True)
class AdmissionDecision:
    """Decision for one question/evidence pair."""

    evidence_id: str
    label: str
    probability_helpful: Optional[float] = None

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

    def __init__(
        self,
        labels: dict[str, str],
    ) -> None:

        invalid = set(labels.values()) - {
            HELPFUL,
            NOT_HELPFUL,
        }

        if invalid:
            raise ValueError(
                f"Invalid labels: {sorted(invalid)}"
            )

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
    """Adapter for an explicitly supplied RAG²-compatible checkpoint.

    No checkpoint is fabricated here. The caller must explicitly provide
    the model path/name.
    """

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: Optional[str] = None,
        max_input_length: int = 512,
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

        self._tokenizer = (
            AutoTokenizer.from_pretrained(
                model_name_or_path
            )
        )

        self._model = (
            AutoModelForSeq2SeqLM.from_pretrained(
                model_name_or_path
            )
        )

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
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

        if len(helpful_ids) != 1:
            raise ValueError(
                f"{HELPFUL!r} must map to one token."
            )

        if len(not_helpful_ids) != 1:
            raise ValueError(
                f"{NOT_HELPFUL!r} must map to one token."
            )

        self._helpful_id = helpful_ids[0]
        self._not_helpful_id = not_helpful_ids[0]

    @staticmethod
    def _format_input(
        question: str,
        evidence_text: str,
    ) -> str:
        """Build the classifier input.

        Keep this isolated because the exact training-format template must
        ultimately match the RAG² training data/reproduction procedure.
        """

        return (
            "Given the following evidence, determine whether "
            "it helps answer the question.\n\n"
            f"Evidence: {evidence_text}\n\n"
            f"Question: {question}"
        )

    def predict(
        self,
        question: str,
        candidate: Candidate,
    ) -> AdmissionDecision:

        text = self._format_input(
            question,
            candidate.evidence.text,
        )

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

        decoder_start_id = (
            self._model.config.decoder_start_token_id
        )

        if decoder_start_id is None:
            decoder_start_id = (
                self._model.config.pad_token_id
            )

        decoder_input_ids = self._torch.tensor(
            [[decoder_start_id]],
            device=self._device,
        )

        with self._torch.no_grad():
            outputs = self._model(
                **encoded,
                decoder_input_ids=decoder_input_ids,
            )

        logits = outputs.logits[:, 0, :]

        label_logits = logits[
            :,
            [
                self._helpful_id,
                self._not_helpful_id,
            ],
        ]

        probabilities = self._torch.softmax(
            label_logits,
            dim=-1,
        )[0]

        helpful_probability = float(
            probabilities[0].item()
        )

        not_helpful_probability = float(
            probabilities[1].item()
        )

        label = (
            HELPFUL
            if helpful_probability >= not_helpful_probability
            else NOT_HELPFUL
        )

        return AdmissionDecision(
            evidence_id=candidate.evidence.evidence_id,
            label=label,
            probability_helpful=helpful_probability,
        )
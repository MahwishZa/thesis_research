"""Training configuration for the RAG² filter, and what deviates from it.

Every value here is either the paper's (taken from the released launch script,
`_archive/docs_legacy/research_experimental_specification.md` §10.3) or a deviation with a
reason, §10.4. The
point of keeping them side by side in one object is that a deviation cannot be
made without it appearing in the training report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

#: The base checkpoint. The paper record says Flan-T5-large (≈770M, ledger
#: E1); the released launch script does not pin a size, so the paper is the
#: only authority and this follows it.
BASE_MODEL = "google/flan-t5-large"

#: Added as special tokens with the embedding matrix resized, exactly as the
#: released code does.
LABEL_TOKENS = ("[HELPFUL]", "[NOT_HELPFUL]")


class ConfigError(RuntimeError):
    """Raised when a training configuration would not be defensible."""


@dataclass(frozen=True)
class FilterTrainingConfig:
    """Hyperparameters, with the reference value beside each deviation."""

    base_model: str = BASE_MODEL
    learning_rate: float = 3e-5          # paper
    optimizer: str = "adamw"             # paper
    max_seq_length: int = 512            # paper
    doc_stride: int = 128                # paper
    weight_decay: float = 0.0            # paper (default)
    warmup_steps: int = 0                # paper (default)

    #: Paper: 16 per device on its own GPU. A free T4 at seq 512 on a 770M
    #: seq2seq model cannot hold that, so the *effective* batch is preserved
    #: through accumulation instead of quietly shrinking it.
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 4

    #: Paper: 40. Left unset deliberately - see `validate()`. The number that
    #: fits a free session depends on the labelled-set size, which depends on
    #: how much label generation the budget bought.
    epochs: Optional[int] = None

    mixed_precision: str = "bf16"
    seed: int = 42

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_batch_size * self.gradient_accumulation_steps

    def deviations(self) -> dict[str, str]:
        """What differs from the reference recipe, and why."""
        out: dict[str, str] = {}
        if self.per_device_batch_size != 16:
            out["per_device_batch_size"] = (
                f"{self.per_device_batch_size} vs paper 16; effective batch "
                f"{self.effective_batch_size} preserved via "
                f"{self.gradient_accumulation_steps} accumulation steps, "
                "because a free 16 GB T4 cannot hold batch 16 at seq 512"
            )
        if self.epochs is not None and self.epochs != 40:
            out["epochs"] = (
                f"{self.epochs} vs paper 40; reduced to fit free-tier session "
                "limits. The count actually run is reported."
            )
        if self.base_model != BASE_MODEL:
            out["base_model"] = f"{self.base_model} vs paper {BASE_MODEL}"
        return out

    def validate(self) -> None:
        """Refuse a configuration that cannot be reported honestly."""
        if self.epochs is None:
            raise ConfigError(
                "epochs is unset. The paper trains 40; what fits a free "
                "session depends on the labelled-set size, so this must be "
                "chosen and recorded rather than defaulted."
            )
        if self.epochs <= 0:
            raise ConfigError("epochs must be positive")
        if self.effective_batch_size != 16:
            raise ConfigError(
                f"effective batch size is {self.effective_batch_size}, not "
                "the paper's 16. Accumulation exists to preserve it; changing "
                "it is a deviation that needs its own justification."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_model": self.base_model,
            "learning_rate": self.learning_rate,
            "optimizer": self.optimizer,
            "max_seq_length": self.max_seq_length,
            "doc_stride": self.doc_stride,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "per_device_batch_size": self.per_device_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "effective_batch_size": self.effective_batch_size,
            "epochs": self.epochs,
            "mixed_precision": self.mixed_precision,
            "seed": self.seed,
            "label_tokens": list(LABEL_TOKENS),
            "deviations_from_paper": self.deviations(),
        }


@dataclass(frozen=True)
class CheckpointRecord:
    """What must be true before a checkpoint may produce thesis numbers.

    `FlanT5RAG2Filter` already refuses to run without an explicit checkpoint
    path. This records the rest: which data trained it, how well it did, and
    on what. A checkpoint without these is not usable in the thesis, because
    the baseline's strength could not be reported.
    """

    checkpoint_path: str
    base_model: str
    n_training_examples: int
    n_validation_examples: int
    validation_accuracy: float
    epochs_run: int
    label_distribution: dict[str, int] = field(default_factory=dict)
    trained_on: str = ""
    notes: str = ""

    def validate(self) -> None:
        if not 0.0 <= self.validation_accuracy <= 1.0:
            raise ConfigError("validation_accuracy must be a fraction in [0,1]")
        if self.n_training_examples <= 0:
            raise ConfigError("a checkpoint must record its training-set size")
        if self.epochs_run <= 0:
            raise ConfigError("a checkpoint must record epochs actually run")

    def is_usable(self, *, floor: float = 0.5) -> bool:
        """Whether this checkpoint can stand as the baseline filter.

        A binary classifier at or below chance has not learned the label
        function, and a baseline built on it would be a broken system rather
        than a weak one - which changes what a HAR difference means.
        """
        self.validate()
        return self.validation_accuracy > floor

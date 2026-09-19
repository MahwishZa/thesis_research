"""Training the RAG² evidence filter, because the checkpoint is not released.

The label function is the paper's (``labeling.py``); the hyperparameters are
the paper's except where a free-tier GPU forces a deviation, and each
deviation states itself (``config.py``). Nothing here loads a model: the
expensive half runs on the remote machine, and what lives in the repository is
the part that has to be correct and reproducible.

See ``docs/research_experimental_specification.md`` §10 for the strategy and
the cost, and ``docs/status_and_decisions.md`` for the status.
"""

from .config import (
    BASE_MODEL,
    LABEL_TOKENS,
    CheckpointRecord,
    ConfigError,
    FilterTrainingConfig,
)
from .labeling import (
    HELPFUL,
    NOT_HELPFUL,
    TAU,
    LabelledExample,
    LabelingError,
    PairOutcome,
    RationaleOutcome,
    label_dataset,
    label_distribution,
    label_pair,
    perplexity_threshold,
    write_training_file,
)

__all__ = [
    "BASE_MODEL", "HELPFUL", "LABEL_TOKENS", "NOT_HELPFUL", "TAU",
    "CheckpointRecord", "ConfigError", "FilterTrainingConfig",
    "LabelledExample", "LabelingError", "PairOutcome", "RationaleOutcome",
    "label_dataset", "label_distribution", "label_pair",
    "perplexity_threshold", "write_training_file",
]

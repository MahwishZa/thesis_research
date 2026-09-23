"""Training the RAG² evidence filter, because the checkpoint is not released.

The label function is the paper's (``labeling.py``); the hyperparameters are
the paper's except where a free-tier GPU forces a deviation, and each
deviation states itself (``config.py``). ``medqa_data.py`` and
``rationale.py`` do load a model/datasets, but only lazily, inside their
functions - importing this package still costs nothing on a machine without
torch/transformers/datasets installed, and the module-level imports below
stay safe as a result. The expensive half (``build_labels.py``,
``train.py``, both meant to run on Colab) is kept correct and testable at
the logic level here; what actually needs a GPU is exercised there, not in
this repository's unit-test suite.

See ``_archive/docs_legacy/research_experimental_specification.md`` §10 for the strategy and
the cost, and ``_archive/docs_legacy/status_and_decisions.md`` for the status.
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
from .medqa_data import (
    MEDQA_DATASET_ID,
    TEXTBOOK_DATASET_ID,
    MedQADataError,
    MedQAItem,
    extract_answer_letter,
    load_medqa,
    load_textbook_passages,
)

__all__ = [
    "BASE_MODEL", "HELPFUL", "LABEL_TOKENS", "MEDQA_DATASET_ID",
    "NOT_HELPFUL", "TAU", "TEXTBOOK_DATASET_ID",
    "CheckpointRecord", "ConfigError", "FilterTrainingConfig",
    "LabelledExample", "LabelingError", "MedQADataError", "MedQAItem",
    "PairOutcome", "RationaleOutcome",
    "extract_answer_letter", "label_dataset", "label_distribution",
    "label_pair", "load_medqa", "load_textbook_passages",
    "perplexity_threshold", "write_training_file",
]

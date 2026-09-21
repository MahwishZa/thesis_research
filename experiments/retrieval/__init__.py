"""Upstream retrieval and reranking — shared by every arm.

This stage runs once per question and its output is frozen. Both arms read the
frozen set; neither retrieves for itself. That is what makes the comparison a
comparison of admission rules.

Real execution needs the corpus (Step 1) and ``torch``/``transformers``. The
code, its configuration and its tests do not, and are complete now.
"""

from .corpus import (
    CorpusError,
    CorpusPassage,
    dated_only,
    parse_publication_date,
    read_passages,
    read_passages_with_snapshot,
    snapshot_id,
)
from .encoders import (
    ARTICLE_ENCODER,
    CROSS_ENCODER,
    QUERY_ENCODER,
    CrossEncoderReranker,
    Encoder,
    HashingEncoder,
    LexicalOverlapReranker,
    MedCPTEncoder,
    MedCPTReranker,
    medcpt_article_encoder,
    medcpt_query_encoder,
)
from .index import DenseIndex, Hit, IndexError_, build_index, l2_normalize
from .pipeline import (
    DEFAULT_CANDIDATE_COUNT,
    DEFAULT_RETRIEVAL_DEPTH,
    RetrievalConfig,
    RetrievalError,
    RetrievalPipeline,
    RetrievedSet,
)

__all__ = [
    "ARTICLE_ENCODER", "CROSS_ENCODER", "QUERY_ENCODER",
    "CorpusError", "CorpusPassage", "CrossEncoderReranker",
    "DEFAULT_CANDIDATE_COUNT", "DEFAULT_RETRIEVAL_DEPTH", "DenseIndex",
    "Encoder", "HashingEncoder", "Hit", "IndexError_",
    "LexicalOverlapReranker", "MedCPTEncoder", "MedCPTReranker",
    "RetrievalConfig", "RetrievalError", "RetrievalPipeline", "RetrievedSet",
    "build_index", "dated_only", "l2_normalize", "medcpt_article_encoder",
    "medcpt_query_encoder", "parse_publication_date", "read_passages",
    "read_passages_with_snapshot", "snapshot_id",
]

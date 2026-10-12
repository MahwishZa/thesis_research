"""The pipeline of the study, on the as-of benchmark built from MedChange (Vladika et al., EMNLP 2025 Findings): Cochrane
reviews whose verdict changed between versions, plus unchanged controls.

Benchmark and splits (``benchmark``, ``ad_benchmark``, ``fresh_benchmark``), as-of evidence and retrieval (``pubmed_asof``,
``freeze_candidates``, ``encoders``, ``abstracts``), the systems (``arms``, ``prompts``, ``generate_answers``, ``rag2``,
``rag2_run``, ``rag2_pipeline`` with ``pipeline_support``), the analyses and audits (``scoring``, ``stats``, ``analyze_rag2``,
``analyze_fresh``, ``class_balance``, ``report``, ``label_audit``, ``verdict_consistency``, ``ad_subgroup``), the evidence records and
recency formula of the earlier approach (``stage1``) and the committed ``results/``. See ``docs/protocol.md`` and
``docs/methodology.md``."""

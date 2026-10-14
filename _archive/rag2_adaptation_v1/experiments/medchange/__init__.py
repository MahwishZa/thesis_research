"""The pipeline of the study, on the as-of benchmark built from MedChange (Vladika et al., EMNLP 2025 Findings): Cochrane
reviews whose verdict changed between versions, plus unchanged controls.

A module that can be run with ``python -m`` is a *command*; every other module is a *library*, and no module imports a command
(``tests/test_repository_guards.py``, ``LayeringTests``).

Commands: the benchmark and its question sets (``build_benchmark``, ``ad_benchmark``, ``fresh_supply``, ``fresh_benchmark``,
``ad_cochrane_supply``), as-of evidence and candidate pools (``pubmed_asof``, ``freeze_candidates``), the systems (``generate_answers``,
``rag2_run``, ``rag2_pipeline``) and the analyses and audits (``analyze_rag2``, ``analyze_fresh``, ``class_balance``, ``report``,
``ad_subgroup``, ``ad_case_study``, ``label_audit``, ``verdict_consistency``).

Libraries: the release as as-of items, the questions no split uses and the release's closed-book answers (``benchmark``, ``selection``,
``released_answers``), PubMed access and retrieval models (``eutils``, ``abstracts``, ``encoders``), the systems (``prompts``, ``arms``,
``generator``, ``files``, ``rag2``, ``pipeline_support``), the measurements (``analysis``, ``scoring``, ``stats``) and the committed
``results/``. The earlier approaches (stages 1 and 2) are not part of the package: their results are in ``results/earlier_stages/`` and their code is
in Git history (``docs/methodology.md`` §10). See ``docs/protocol.md`` and ``docs/methodology.md``."""

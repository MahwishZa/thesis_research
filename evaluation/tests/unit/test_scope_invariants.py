"""Guards against silent research-scope drift.

The repository has already changed research question twice, and each change
left stale statements in several documents that took a dedicated audit to
find. These tests make the *current* scope a thing that fails loudly when
contradicted rather than a thing someone has to notice.

They check documentation, which is unusual for a test suite. That is
deliberate: in this repository the scope statement is the artifact that
governs every other decision, and nothing else was checking it.

2026-09-24: the repository was reorganized to a research-paper-friendly
layout (``systems/``/``experiments/evaluation`` -> ``src/``,
``alzheimer_corpus/`` -> ``corpus/``, ``experiments/outputs/`` ->
``results/``) and ``docs/`` was replaced with five topic docs
(``methodology.md``, ``data.md``, ``glossary.md``,
``evaluation.md``, ``reproducibility.md``). The four previous docs
(``current_objectives.md``, ``research_experimental_specification.md``,
``status_and_decisions.md``, ``question_review.md``) were archived to
``_archive/docs_legacy/`` rather than deleted, and this file's guards were
retargeted at the new docs and at README.md, which now states the research
question and objectives directly rather than pointing to a separate
canonical-scope document.

2026-10-02: the research direction was re-scoped to an as-of evaluation on MedChangeQA
(primary) with the Alzheimer's study as a secondary case study; the RAG2 filter-reproduction
attempt and the first Alzheimer's pilot runners moved to ``_archive/``. The guards below
were retargeted accordingly, and ``DocumentationIntegrityTests`` was added so that a path or
command named in the documentation can no longer silently stop existing.

2026-10-03: stage 1 (the Temporal Filter) failed its dev gate, and the study gained a stage 2, an
evidence-synthesis layer with two pre-specified questions (RQ1, RQ2) tested once on the confirmatory
split. The README guards below keep those questions, and the stated limit of what 528 items can
confirm, from drifting; the plan guard keeps the list of decisions taken after seeing dev data.

2026-10-05: ``docs/evaluation.md`` was merged into ``docs/experiment_plan.md`` (one document for the protocol
and the evaluation; its section numbers are cited from the code and stay stable), and that file was then renamed
``docs/experimentation.md``. ``docs/`` now holds the topic docs ``methodology.md``, ``data.md``, ``glossary.md``
and ``reproducibility.md``, the chronological record ``log.md`` and ``experimentation.md``. Citations of the
old stage-1/stage-2 protocol name the file ``experiment_plan.md`` as it was at commit 92e3aaf.

2026-10-05: the study was realigned around an adapted RAG² baseline and an evidence-criteria
verification extension, with a supervisor requirement of +1 percentage point. The stage-2 guard was
retargeted to the realigned question; it keeps the "tested once" rule and the statement of what 528
questions can and cannot confirm, so the requirement cannot silently turn into an unqualified claim.

2026-10-05 (repository audit): the superseded Alzheimer's-specific framework (the local corpus, the question
pool, the three-arm runner, the human annotation workflow) moved to ``_archive/alzheimers_framework/``, the
MedCPT encoders to ``experiments/medchange/encoders.py`` and the Temporal Filter formula to
``src/temporal_filter/``. The layout guards below were retargeted. Two deliberate choices: the active tree is
checked by named *files*, not directories, because a laptop that pulls this change still holds the
gitignored ``corpus/data/`` and ``experiments/results/index/`` folders; and the documentation check refuses
any current document that names moved code by its old path.
"""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
METHODOLOGY = ROOT / "docs" / "methodology.md"
README = ROOT / "README.md"

def text(path):
    return path.read_text(encoding="utf-8")


class CurrentScopeTests(unittest.TestCase):

    def test_readme_states_the_research_objectives(self):
        body = " ".join(text(README).lower().split())
        self.assertIn("implement and validate the proposed system", body)
        self.assertIn("compared with relevant baseline models and existing works", body)

    def test_readme_states_exactly_the_two_objectives(self):
        """The objectives are a numbered list in README section 2; an objective
        added or dropped without updating the plan fails here."""
        section = text(README).split("## 2.", 1)[1].split("## 3.", 1)[0]
        numbered = re.findall(r"^\d+\.\s", section, flags=re.MULTILINE)
        self.assertEqual(len(numbered), 2, "README section 2 must list exactly two objectives")

    def test_readme_names_the_temporal_filter_and_does_not_overclaim_it(self):
        """"The Temporal Filter" is the one consistent name for the proposed
        system. Its mechanism is a recency term added to a relevance score,
        which published work (e.g. TempRALM) already does, so the README must say
        that no novelty is claimed for it."""
        body = " ".join(text(README).split())
        self.assertIn("Temporal Filter", body)
        self.assertIn("no novelty is claimed for the formula", body)

    def test_readme_names_verdict_accuracy_as_the_primary_outcome(self):
        body = " ".join(text(README).split())
        self.assertIn("primary outcome is **verdict accuracy**", body)
        self.assertNotIn("Primary | Currency", body)

    def test_readme_does_not_present_the_rag2_stand_in_as_a_reproduction(self):
        body = " ".join(text(README).split())
        self.assertIn("**not** a RAG² reproduction", body)

    def test_readme_does_not_assume_the_answer(self):
        """The point of the comparison is to determine improvement
        experimentally, not to assert it - dropping this framing would
        silently turn a research question into a foregone conclusion."""
        body = " ".join(text(README).lower().split())
        self.assertIn("does not commit in advance to which one it will report", body)

    def test_readme_states_the_realigned_question_and_its_limits(self):
        """The realigned study tests one proposed component against an adapted RAG² baseline, once, on
        a held-out split, under a +1 pp requirement read by a pre-declared rule. Dropping the question,
        the rule, or the sentence that says a 1-point difference cannot be confirmed at this sample size
        would turn a hedged design into an unqualified claim."""
        body = " ".join(text(README).split())
        for needle in ("adapted RAG²", "evidence-criteria verification", "tested once",
                       "at least 1 percentage point", "met as a point estimate, not confirmed",
                       "A 1-point difference cannot be confirmed with 528 questions"):
            self.assertIn(needle, body)

    def test_plan_lists_the_decisions_taken_after_seeing_dev_data(self):
        body = " ".join(text(ROOT / "docs" / "experimentation.md").split())
        self.assertIn("Decisions taken after seeing dev data", body)
        self.assertIn("forking-path ledger", body)

    def test_methodology_states_what_is_held_constant(self):
        body = text(METHODOLOGY)
        self.assertIn("held constant", body.lower())
        self.assertIn("Temporal Filter", body)


class SupersededScopeTests(unittest.TestCase):
    """The old (hallucination-rate-primary) question may be remembered as
    history, never asserted as current."""

    def test_archive_readme_records_why_the_earlier_work_is_not_current(self):
        """Both earlier research questions must stay *recorded somewhere* -
        deleting the explanation would make _archive/test_pairs/
        inexplicable."""
        archive_readme = ROOT / "_archive" / "README.md"
        self.assertTrue(archive_readme.exists())
        body = text(archive_readme)
        self.assertIn("not part of the current", body.lower())
        self.assertIn("test_pairs", body)

    def test_active_code_does_not_import_the_archive(self):
        """The archive move is only real isolation if nothing active
        depends on it - the one thing this whole cleanup could get wrong
        silently."""
        import ast

        hits = []
        for path in ROOT.rglob("*.py"):
            parts = path.relative_to(ROOT).parts
            if parts[0] in ("_archive", "__pycache__") or "__pycache__" in parts:
                continue
            if parts[:3] == ("experiments", "medchange", "data"):
                continue
            tree = ast.parse(text(path), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if node.module.split(".")[0] == "_archive":
                        hits.append(str(path))
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[0] == "_archive":
                            hits.append(str(path))
        self.assertEqual(hits, [])

    def test_the_current_docs_all_exist(self):
        """docs/ holds four topic docs (methodology, data, glossary,
        reproducibility), the chronological record ``log.md``, and
        ``experimentation.md``, the protocol and evaluation fixed before any result existed.
        A missing expected file means a reference in this repository now
        dangles; an unexpected one means the docs set drifted."""
        docs = ROOT / "docs"
        expected = {
            "methodology.md",
            "data.md",
            "glossary.md",
            "reproducibility.md",
            "log.md",
            "experimentation.md",
        }
        self.assertEqual({p.name for p in docs.glob("*.md")}, expected)

    def test_legacy_docs_are_archived_not_deleted(self):
        """The four previous docs were replaced, not discarded - they must
        still be readable for anyone who wants the fuller historical
        record."""
        legacy = ROOT / "_archive" / "docs_legacy"
        expected = {
            "current_objectives.md",
            "research_experimental_specification.md",
            "status_and_decisions.md",
            "question_review.md",
        }
        self.assertEqual({p.name for p in legacy.glob("*.md")}, expected)

    def test_readme_does_not_state_the_old_question_as_current(self):
        """The old primary/secondary framing (hallucination rate primary,
        QA accuracy secondary, no third objective) must not reappear as if
        it still governed - it is exactly the drift this file exists to
        catch, twice now."""
        body = text(README)
        self.assertNotIn("Primary | Hallucination rate", body)
        self.assertNotIn("stays primary", body)


class ArchiveLayoutTests(unittest.TestCase):

    #: Files (not folders: see the module docstring) that were active before 2026-10-05 and now live only
    #: in the archive. The first group is the abandoned filter-reproduction and pilot work, the second the
    #: Alzheimer's-specific framework.
    MOVED_OUT_OF_THE_ACTIVE_TREE = (
        "experiments/baseline", "experiments/shared/runners/fit_and_evaluate.py",
        "experiments/shared/runners/run_real_evaluation.py", "experiments/results/fit_and_evaluate",
        "corpus/scripts/01_pubmed_download.py", "corpus/config/search_queries.yaml",
        "corpus/metadata/pmc.csv", "src/baseline/admission.py", "src/proposed/scorer.py",
        "src/common/generator.py", "src/common/system.py", "evaluation/runner.py",
        "evaluation/freezing.py", "evaluation/questions.py", "evaluation/rag_metrics.py",
        "experiments/shared/questions/build_pool.py", "experiments/shared/retrieval/pipeline.py",
        "experiments/shared/retrieval/encoders.py", "experiments/shared/runners/run_end_to_end.py",
        "evaluation/tests/integration/test_end_to_end_runner.py", "evaluation/tests/corpus_scaffold.py",
        "evaluation/tests/unit/test_runner_parity.py",
    )
    #: Where they are now (and the active files that replaced the shared ones).
    NOW_IN_THE_ARCHIVE = (
        "_archive/rag2_filter_reproduction/filter_training", "_archive/alzheimers_pilot_v1/fit_and_evaluate.py",
        "_archive/alzheimers_framework/corpus/scripts/01_pubmed_download.py",
        "_archive/alzheimers_framework/corpus/metadata/pmc.csv",
        "_archive/alzheimers_framework/src/baseline/admission.py",
        "_archive/alzheimers_framework/evaluation/runner.py",
        "_archive/alzheimers_framework/experiments/shared/questions/splits.json",
        "_archive/alzheimers_framework/experiments/shared/retrieval/pipeline.py",
        "_archive/alzheimers_framework/tests/unit",
    )
    STAYED_ACTIVE = ("experiments/medchange/encoders.py", "src/temporal_filter/scorer.py",
                     "src/temporal_filter/temporal.py", "src/common/evidence.py", "evaluation/stats.py")

    def test_abandoned_work_lives_in_the_archive_not_in_the_active_tree(self):
        for moved in self.MOVED_OUT_OF_THE_ACTIVE_TREE:
            self.assertFalse((ROOT / moved).exists(), f"{moved} should live under _archive/")
        for archived in self.NOW_IN_THE_ARCHIVE + self.STAYED_ACTIVE:
            self.assertTrue((ROOT / archived).exists(), archived)

    def test_every_active_package_is_listed_for_installation_and_nothing_else_is(self):
        """``pip install -e .`` must make exactly the active packages importable. The list in
        ``pyproject.toml`` is hand-maintained, and moving a package (as the 2026-10-05 reorganisation did)
        leaves a stale entry that no test notices, because every test runs from the repository root and
        never needs the installed package."""
        body = text(ROOT / "pyproject.toml")
        listed = set(re.findall(r'^\s*"([\w.]+)",?\s*$', body.split("packages = [", 1)[1].split("]", 1)[0],
                                flags=re.MULTILINE))
        found = set()
        for init in ROOT.rglob("__init__.py"):
            parts = init.relative_to(ROOT).parts[:-1]
            # build output (``pip install .`` builds in the source tree: build/lib/...) is not a package
            if (not parts or parts[0] in ("_archive", "build", "dist") or parts[0].endswith(".egg-info")
                    or "__pycache__" in parts or "tests" in parts or parts[:3] == ("experiments", "medchange", "data")):
                continue
            found.add(".".join(parts))
        self.assertEqual(listed, found)

    def test_archive_readme_lists_every_archived_folder(self):
        body = text(ROOT / "_archive" / "README.md")
        for child in sorted(p.name for p in (ROOT / "_archive").iterdir()
                            if p.is_dir() and p.name != "__pycache__"):
            self.assertIn(child, body, f"_archive/{child} is not described in _archive/README.md")


#: Documentation that must describe the CURRENT repository exactly. ``log.md`` is a dated
#: record of what was true when written and is deliberately excluded.
CURRENT_DOCS = ("README.md", "docs/methodology.md", "docs/data.md",
                "docs/reproducibility.md", "docs/glossary.md", "docs/experimentation.md",
                "experiments/medchange/README.md", "experiments/medchange/results/README.md")
#: The archive's own READMEs. They name old locations on purpose (a "was here, is there now" table), so only the
#: ``_archive/...`` paths and commands they give are checked, and the moved-code check does not apply.
ARCHIVE_DOCS = ("_archive/README.md", "_archive/alzheimers_framework/README.md")
#: Paths named in the docs that are generated or machine-local and may be absent.
GENERATED_PREFIXES = (
    "experiments/medchange/data", "experiments/medchange/results/analysis_",
    "experiments/medchange/results/rag2_", "experiments/medchange/results/RAG2_",
    "experiments/medchange/results/answers_", "experiments/medchange/results/helpfulness_",
    "_archive/alzheimers_framework/experiments/results/index",
    "_archive/alzheimers_framework/corpus/data/chunks", "_archive/alzheimers_framework/corpus/data/normalized",
    "_archive/alzheimers_framework/corpus/data/deduplicated", "_archive/alzheimers_framework/corpus/data/raw/pmc",
    "_archive/rag2_filter_reproduction/filter_training/labels",
    "_archive/rag2_filter_reproduction/filter_training/.textbook_index_cache",
    "experiments/baseline",
)
#: Code that moved on 2026-10-05, by its old active path. A current document that names one of these is
#: pointing at a path that no longer exists; the archive path (``_archive/alzheimers_framework/...``) is fine,
#: because the lookbehind rejects a match preceded by ``/``.
MOVED_CODE = re.compile(
    r"(?<![\w./-])(?:experiments/shared|experiments/results|src/baseline|src/proposed"
    r"|src/common/(?:generator|hf_generator|system)"
    r"|evaluation/(?:runner|freezing|questions|accuracy|rag_metrics|annotation)\.py"
    r"|evaluation/tests/(?:integration|corpus_scaffold)"
    r"|corpus/(?:scripts|config|metadata|reports|data|logs))")


class DocumentationIntegrityTests(unittest.TestCase):
    """Every repository path and every ``python -m`` command that the current docs name
    must exist. Dangling references were the most common defect found in the 2026-10-02
    audit; this makes them fail loudly."""

    PATH = re.compile(r"`((?:src|evaluation|experiments|corpus|docs|_archive)/[^`\s]*)`")
    COMMAND = re.compile(r"python -m ([A-Za-z_][\w.]*)")

    def _docs(self):
        for rel in CURRENT_DOCS:
            yield rel, text(ROOT / rel)

    def _archive_docs(self):
        for rel in ARCHIVE_DOCS:
            yield rel, text(ROOT / rel)

    def test_named_paths_exist(self):
        missing = []
        for rel, body, only_archive in ([(r, b, False) for r, b in self._docs()]
                                        + [(r, b, True) for r, b in self._archive_docs()]):
            for m in self.PATH.finditer(body):
                path = m.group(1).rstrip("/.,:;")
                if only_archive and not path.startswith("_archive/"):
                    continue                            # an archive README names old locations on purpose
                if any(c in path for c in "*<>{}") or path.startswith(GENERATED_PREFIXES):
                    continue
                if not (ROOT / path).exists():
                    missing.append(f"{rel}: {path}")
        self.assertEqual(missing, [])

    def test_named_commands_exist(self):
        missing = []
        for rel, body in list(self._docs()) + list(self._archive_docs()):
            for m in self.COMMAND.finditer(body):
                module = m.group(1)
                if module in ("unittest", "pyflakes"):
                    continue
                base = ROOT.joinpath(*module.split("."))
                if not (base.with_suffix(".py").exists() or (base / "__main__.py").exists()):
                    missing.append(f"{rel}: python -m {module}")
        self.assertEqual(missing, [])

    #: A repository file cited as ``dir/.../name.ext`` in a comment or docstring.
    CODE_PATH = re.compile(
        r"(?<![\w./-])((?:src|evaluation|experiments|corpus|docs|_archive)/[\w./-]*\w"
        r"\.(?:py|md|yaml|yml|json|jsonl|csv|toml|log))(?![\w-])")

    def test_paths_cited_in_code_comments_and_docstrings_exist(self):
        """Comments cite files (``docs/...``, ``src/...``) and go stale when the files move;
        the 2026-09-24 move from ``systems/`` to ``src/`` left six such references that no
        earlier check could see. This file is skipped because it names the old locations
        on purpose, and the archive because its code describes its own moment in time."""
        missing = []
        for path in sorted(ROOT.rglob("*.py")):
            parts = path.relative_to(ROOT).parts
            if (parts[0] == "_archive" or "__pycache__" in parts
                    or parts[:3] == ("experiments", "medchange", "data")
                    or path == Path(__file__).resolve()):
                continue
            for number, line in enumerate(text(path).splitlines(), 1):
                for m in self.CODE_PATH.finditer(line):
                    cited = m.group(1)
                    if any(c in cited for c in "*<>{}") or cited.startswith(GENERATED_PREFIXES):
                        continue
                    if not (ROOT / cited).exists():
                        missing.append(f"{'/'.join(parts)}:{number}: {cited}")
        self.assertEqual(missing, [])

    def test_current_docs_do_not_point_at_moved_or_retired_code(self):
        stale = ("experiments.baseline", "experiments/baseline/filter_training",
                 "experiments.shared.runners.fit_and_evaluate",
                 "experiments.shared.runners.run_real_evaluation", "next_phase_plan")
        offenders = []
        for rel, body in self._docs():
            for term in stale:
                if term in body:
                    offenders.append(f"{rel}: {term}")
        self.assertEqual(offenders, [])

    def test_current_docs_name_moved_code_only_by_its_archive_path(self):
        """The 2026-10-05 move left ``corpus/``, ``experiments/shared/``, ``src/baseline/`` and the like
        in the archive. A current document that still gives the old path points at nothing (on a laptop
        that holds the gitignored ``corpus/data/`` it would even look as if it worked)."""
        offenders = []
        for rel, body in self._docs():
            for number, line in enumerate(body.splitlines(), 1):
                for m in MOVED_CODE.finditer(line):
                    offenders.append(f"{rel}:{number}: {m.group(0)}")
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()

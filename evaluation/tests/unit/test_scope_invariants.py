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

    def test_readme_states_the_stage_2_questions_and_their_limits(self):
        """Stage 2 is two pre-specified questions tested once on a held-out split. Dropping the
        questions, or the sentence that says how small an effect the benchmark can confirm, would
        turn a hedged design into an unqualified claim."""
        body = " ".join(text(README).split())
        for needle in ("RQ1", "RQ2", "evidence-synthesis layer", "tested once",
                       "only effects of about 5 percentage points or more can be confirmed"):
            self.assertIn(needle, body)

    def test_plan_lists_the_decisions_taken_after_seeing_dev_data(self):
        body = " ".join(text(ROOT / "docs" / "experiment_plan.md").split())
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
        """docs/ holds five topic docs (methodology, data, glossary,
        evaluation, reproducibility), the chronological record ``log.md``, and
        ``experiment_plan.md``, the protocol fixed before any result existed.
        A missing expected file means a reference in this repository now
        dangles; an unexpected one means the docs set drifted."""
        docs = ROOT / "docs"
        expected = {
            "methodology.md",
            "data.md",
            "glossary.md",
            "evaluation.md",
            "reproducibility.md",
            "log.md",
            "experiment_plan.md",
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

    def test_abandoned_work_lives_in_the_archive_not_in_the_active_tree(self):
        for moved in ("experiments/baseline", "experiments/shared/runners/fit_and_evaluate.py",
                      "experiments/shared/runners/run_real_evaluation.py",
                      "experiments/results/fit_and_evaluate"):
            self.assertFalse((ROOT / moved).exists(), f"{moved} should live under _archive/")
        for archived in ("_archive/rag2_filter_reproduction/filter_training",
                         "_archive/alzheimers_pilot_v1/fit_and_evaluate.py"):
            self.assertTrue((ROOT / archived).exists(), archived)

    def test_archive_readme_lists_every_archived_folder(self):
        body = text(ROOT / "_archive" / "README.md")
        for child in sorted(p.name for p in (ROOT / "_archive").iterdir()
                            if p.is_dir() and p.name != "__pycache__"):
            self.assertIn(child, body, f"_archive/{child} is not described in _archive/README.md")


#: Documentation that must describe the CURRENT repository exactly. ``log.md`` is a dated
#: record of what was true when written and is deliberately excluded.
CURRENT_DOCS = ("README.md", "docs/methodology.md", "docs/data.md", "docs/evaluation.md",
                "docs/reproducibility.md", "docs/glossary.md", "docs/experiment_plan.md",
                "experiments/medchange/README.md", "experiments/medchange/results/README.md",
                "experiments/results/README.md", "_archive/README.md")
#: Paths named in the docs that are generated or machine-local and may be absent.
GENERATED_PREFIXES = (
    "experiments/medchange/data", "experiments/medchange/results/analysis_",
    "experiments/medchange/results/answers_", "experiments/medchange/results/helpfulness_",
    "experiments/results/index", "corpus/data/chunks", "corpus/data/normalized",
    "corpus/data/deduplicated", "corpus/data/raw/pmc", "corpus/data/raw/guidelines",
    "corpus/data/raw/textbooks", "corpus/data/raw/currency_pack",
    "_archive/rag2_filter_reproduction/filter_training/labels",
    "_archive/rag2_filter_reproduction/filter_training/.textbook_index_cache",
    "experiments/baseline",
)


class DocumentationIntegrityTests(unittest.TestCase):
    """Every repository path and every ``python -m`` command that the current docs name
    must exist. Dangling references were the most common defect found in the 2026-10-02
    audit; this makes them fail loudly."""

    PATH = re.compile(r"`((?:src|evaluation|experiments|corpus|docs|_archive)/[^`\s]*)`")
    COMMAND = re.compile(r"python -m ([A-Za-z_][\w.]*)")

    def _docs(self):
        for rel in CURRENT_DOCS:
            yield rel, text(ROOT / rel)

    def test_named_paths_exist(self):
        missing = []
        for rel, body in self._docs():
            for m in self.PATH.finditer(body):
                path = m.group(1).rstrip("/.,:;")
                if any(c in path for c in "*<>{}") or path.startswith(GENERATED_PREFIXES):
                    continue
                if not (ROOT / path).exists():
                    missing.append(f"{rel}: {path}")
        self.assertEqual(missing, [])

    def test_named_commands_exist(self):
        missing = []
        for rel, body in self._docs():
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
            if rel == "_archive/README.md":
                continue
            for term in stale:
                if term in body:
                    offenders.append(f"{rel}: {term}")
        self.assertEqual(offenders, [])


class AblationMetricsTests(unittest.TestCase):
    """ROUGE/BLEU-style automatic metrics were forbidden under the old
    scope and are REQUIRED under the current one (objective 2, the ablation
    study) - this is the current scope's most easily-missed inversion of
    the old one, so it gets its own explicit guard."""

    def test_rag_metrics_module_exists_and_implements_standard_metrics(self):
        from evaluation import rag_metrics as rm
        for name in ("exact_match", "token_f1", "rouge_l_f1", "context_scores",
                    "groundedness"):
            self.assertTrue(hasattr(rm, name), f"rag_metrics.py is missing {name}")

    def test_accuracy_module_still_excludes_automatic_scoring_from_its_own_judgement(self):
        """rag_metrics.py owns automatic metrics now; accuracy.py's separate
        decision not to compute them itself still stands (see its
        docstring) - this just confirms accuracy.py wasn't quietly given an
        automatic scorer of its own, which would duplicate rag_metrics.py
        under a different name."""
        from evaluation import accuracy
        source = Path(accuracy.__file__).read_text(encoding="utf-8")
        self.assertNotIn("def rouge", source.lower())
        self.assertNotIn("def bleu", source.lower())


if __name__ == "__main__":
    unittest.main()

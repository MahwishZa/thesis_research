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
"""

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
METHODOLOGY = ROOT / "docs" / "methodology.md"
README = ROOT / "README.md"

#: The three objectives, as headline verbs - each must appear so dropping an
#: objective silently is caught.
OBJECTIVE_VERBS = (
    "validation",
    "ablation",
    "comparison",
)


def text(path):
    return path.read_text(encoding="utf-8")


class CurrentScopeTests(unittest.TestCase):

    def test_readme_states_the_research_objectives(self):
        body = text(README).lower()
        self.assertIn("implement and validate the proposed system", body)
        self.assertIn("compared with the baseline model", body)

    def test_readme_states_the_three_objectives(self):
        body = text(README)
        for verb in OBJECTIVE_VERBS:
            self.assertIn(verb, body.lower(),
                          f"README.md is missing an objective: {verb!r}")

    def test_readme_names_the_temporal_filter(self):
        """"The Temporal Filter" is the one consistent name for the
        proposed method - dropping it silently back to "recency-aware
        admission" or similar is exactly the terminology drift a 2026-09-20
        cleanup fixed once already."""
        body = text(README)
        self.assertIn("Temporal Filter", body)
        self.assertNotIn("recency", body.lower())

    def test_readme_does_not_assume_the_answer(self):
        """The point of the comparison is to determine improvement
        experimentally, not to assert it - dropping this framing would
        silently turn a research question into a foregone conclusion."""
        body = text(README).lower()
        self.assertIn("does not commit in advance to which one it will report", body)

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

    def test_the_five_current_docs_all_exist(self):
        """The 2026-09-24 reorganization reduced docs/ to five topic files
        and pointed code, tests and README at them. ``research_log.md`` was
        added after that reorganization as a sixth, standing doc (the
        chronological implementation record - see its own header) rather
        than a topic doc describing current state, so it is listed here
        too. A missing expected file means a reference in this repository
        now dangles."""
        docs = ROOT / "docs"
        expected = {
            "methodology.md",
            "data.md",
            "glossary.md",
            "evaluation.md",
            "reproducibility.md",
            "research_log.md",
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

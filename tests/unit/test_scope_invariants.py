"""Guards against silent research-scope drift.

The repository has already changed research question twice, and each change
left stale statements in several documents that took a dedicated audit to
find. These tests make the *current* scope a thing that fails loudly when
contradicted rather than a thing someone has to notice.

They check documentation, which is unusual for a test suite. That is
deliberate: in this repository the scope statement is the artifact that
governs every other decision, and nothing else was checking it.

This file guards ``docs/current_objectives.md`` (adopted 2026-09-18), the
current canonical scope. It previously guarded ``docs/frozen_scope.md``,
which is itself now superseded by current_objectives.md; that history is
why ROUGE/BLEU/BERTScore, once a forbidden outcome, are now a REQUIRED part
of the ablation study (``experiments/evaluation/rag_metrics.py``) - the
opposite invariant. Guarding a superseded document's *old* invariants after
the scope moved on would itself be the kind of drift this file exists to
catch, so this file was retargeted rather than left in place.
"""

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCOPE = ROOT / "docs" / "current_objectives.md"
README = ROOT / "README.md"

#: Load-bearing fragments of the current scope. Wording may be reformatted
#: around them, so these are phrases rather than whole sentences.
CURRENT_OBJECTIVES = (
    "Proposed-system validation",
    "Ablation study",
    "RAG",  # "RAG² comparison" - the ² is easy to garble in an edit, don't require it literally
)

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

    def test_current_objectives_states_the_three_objectives(self):
        body = text(SCOPE)
        for fragment in CURRENT_OBJECTIVES:
            self.assertIn(fragment, body,
                          f"current_objectives.md no longer states: {fragment!r}")

    def test_current_objectives_names_rag2_improvement_as_the_main_contribution(self):
        body = text(SCOPE).lower()
        self.assertIn("main contribution", body)
        self.assertIn("improves rag", body)

    def test_current_objectives_does_not_assume_the_answer(self):
        """The main contribution is to determine improvement experimentally,
        not to assert it - dropping this framing would silently turn a
        research question into a foregone conclusion."""
        body = text(SCOPE).lower()
        self.assertIn("do not assume the answer", body)

    def test_readme_states_the_three_objectives(self):
        body = text(README)
        for verb in OBJECTIVE_VERBS:
            self.assertIn(verb, body.lower(),
                          f"README.md is missing an objective: {verb!r}")

    def test_readme_points_to_current_objectives_as_canonical(self):
        body = text(README)
        self.assertIn("current_objectives.md", body)


class SupersededScopeTests(unittest.TestCase):
    """The old (hallucination-rate-primary) question may be remembered as
    history, never asserted as current."""

    def test_frozen_scope_is_marked_superseded(self):
        body = text(ROOT / "docs" / "frozen_scope.md")
        self.assertIn("SUPERSEDED", body)
        self.assertIn("current_objectives.md", body)

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
        from experiments.evaluation import rag_metrics as rm
        for name in ("exact_match", "token_f1", "rouge_l_f1", "context_scores",
                    "groundedness"):
            self.assertTrue(hasattr(rm, name), f"rag_metrics.py is missing {name}")

    def test_accuracy_module_still_excludes_automatic_scoring_from_its_own_judgement(self):
        """rag_metrics.py owns automatic metrics now; accuracy.py's separate
        decision not to compute them itself still stands (see its
        docstring) - this just confirms accuracy.py wasn't quietly given an
        automatic scorer of its own, which would duplicate rag_metrics.py
        under a different name."""
        from experiments.evaluation import accuracy
        source = Path(accuracy.__file__).read_text(encoding="utf-8")
        self.assertNotIn("def rouge", source.lower())
        self.assertNotIn("def bleu", source.lower())


if __name__ == "__main__":
    unittest.main()

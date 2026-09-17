"""Guards against silent research-scope drift.

The repository has already changed research question once, and the change left
stale statements in several documents that took a dedicated audit to find.
These tests make the *current* scope a thing that fails loudly when contradicted
rather than a thing someone has to notice.

They check documentation, which is unusual for a test suite. That is
deliberate: in this repository the scope statement is the artifact that
governs every other decision, and nothing else was checking it.
"""

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCOPE = ROOT / "docs" / "frozen_scope.md"
README = ROOT / "README.md"
METHODOLOGY = ROOT / "docs" / "methodology.md"

#: Phrases from the current research question. Wording may be reformatted, so
#: these are the load-bearing fragments rather than the whole sentence.
CURRENT_QUESTION = (
    "reduce the rate of hallucinated answers",
    "relative to the baseline system",
    "comparable QA accuracy",
)

#: Outcomes that are NOT research objectives. Naming one as an objective is
#: the specific drift these tests exist to catch.
FORBIDDEN_OBJECTIVES = (
    "hallucination subtype",
    "error taxonomy",
    "retrieval quality as",
    "ROUGE",
    "BLEU",
    "BERTScore",
)


def text(path):
    return path.read_text(encoding="utf-8")


class CurrentScopeTests(unittest.TestCase):

    def test_frozen_scope_states_the_current_research_question(self):
        body = text(SCOPE)
        for fragment in CURRENT_QUESTION:
            self.assertIn(fragment, body,
                          f"frozen_scope.md no longer states: {fragment!r}")

    def test_readme_states_the_same_question(self):
        body = text(README)
        for fragment in CURRENT_QUESTION:
            self.assertIn(fragment, body,
                          f"README.md no longer states: {fragment!r}")

    def test_exactly_two_objectives_are_declared_primary_and_secondary(self):
        body = text(SCOPE)
        self.assertIn("PRIMARY", body)
        self.assertIn("SECONDARY", body)
        self.assertIn("Hallucination rate", body)
        self.assertIn("QA accuracy", body)

    def test_frozen_scope_denies_a_third_objective(self):
        self.assertIn("no diagnostic objective", text(SCOPE).lower())


class SupersededScopeTests(unittest.TestCase):
    """The old question may be remembered, never asserted as current."""

    def test_the_pivot_is_recorded_as_superseded(self):
        body = text(SCOPE)
        self.assertIn("asymmetry", body, "the pivot should stay documented")
        self.assertIn("Superseded design", body)
        self.assertIn("no longer the research question", body)

    def test_the_primary_question_section_is_free_of_the_old_question(self):
        """Mentions elsewhere are fine - in the exclusion list, or as history.

        What must never happen is the old question reappearing where the
        current one is stated.
        """
        body = text(SCOPE)
        start = body.index("## 1. Primary research question")
        section = body[start:body.index("## 2.")]
        self.assertNotIn("asymmetry", section.lower())
        self.assertIn("hallucinated answers", section)

    def test_methodology_does_not_claim_frozen_scope_is_stale(self):
        """A note saying frozen_scope.md disagrees was itself the stale part."""
        body = text(METHODOLOGY)
        self.assertNotIn("has not been rewritten", body)


class ForbiddenObjectiveTests(unittest.TestCase):

    def test_scope_names_the_excluded_outcomes_as_excluded(self):
        """Excluded outcomes must be listed, so exclusion is explicit."""
        body = text(SCOPE)
        excluded_block = body[body.index("no third objective"):]
        for phrase in ("subtype", "taxonomy", "ROUGE"):
            self.assertIn(phrase, excluded_block)

    def test_readme_does_not_promote_an_excluded_outcome(self):
        """A forbidden term may appear only where something rules it out.

        Checked per paragraph rather than per line: the negation and the term
        it negates are routinely separated by a line wrap.
        """
        paragraphs = re.split(r"\n\s*\n", text(README))
        for para in paragraphs:
            for phrase in FORBIDDEN_OBJECTIVES:
                if phrase.lower() in para.lower():
                    self.assertTrue(
                        re.search(r"\bno\b|\bnot\b|exclud|without|deliberately",
                                  para, re.I),
                        f"README paragraph promotes an excluded outcome: "
                        f"{para!r}",
                    )


if __name__ == "__main__":
    unittest.main()

"""Scope guard for the archived framework's automatic-metric modules.

Moved here from ``evaluation/tests/unit/test_scope_invariants.py`` on 2026-10-05: it tests archived modules, and
active tests must not import the archive. The test bodies are unchanged.

ROUGE/BLEU-style automatic metrics were forbidden under the first scope and required under the second (objective 2,
the ablation study); ``rag_metrics.py`` owns them and ``accuracy.py`` deliberately does not compute them.
"""

import unittest
from pathlib import Path


class AblationMetricsTests(unittest.TestCase):

    def test_rag_metrics_module_exists_and_implements_standard_metrics(self):
        from _archive.alzheimers_framework.evaluation import rag_metrics as rm
        for name in ("exact_match", "token_f1", "rouge_l_f1", "context_scores",
                     "groundedness"):
            self.assertTrue(hasattr(rm, name), f"rag_metrics.py is missing {name}")

    def test_accuracy_module_still_excludes_automatic_scoring_from_its_own_judgement(self):
        """rag_metrics.py owns automatic metrics now; accuracy.py's separate
        decision not to compute them itself still stands (see its
        docstring) - this just confirms accuracy.py wasn't quietly given an
        automatic scorer of its own, which would duplicate rag_metrics.py
        under a different name."""
        from _archive.alzheimers_framework.evaluation import accuracy
        source = Path(accuracy.__file__).read_text(encoding="utf-8")
        self.assertNotIn("def rouge", source.lower())
        self.assertNotIn("def bleu", source.lower())


if __name__ == "__main__":
    unittest.main()

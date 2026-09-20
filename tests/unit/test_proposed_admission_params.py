"""Parameter-validation guards for the proposed system.

theta, lambda and the half-life are the three quantities the design says
must be fitted on the validation split and never defaulted. Two of them
already refused an out-of-range value; theta did not, and the runner
hard-coded all three with no way to supply a fitted value. These tests lock
both corrections.
"""

import unittest
from datetime import date

from systems.proposed.admission import AdmissionConfig
from systems.proposed.scorer import AdmissionScorer


def config(theta):
    return AdmissionConfig(
        admit_threshold=theta,
        question_date=date(2026, 1, 1),
        max_admitted_passages=5,
    )


class ThetaRangeTests(unittest.TestCase):
    """A(s) is a convex combination of two [0, 1] quantities, so theta is
    only meaningful on [0, 1]. Outside it, theta is not merely wrong - it
    is silently degenerate, and a validation-split sweep that strays out of
    range would produce a plausible-looking but meaningless arm."""

    def test_theta_above_one_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            config(1.5).validate()
        self.assertIn("[0, 1]", str(ctx.exception))

    def test_theta_below_zero_is_rejected(self):
        with self.assertRaises(ValueError):
            config(-0.1).validate()

    def test_the_endpoints_are_valid(self):
        """theta=0 admits everything that scores at all and theta=1 admits
        only a perfect score. Both are legitimate extremes of the design,
        not errors - the range check must not exclude them."""
        config(0.0).validate()
        config(1.0).validate()

    def test_unresolved_theta_is_still_rejected(self):
        """The pre-existing guard must survive the range check."""
        with self.assertRaises(ValueError) as ctx:
            config(None).validate()
        self.assertIn("validation split", str(ctx.exception))


class LambdaRangeTests(unittest.TestCase):

    def test_out_of_range_lambda_is_rejected(self):
        for bad in (1.5, -0.5):
            with self.assertRaises(ValueError):
                AdmissionScorer(bad)

    def test_unresolved_lambda_is_rejected(self):
        with self.assertRaises(ValueError):
            AdmissionScorer(None)

    def test_lambda_actually_reaches_the_score(self):
        """A parameter that is accepted but not used would be the worst
        case: every ablation cell would be identical and nothing would say
        so. At lambda=0 the temporal term must contribute exactly nothing;
        at lambda=1 it must be the whole score."""

        class Candidate:
            def __init__(self, rank):
                self.rerank_rank = rank

        zero = AdmissionScorer(0.0)
        scores = {
            zero.score(Candidate(2), temporal=r, candidate_count=4).total
            for r in (0.0, 0.5, 1.0)
        }
        self.assertEqual(len(scores), 1, "temporal still moved A(s) at lambda=0")
        self.assertAlmostEqual(scores.pop(), zero.normalize_rank(2, 4))

        one = AdmissionScorer(1.0)
        for r in (0.0, 0.5, 1.0):
            self.assertAlmostEqual(
                one.score(Candidate(2), temporal=r, candidate_count=4).total, r
            )


if __name__ == "__main__":
    unittest.main()

"""The answer prompt and the reading of the verdict line and of the judge's label line (deterministic; no model)."""

import unittest

from experiments.medchange.prompts import build_prompt, parse_label, parse_verdict


def cand(pmid, rank, year, helpful=None, month="01"):
    c = {"pmid": pmid, "rank": rank, "lower": f"{year}-{month}-01", "upper": f"{year}-{month}-01",
         "title": f"T{pmid}", "abstract": f"abstract {pmid}", "pubtypes": [], "journal": "J"}
    if helpful is not None:
        c["helpful"] = helpful
    return c


class PromptTests(unittest.TestCase):

    def test_prompt_has_evidence_only_when_admitted_and_never_dates(self):
        p = build_prompt("Does X help Y?", [cand("1", 1, 2001)])
        self.assertIn("[1] T1. abstract 1", p)
        self.assertNotIn("2001", p)
        self.assertNotIn("Evidence:", build_prompt("Does X help Y?", []))

    def test_verdict_parsing(self):
        self.assertEqual(parse_verdict("VERDICT: SUPPORTED\nbecause"), "SUPPORTED")
        self.assertEqual(parse_verdict("verdict: **Not  enough information**"), "NOT ENOUGH INFORMATION")
        self.assertEqual(parse_verdict("VERDICT: REFUTED"), "REFUTED")
        self.assertIsNone(parse_verdict("The evidence supports it."))   # never guessed
        self.assertIsNone(parse_verdict(None))
    def test_label_parsing_of_the_judge_reply(self):
        self.assertEqual(parse_label("LABEL: not  enough information"), "NOT ENOUGH INFORMATION")
        self.assertEqual(parse_label("label: **Refuted**"), "REFUTED")
        self.assertEqual(parse_label("LABEL: SUPPORTED\nbecause"), "SUPPORTED")
        self.assertIsNone(parse_label("I think it works"))                 # never guessed
        self.assertIsNone(parse_label("VERDICT: SUPPORTED"))               # a different key is not a label
        self.assertIsNone(parse_label(None))


if __name__ == "__main__":
    unittest.main()

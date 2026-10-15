"""Acceptance tests for docs/methodology.md section 3 (rationale query). They skip until src/rationale_query exists."""


from tests.spec_targets import RATIONALE_PROMPT, SpecTestCase, resolve

QUESTION = (
    "A 72-year-old woman has progressive memory loss over two years. Which of the following is the most "
    "likely diagnosis?\nA) Delirium B) Alzheimer disease C) Hypothyroidism D) Major depression"
)
RATIONALE = (
    "The patient is elderly with gradual, progressive memory decline.\nDelirium is acute, so it is unlikely.\n"
    "The answer is B) Alzheimer disease."
)


def squeeze(text):
    return " ".join(text.split())


class RationalePromptTests(SpecTestCase):
    """Paper p.4: the chain-of-thought prompt of Kim et al., with the question at the end."""

    requires = ("build_rationale_prompt",)

    @staticmethod
    def build(question):
        return resolve("build_rationale_prompt")(question)

    def test_the_prompt_is_the_papers_text_followed_by_the_question(self):
        self.assertEqual(squeeze(self.build(QUESTION)), squeeze(RATIONALE_PROMPT + " " + QUESTION))

    def test_the_question_is_given_unchanged_at_the_end(self):
        prompt = self.build(QUESTION)
        self.assertTrue(prompt.rstrip().endswith(QUESTION))

    def test_only_the_question_differs_between_prompts(self):
        other = "Which vitamin deficiency causes subacute combined degeneration? A) B12 B) C C) D D) K"
        first, second = self.build(QUESTION), self.build(other)
        self.assertEqual(first[: first.rindex(QUESTION)], second[: second.rindex(other)])

    def test_the_prompt_is_deterministic(self):
        self.assertEqual(self.build(QUESTION), self.build(QUESTION))

    def test_an_empty_question_is_an_error(self):
        for bad in ("", "   \n"):
            with self.subTest(question=bad), self.assertRaises(ValueError):
                self.build(bad)


class DenseQueryTests(SpecTestCase):
    """Paper p.4-5: the rationale alone is the dense query; the question is not added."""

    requires = ("dense_query",)

    @staticmethod
    def query(rationale):
        return resolve("dense_query")(rationale)

    def test_the_query_is_the_rationale(self):
        self.assertEqual(self.query(RATIONALE), RATIONALE)

    def test_surrounding_whitespace_is_removed_and_the_inside_is_kept(self):
        self.assertEqual(self.query("  \n" + RATIONALE + "\n\n "), RATIONALE)

    def test_no_separator_marker_is_added(self):
        self.assertNotIn("[SEP]", self.query("First sentence. Second sentence. Third sentence."))

    def test_an_empty_rationale_is_an_error_not_a_fallback(self):
        for bad in ("", "  ", "\n\t\n"):
            with self.subTest(rationale=bad), self.assertRaises(ValueError):
                self.query(bad)
        with self.assertRaises((TypeError, ValueError)):
            self.query(None)


class RerankQueryTests(SpecTestCase):
    """Paper p.5 scores against the original question; the released code scores against the rationale."""

    requires = ("rerank_query",)

    @staticmethod
    def query(**kwargs):
        return resolve("rerank_query")(question=QUESTION, rationale=RATIONALE, **kwargs)

    def test_the_default_is_the_original_question(self):
        self.assertEqual(self.query(), QUESTION)

    def test_the_question_variant_is_the_original_question(self):
        self.assertEqual(self.query(against="question"), QUESTION)

    def test_the_rationale_variant_is_the_rationale(self):
        self.assertEqual(self.query(against="rationale"), RATIONALE)

    def test_surrounding_whitespace_is_removed(self):
        query = resolve("rerank_query")(question="  " + QUESTION + "\n", rationale=" " + RATIONALE + "  ")
        self.assertEqual(query, QUESTION)

    def test_an_unknown_variant_is_an_error(self):
        for bad in ("both", "Question", "", None):
            with self.subTest(against=bad), self.assertRaises(ValueError):
                self.query(against=bad)

    def test_the_text_that_is_used_must_not_be_empty(self):
        rerank_query = resolve("rerank_query")
        with self.assertRaises(ValueError):
            rerank_query(question="  ", rationale=RATIONALE)
        with self.assertRaises(ValueError):
            rerank_query(question=QUESTION, rationale="  ", against="rationale")

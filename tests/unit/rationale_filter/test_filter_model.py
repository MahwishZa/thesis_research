"""rationale_filter.filter_model without torch: windows, averaging, the label tokens, with a stand-in tokenizer."""

import unittest

from rationale_filter import filter_model as fm
from rationale_filter import inference


class Tokenizer:
    """Records how it is called. Texts become one id per word; over ``limit`` ids a text is cut into overlapping windows."""

    model_max_length = 512
    unk_token_id = 2
    eos_token_id = 1

    def __init__(self, ids=None, limit=6):
        self.calls = []
        self.ids = ids if ids is not None else {fm.TOKEN_HELPFUL: 50, fm.TOKEN_NOT_HELPFUL: 51}
        self.limit = limit

    def __call__(self, texts, **options):
        self.calls.append((list(texts), options))
        stride = options["stride"]
        ids, owners = [], []
        for owner, text in enumerate(texts):
            words = [len(w) + 10 for w in text.split()]
            if len(words) + 1 <= self.limit:
                ids.append([*words, 1])
                owners.append(owner)
                continue
            body = self.limit - 1
            start = 0
            while True:
                ids.append([*words[start : start + body], 1])
                owners.append(owner)
                if start + body >= len(words):
                    break
                start += body - stride
        return {"input_ids": ids, "overflow_to_sample_mapping": owners}

    def convert_tokens_to_ids(self, token):
        return self.ids.get(token, self.unk_token_id)


class WindowTests(unittest.TestCase):
    def test_a_text_that_fits_is_one_window_and_is_stripped_first(self):
        tokenizer = Tokenizer()
        windows, owners = fm.tokenize_windows(tokenizer, ["  one two  \n"])
        self.assertEqual((windows, owners), ([[13, 13, 1]], [0]))
        self.assertEqual(tokenizer.calls[0][0], ["one two"])

    def test_the_options_are_the_authors(self):
        tokenizer = Tokenizer()
        fm.tokenize_windows(tokenizer, ["a"])
        self.assertEqual(tokenizer.calls[0][1], {"truncation": True, "max_length": 512, "stride": 128, "return_overflowing_tokens": True})

    def test_a_long_text_gives_windows_that_overlap_and_all_end_with_the_end_token(self):
        tokenizer = Tokenizer(limit=6)
        windows, owners = fm.tokenize_windows(tokenizer, ["w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11", "short"], max_seq_length=8, doc_stride=3)
        self.assertEqual(owners.count(0), len(windows) - 1)
        self.assertEqual(owners[-1], 1)
        self.assertTrue(all(window[-1] == 1 for window in windows))
        self.assertGreater(owners.count(0), 1)

    def test_the_tokenizers_own_limit_wins_when_it_is_smaller(self):
        tokenizer = Tokenizer()
        tokenizer.model_max_length = 100
        self.assertEqual(fm.effective_length(tokenizer, 512), 100)
        tokenizer.model_max_length = 10**30  # the value some tokenizers use for "no limit"
        self.assertEqual(fm.effective_length(tokenizer, 512), 512)

    def test_bad_lengths_and_strides_are_refused(self):
        for bad in (4, 0, True, 7.5):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                fm.effective_length(Tokenizer(), bad)
        for stride in (-1, 510, True):
            with self.subTest(stride=stride), self.assertRaises(ValueError):
                fm.check_stride(512, stride)

    def test_an_empty_text_is_an_error_and_no_texts_give_no_windows(self):
        with self.assertRaises(ValueError):
            fm.tokenize_windows(Tokenizer(), ["fine", "  "])
        self.assertEqual(fm.tokenize_windows(Tokenizer(), []), ([], []))


class AverageTests(unittest.TestCase):
    def test_the_logits_of_the_windows_of_a_text_are_averaged(self):
        rows = [(1.0, 3.0), (3.0, 5.0), (10.0, 0.0)]
        self.assertEqual(fm.mean_over_windows(rows, [0, 0, 1], 2), [(2.0, 4.0), (10.0, 0.0)])

    def test_a_text_without_a_window_is_an_error(self):
        with self.assertRaises(fm.FilterModelError):
            fm.mean_over_windows([(1.0, 2.0)], [0], 2)

    def test_the_order_of_the_windows_in_the_list_does_not_matter(self):
        rows = [(1.0, 0.0), (5.0, 2.0), (3.0, 4.0)]
        self.assertEqual(fm.mean_over_windows(rows, [0, 1, 0], 2), fm.mean_over_windows([rows[2], rows[1], rows[0]], [0, 1, 0], 2))


class TokenIdTests(unittest.TestCase):
    def test_the_two_ids_are_returned_in_the_order_helpful_then_not_helpful(self):
        self.assertEqual(fm.label_token_ids(Tokenizer()), (50, 51))

    def test_an_unknown_token_is_refused_with_the_advice_to_prepare_the_model(self):
        with self.assertRaises(fm.FilterModelError) as caught:
            fm.label_token_ids(Tokenizer(ids={fm.TOKEN_HELPFUL: 50}))
        self.assertIn("prepare-model", str(caught.exception))

    def test_two_tokens_with_one_id_are_refused(self):
        with self.assertRaises(fm.FilterModelError):
            fm.label_token_ids(Tokenizer(ids={fm.TOKEN_HELPFUL: 50, fm.TOKEN_NOT_HELPFUL: 50}))


class BatchDecisionTests(unittest.TestCase):
    def test_one_call_judges_every_snippet_from_its_own_input_in_order(self):
        seen = []

        def judge_many(inputs):
            seen.append(list(inputs))
            return [("first" in text) for text in inputs]

        result = inference.decisions_for(question="Which?", snippets=["first snippet", "second snippet"], judge_many=judge_many)
        self.assertEqual(result, [True, False])
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0], [inference.build_filter_input(evidence=s, question="Which?") for s in ("first snippet", "second snippet")])

    def test_no_snippets_means_no_call(self):
        self.assertEqual(inference.decisions_for(question="Which?", snippets=[], judge_many=lambda inputs: 1 / 0), [])

    def test_a_count_that_does_not_match_is_an_error_and_the_results_are_plain_bools(self):
        with self.assertRaises(ValueError):
            inference.decisions_for(question="Which?", snippets=["a", "b"], judge_many=lambda inputs: [True])
        result = inference.decisions_for(question="Which?", snippets=["a", "b"], judge_many=lambda inputs: [1, 0])
        self.assertTrue(all(type(flag) is bool for flag in result))


if __name__ == "__main__":
    unittest.main()

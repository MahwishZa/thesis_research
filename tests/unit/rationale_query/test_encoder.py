"""Tests of src/rationale_query/encoder.py with a stand-in tokenizer and model. They need numpy and skip without it."""

import sys
import types
import unittest
from unittest import mock

from tests.support import FakeModel, FakeTokenizer, RecordingNoGrad, np

try:
    from rationale_query.encoder import ENCODER_NAME, MAX_LENGTH, POOLING, QueryEncoder
except ImportError:  # numpy is not installed
    QueryEncoder = None


@unittest.skipIf(QueryEncoder is None, "numpy is not installed")
class QueryEncoderTests(unittest.TestCase):
    def make(self, **kwargs):
        self.tokenizer, self.model = FakeTokenizer(), FakeModel()
        return QueryEncoder(self.tokenizer, self.model, **kwargs)

    def test_the_first_token_vector_is_kept_for_every_text(self):
        # "a" and "b" are the words 107 and 108; the stand-in model puts its summary of the text at position 0 and
        # negative numbers at every other position, so any other pooling would show.
        vectors = self.make().encode(["a b"])
        np.testing.assert_array_equal(vectors, [[218, 4, 108, 1, 107, 108]])

    def test_one_float32_vector_per_text_in_the_order_given(self):
        texts = ["a b", "a", "b a b a"]
        vectors = self.make().encode(texts)
        self.assertEqual(vectors.shape, (3, 6))
        self.assertEqual(vectors.dtype, np.float32)
        np.testing.assert_array_equal(vectors[:, 1], [4, 3, 6])  # tokens per text, special tokens included

    def test_the_vectors_are_not_normalised(self):
        vectors = self.make().encode(["a b", "a b a b a b"])
        self.assertTrue(np.all(np.linalg.norm(vectors, axis=1) > 10))

    def test_the_tokenizer_is_called_as_in_the_authors_code(self):
        self.make().encode(["a b"])
        self.assertEqual(
            self.tokenizer.calls,
            [{"texts": ["a b"], "truncation": True, "padding": True, "return_tensors": "pt", "max_length": 512}],
        )
        self.assertEqual(MAX_LENGTH, 512)

    def test_one_text_at_a_time_by_default(self):
        self.make().encode(["a", "b", "c"])
        self.assertEqual([call["texts"] for call in self.tokenizer.calls], [["a"], ["b"], ["c"]])
        self.assertEqual(self.model.calls, 3)

    def test_a_larger_batch_gives_the_same_vectors(self):
        texts = ["a b", "a", "b a b a", "a b c d e f g", "c"]
        single = self.make().encode(texts)
        batched = self.make(batch_size=2).encode(texts)
        self.assertEqual([len(call["texts"]) for call in self.tokenizer.calls], [2, 2, 1])
        np.testing.assert_allclose(batched, single)

    def test_the_length_limit_reaches_the_tokenizer(self):
        encoder = self.make(max_length=8)
        encoder.encode([" ".join("word" for _ in range(20))])
        self.assertEqual(self.tokenizer.calls[0]["max_length"], 8)
        np.testing.assert_array_equal(encoder.encode(["a b"])[:, 1], [4])

    def test_a_text_longer_than_the_limit_is_cut_with_its_end_marker_kept(self):
        encoder = self.make(max_length=8)
        vectors = encoder.encode([" ".join("a" for _ in range(20))])
        self.assertEqual(vectors[0, 1], 8)  # the stand-in tokenizer cuts to max_length tokens

    def test_the_inputs_are_moved_to_the_device(self):
        self.make(device="cuda").encode(["a b"])
        input_ids, attention_mask = self.model.last_inputs
        self.assertEqual(input_ids.moved_to, ["cuda"])
        self.assertEqual(attention_mask.moved_to, ["cuda"])

    def test_gradients_are_switched_off_around_every_call_of_the_model(self):
        no_grad = RecordingNoGrad()
        encoder = self.make(batch_size=2, no_grad=no_grad)
        encoder.encode(["a", "b", "c"])
        self.assertEqual((no_grad.entered, no_grad.left), (2, 2))

    def test_empty_input_is_refused_before_the_model_is_used(self):
        encoder = self.make()
        for texts in ([], [""], ["a", "   "], ["a", None], ["a", 5], ["\n\t"]):
            with self.subTest(texts=texts), self.assertRaises(ValueError):
                encoder.encode(texts)
        self.assertEqual(self.model.calls, 0)

    def test_no_text_at_all_is_refused_with_a_message_that_says_so(self):
        with self.assertRaises(ValueError) as caught:
            self.make().encode([])
        self.assertIn("no texts to encode", str(caught.exception))

    def test_the_position_of_the_bad_text_is_named(self):
        with self.assertRaises(ValueError) as caught:
            self.make().encode(["a", "b", ""])
        self.assertIn("text 2", str(caught.exception))

    def test_counting_tokens_includes_the_special_tokens(self):
        self.assertEqual(self.make().token_counts(["a b c", "a"]), [5, 3])

    def test_the_settings_are_validated(self):
        for kwargs in ({"batch_size": 0}, {"batch_size": -1}, {"batch_size": "2"}, {"max_length": 2}, {"max_length": "512"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                QueryEncoder(FakeTokenizer(), FakeModel(), **kwargs)

    def test_the_info_names_everything_that_shapes_the_vectors(self):
        info = self.make(device="cpu", name="some/encoder", revision="abc123").info
        self.assertEqual(
            info,
            {
                "encoder": "some/encoder",
                "revision": "abc123",
                "max_length": 512,
                "pooling": POOLING,
                "normalised": False,
                "dtype": "float32",
                "device": "cpu",
                "batch_size": 1,
            },
        )
        self.assertEqual(ENCODER_NAME, "ncbi/MedCPT-Query-Encoder")


@unittest.skipIf(QueryEncoder is None, "numpy is not installed")
class FromPretrainedTests(unittest.TestCase):
    """The loading code with stand-ins for torch and transformers; the real ones are used in test_real_libraries."""

    def load(self, *, cuda, commit="0123456789abcdef0123456789abcdef01234567", **kwargs):
        calls, made = [], []

        class Model(FakeModel):
            def __init__(self):
                super().__init__()
                self.config = types.SimpleNamespace(_commit_hash=commit)
                self.evaluated, self.moved_to = False, None

            def eval(self):
                self.evaluated = True
                return self

            def to(self, device):
                self.moved_to = device
                return self

        class AutoTokenizer:
            @staticmethod
            def from_pretrained(name, **options):
                calls.append(("tokenizer", name, options))
                return FakeTokenizer()

        class AutoModel:
            @staticmethod
            def from_pretrained(name, **options):
                calls.append(("model", name, options))
                made.append(Model())
                return made[-1]

        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(is_available=lambda: cuda)
        torch.no_grad = RecordingNoGrad()
        transformers = types.ModuleType("transformers")
        transformers.AutoModel, transformers.AutoTokenizer = AutoModel, AutoTokenizer
        with mock.patch.dict(sys.modules, {"torch": torch, "transformers": transformers}):
            encoder = QueryEncoder.from_pretrained("some/encoder", **kwargs)
        return encoder, made[0], calls, torch

    def test_the_device_is_the_graphics_card_when_there_is_one_and_the_processor_otherwise(self):
        for cuda, expected in ((True, "cuda"), (False, "cpu")):
            with self.subTest(cuda=cuda):
                encoder, model, _, _ = self.load(cuda=cuda)
                self.assertEqual((encoder.device, model.moved_to), (expected, expected))

    def test_a_named_device_is_used_as_it_is(self):
        encoder, model, _, _ = self.load(cuda=True, device="cpu")
        self.assertEqual((encoder.device, model.moved_to), ("cpu", "cpu"))

    def test_the_model_is_put_in_evaluation_mode(self):
        _, model, _, _ = self.load(cuda=False)
        self.assertTrue(model.evaluated)

    def test_the_name_the_revision_and_the_offline_switch_reach_both_loaders(self):
        _, _, calls, _ = self.load(cuda=False, revision="main", local_files_only=True)
        expected = {"revision": "main", "local_files_only": True}
        self.assertEqual(calls, [("tokenizer", "some/encoder", expected), ("model", "some/encoder", expected)])
        _, _, calls, _ = self.load(cuda=False)
        self.assertEqual(calls[0][2], {"revision": None, "local_files_only": False})

    def test_the_revision_in_use_is_the_commit_of_the_files_that_were_read(self):
        encoder, _, _, _ = self.load(cuda=False, revision="main")
        self.assertEqual(encoder.info["revision"], "0123456789abcdef0123456789abcdef01234567")
        encoder, _, _, _ = self.load(cuda=False, revision="main", commit=None)
        self.assertEqual(encoder.info["revision"], "main")

    def test_gradients_are_switched_off_with_torchs_own_context(self):
        encoder, _, _, torch = self.load(cuda=False)
        encoder.encode(["a b"])
        self.assertEqual((torch.no_grad.entered, torch.no_grad.left), (1, 1))

    def test_the_settings_are_passed_on(self):
        encoder, _, _, _ = self.load(cuda=False, batch_size=3)
        self.assertEqual((encoder.batch_size, encoder.name), (3, "some/encoder"))


if __name__ == "__main__":
    unittest.main()

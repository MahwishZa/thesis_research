"""The hf_local adapter with stand-ins for PyTorch and transformers (no model, no GPU, no download, no token)."""

import contextlib
import io
import sys
import types
import unittest
from unittest import mock

from PIL import Image

from src import models

EOS = [1, 106]  # Gemma 3: <eos> and <end_of_turn>
COMMIT = "0123456789abcdef0123456789abcdef01234567"
CACHED = f"/cache/models--org--model/snapshots/{COMMIT}/config.json"
SETTINGS = {"id": "org/model", "revision": "main", "dtype": "bfloat16", "seed": 7, "max_new_tokens": 5,
            "gpu_memory_gib": 3.0, "cpu_memory_gib": 10.0}


class FakeIds(list):
    shape = (1, 300)


class FakeInputs(dict):
    def __init__(self):
        super().__init__(input_ids=FakeIds([0] * 300), pixel_values="pixels")
        self.moved = None

    def to(self, device, dtype=None):
        self.moved = (device, dtype)
        return self


class FakeModel:
    device = "cuda:0"
    generation_config = types.SimpleNamespace(eos_token_id=EOS)
    hf_device_map = {"vision": 0, "layers.0": 0, "layers.1": "cpu", "layers.2": "cpu"}

    def __init__(self, new_ids, error=None):
        self.new_ids, self.error, self.calls, self.evaluated = new_ids, error, [], False

    def eval(self):
        self.evaluated = True

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return [[0] * 300 + self.new_ids]


class FakeProcessor:
    def __init__(self, text="Melanoma"):
        self.text, self.messages = text, []

    def apply_chat_template(self, messages, **kwargs):
        self.messages.append((messages, kwargs))
        return FakeInputs()

    def decode(self, ids, skip_special_tokens=False):
        self.decoded = (list(ids), skip_special_tokens)
        return f"  {self.text}\n"


def png() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGBA", (10, 8), (1, 2, 3, 4)).save(buffer, "PNG")
    return buffer.getvalue()


class HFLocalTests(unittest.TestCase):
    def _load(self, settings=None, *, new_ids=(10, 11, 1), cuda=True, text="Melanoma", model_error=None, load_error=None,
              cached=CACHED):
        self.seeds, self.loads, self.cache_lookups, self.stderr = [], [], [], io.StringIO()
        torch = types.ModuleType("torch")
        torch.__version__, torch.bfloat16, torch.float16, torch.float32 = "9.9.9", "BF16", "F16", "F32"
        torch.cuda = types.SimpleNamespace(is_available=lambda: cuda)
        torch.manual_seed = self.seeds.append
        torch.inference_mode = contextlib.nullcontext
        self.model, self.processor = FakeModel(list(new_ids), model_error), FakeProcessor(text)

        def from_pretrained(name, **kwargs):
            self.loads.append((name, kwargs))
            if load_error:
                raise load_error
            return self.model

        transformers = types.ModuleType("transformers")
        transformers.__version__ = "5.0.0"
        transformers.AutoModelForImageTextToText = types.SimpleNamespace(from_pretrained=from_pretrained)
        transformers.AutoProcessor = types.SimpleNamespace(from_pretrained=lambda name, **kwargs: self.processor)
        hub = types.ModuleType("huggingface_hub")
        hub.try_to_load_from_cache = lambda repo, filename, revision=None: self.cache_lookups.append((repo, filename, revision)) or cached
        modules = {"torch": torch, "transformers": transformers, "huggingface_hub": hub}
        with mock.patch.dict(sys.modules, modules), contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(self.stderr):
            return models.load("hf_local", SETTINGS | (settings or {}))

    def test_the_model_is_loaded_in_its_own_precision_and_split_between_gpu_and_ram(self):
        adapter = self._load()
        name, kwargs = self.loads[0]
        self.assertEqual((name, kwargs["revision"], kwargs["dtype"], kwargs["device_map"]), ("org/model", "main", "BF16", "auto"))
        self.assertEqual(kwargs["max_memory"], {0: "3.0GiB", "cpu": "10.0GiB"})
        self.assertTrue(self.model.evaluated)
        self.assertEqual(adapter.describe()["layers_per_device"], {"0": 2, "cpu": 2})

    def test_without_a_gpu_everything_runs_on_the_cpu(self):
        self._load(cuda=False)
        kwargs = self.loads[0][1]
        self.assertEqual(kwargs["device_map"], "cpu")
        self.assertNotIn("max_memory", kwargs)

    def test_the_picture_and_the_prompt_are_sent_and_decoding_is_greedy(self):
        adapter = self._load()
        answer = adapter.predict(png(), "image/png", "Which one?")
        (messages, kwargs), = self.processor.messages
        (turn,) = messages
        picture = turn["content"][0]["image"]
        self.assertEqual((turn["role"], turn["content"][0]["type"], picture.mode, picture.size), ("user", "image", "RGB", (10, 8)))
        self.assertEqual(turn["content"][1], {"type": "text", "text": "Which one?"})
        self.assertEqual((kwargs["add_generation_prompt"], kwargs["return_dict"], kwargs["tokenize"]), (True, True, True))
        sent = self.model.calls[0]
        self.assertEqual((sent["do_sample"], sent["max_new_tokens"]), (False, 5))
        self.assertEqual(self.seeds, [7])
        self.assertEqual(self.processor.decoded, ([10, 11, 1], True))  # only the new tokens are decoded
        self.assertEqual((answer["text"], answer["finish_reason"], answer["model_version"]), ("Melanoma", "STOP", f"org/model@{COMMIT}"))
        self.assertEqual(answer["usage"], {"prompt_tokens": 300, "output_tokens": 3, "total_tokens": 303})
        self.assertIsNone(answer["response_id"])

    def test_how_an_answer_ended(self):
        cut = self._load(new_ids=(10, 11, 12, 13, 14)).predict(png(), "image/png", "?")  # limit reached, no end token
        self.assertEqual(cut["finish_reason"], "MAX_TOKENS")
        short = self._load(new_ids=(10, 11)).predict(png(), "image/png", "?")  # stopped early without an end token
        self.assertEqual(short["finish_reason"], "OTHER")
        empty = self._load(new_ids=(1,), text="").predict(png(), "image/png", "?")
        self.assertEqual((empty["text"], empty["finish_reason"]), (None, "STOP"))

    def test_running_out_of_memory_and_unreadable_images_are_not_retried(self):
        out_of_memory = self._load(model_error=RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))
        with self.assertRaisesRegex(models.FatalError, "gpu_memory_gib"):
            out_of_memory.predict(png(), "image/png", "?")
        other = self._load(model_error=RuntimeError("device-side assert"))
        with self.assertRaisesRegex(models.FatalError, "generation failed"):
            other.predict(png(), "image/png", "?")
        with self.assertRaisesRegex(models.FatalError, "could not be opened"):
            self._load().predict(b"not an image", "image/png", "?")

    def test_a_refused_download_says_how_to_get_access(self):
        with self.assertRaisesRegex(ValueError, "HF_TOKEN"):
            self._load(load_error=OSError("You are trying to access a gated repo. 401 Client Error"))
        with self.assertRaisesRegex(ValueError, "HF_TOKEN"):  # not every version raises an OSError for a refused download
            self._load(load_error=LookupError("401 Client Error: Unauthorized for url https://huggingface.co/org/model"))
        with self.assertRaisesRegex(ValueError, r"could not be loaded \(OSError: no space left"):
            self._load(load_error=OSError("no space left on device"))

    def test_settings_are_checked_with_the_name_of_the_wrong_one(self):
        for key in ("id", "dtype", "seed", "max_new_tokens"):
            with self.subTest(key), self.assertRaisesRegex(ValueError, f"missing {key}"):
                models.load("hf_local", {k: v for k, v in SETTINGS.items() if k != key})
        for bad in ({"dtype": "int4"}, {"seed": True}, {"max_new_tokens": 0}, {"gpu_memory_gib": -1}, {"cpu_memory_gib": None},
                    {"id": ""}):
            with self.subTest(bad), self.assertRaises(ValueError):
                self._load(bad)

    def test_without_pytorch_the_setup_is_explained(self):
        with mock.patch.dict(sys.modules, {"torch": None}), self.assertRaisesRegex(ValueError, "install PyTorch from pytorch.org"):
            models.load("hf_local", SETTINGS)

    def test_a_missing_helper_package_is_explained_instead_of_a_traceback(self):
        with self.assertRaisesRegex(ValueError, r"accelerate.*pip install -r requirements.txt"):
            self._load(load_error=ImportError("Using a `device_map` requires accelerate"))

    def test_the_run_card_records_what_was_asked_and_what_was_loaded(self):
        description = self._load().describe()
        self.assertEqual((description["model_id"], description["revision_requested"], description["revision_loaded"]),
                         ("org/model", "main", COMMIT))
        self.assertEqual((description["sdk_version"], description["torch_version"]), ("5.0.0", "9.9.9"))
        self.assertEqual(description["generation_config"], {"do_sample": False, "max_new_tokens": 5, "seed": 7, "dtype": "bfloat16",
                                                            "max_memory": {0: "3.0GiB", "cpu": "10.0GiB"}})

    def test_the_commit_that_was_loaded_is_read_from_the_download_cache_and_named_by_the_check(self):
        adapter = self._load()  # revision "main" is not a commit: the cache says which commit it is
        self.assertEqual(self.cache_lookups, [("org/model", "config.json", "main")])
        self.assertIn(f"loaded at commit {COMMIT}", adapter.loaded_note)
        self.assertEqual(self.stderr.getvalue(), "")

    def test_a_pinned_commit_is_used_as_it_is(self):
        adapter = self._load({"revision": COMMIT})
        self.assertEqual((self.cache_lookups, self.loads[0][1]["revision"], adapter.describe()["revision_loaded"]),
                         ([], COMMIT, COMMIT))

    def test_an_unknown_commit_is_reported_loudly_and_the_requested_revision_is_recorded(self):
        adapter = self._load(cached=None)
        self.assertIn("WARNING: the commit of the model files could not be determined", self.stderr.getvalue())
        self.assertIsNone(adapter.describe()["revision_loaded"])
        self.assertIn("UNKNOWN", adapter.loaded_note)
        self.assertEqual(adapter.predict(png(), "image/png", "?")["model_version"], "org/model@main")

    def test_the_packages_whose_versions_are_recorded_include_pytorch(self):
        from src.models import hf_local

        self.assertIn("torch", hf_local.HFLocalModel.packages)
        self.assertIn("transformers", hf_local.HFLocalModel.packages)


if __name__ == "__main__":
    unittest.main()

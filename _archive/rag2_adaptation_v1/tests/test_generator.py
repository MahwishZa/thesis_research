"""An output file is bound to the generator that produced it: the recorded configuration (no model)."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.medchange import arms as A
from experiments.medchange.generator import RESULT_RELEVANT, check_config, config_path, run_config


class RunConfigTests(unittest.TestCase):
    """An answers file is bound to the generator that produced it: a resumed run with a different
    model or result-relevant setting is refused instead of silently mixing configurations."""

    KW = dict(model_path="models/m.gguf", model_sha256="a" * 64, n_ctx=4096, max_new_tokens=160,
              n_threads=8, n_gpu_layers=0, llama_version="0.2.90")

    def test_the_config_records_model_decoding_prompt_and_arm_settings(self):
        cfg = run_config(**self.KW)
        self.assertEqual(cfg["model_file"], "m.gguf")
        self.assertEqual((cfg["temperature"], cfg["seed"]), (0.0, 42))
        self.assertEqual(cfg["arm_settings"], A.settings_hash())
        for key in RESULT_RELEVANT:
            self.assertIn(key, cfg)

    def test_the_first_run_records_the_configuration_and_a_rerun_is_compatible(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "answers_dev.jsonl"
            self.assertEqual(check_config(out, run_config(**self.KW)), [])
            self.assertTrue(config_path(out).exists())
            self.assertEqual(config_path(out).name, "answers_dev.config.json")
            self.assertEqual(check_config(out, run_config(**self.KW)), [])

    def test_a_different_model_or_result_relevant_setting_is_refused(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "answers_dev.jsonl"
            check_config(out, run_config(**self.KW))
            self.assertEqual(check_config(out, run_config(**dict(self.KW, model_sha256="b" * 64))),
                             ["model_sha256"])
            self.assertEqual(check_config(out, run_config(**dict(self.KW, n_ctx=2048,
                                                                 max_new_tokens=256))),
                             ["n_ctx", "max_new_tokens"])

    def test_threads_gpu_layers_and_library_version_are_recorded_but_do_not_block_a_resume(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "answers_dev.jsonl"
            check_config(out, run_config(**self.KW))
            changed = run_config(**dict(self.KW, n_threads=2, n_gpu_layers=10, llama_version="9.9"))
            self.assertEqual(check_config(out, changed), [])
            recorded = json.loads(config_path(out).read_text(encoding="utf-8"))
        self.assertEqual((recorded["n_threads"], recorded["llama_cpp_python"]), (8, "0.2.90"))


if __name__ == "__main__":
    unittest.main()

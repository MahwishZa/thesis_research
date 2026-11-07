"""Reading API keys from a local .env file (no network, no real keys)."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src import models


class LoadEnvTests(unittest.TestCase):
    def _load(self, content, existing=None):
        """Loads ``content`` as a .env file into an otherwise empty environment and returns that environment."""
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, existing or {}, clear=True):
            path = Path(d) / ".env"
            path.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
            models.load_env(path)
            return dict(os.environ)

    def test_reads_names_and_values(self):
        env = self._load('# keys\n\nEXAMPLE_API_KEY = abc123\r\nOTHER="quoted value"\nSINGLE=\'x\'\nno equals sign\nEMPTY=\n')
        self.assertEqual(env, {"EXAMPLE_API_KEY": "abc123", "OTHER": "quoted value", "SINGLE": "x"})

    def test_a_byte_order_mark_from_an_editor_is_ignored(self):
        self.assertEqual(self._load(b"\xef\xbb\xbfEXAMPLE_API_KEY=abc"), {"EXAMPLE_API_KEY": "abc"})

    def test_a_key_already_set_in_the_terminal_wins(self):
        self.assertEqual(self._load("EXAMPLE_API_KEY=from-file", {"EXAMPLE_API_KEY": "from-terminal"}),
                         {"EXAMPLE_API_KEY": "from-terminal"})

    def test_a_missing_file_is_fine_and_a_wrongly_encoded_file_is_explained(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            models.load_env(Path(tempfile.gettempdir()) / "no-such-folder" / ".env")
            self.assertEqual(dict(os.environ), {})
        with self.assertRaises(ValueError):  # Windows PowerShell's ">" saves UTF-16
            self._load("EXAMPLE_API_KEY=abc".encode("utf-16"))


class LoadTests(unittest.TestCase):
    def test_a_provider_without_an_adapter_is_refused_with_an_explanation(self):
        with self.assertRaisesRegex(ValueError, r"unknown model provider: 'nothing' \(adapters: hf_local\)"):
            models.load("nothing", {})


if __name__ == "__main__":
    unittest.main()

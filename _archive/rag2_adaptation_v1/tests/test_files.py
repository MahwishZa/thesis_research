"""Reading JSON-lines files and hashing model and data files."""

import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.medchange.files import file_sha256, load_jsonl, text_file_sha256


class LoadJsonlTests(unittest.TestCase):
    def test_a_missing_file_is_an_empty_list_and_blank_lines_are_skipped(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.jsonl"
            self.assertEqual(load_jsonl(path), [])
            path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
            self.assertEqual(load_jsonl(path), [{"a": 1}, {"a": 2}])

    def test_a_truncated_last_line_is_ignored_but_damage_elsewhere_is_not(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.jsonl"
            path.write_text('{"a": 1}\n{"a": 2, "text": "trunc', encoding="utf-8")          # a run killed mid-write
            self.assertEqual(load_jsonl(path), [{"a": 1}])
            path.write_text('{"a": 1}\n{broken\n{"a": 3}\n', encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError):
                load_jsonl(path)


class HashTests(unittest.TestCase):
    def test_file_sha256_streams_the_whole_file(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.bin"
            payload = b"x" * (3 * (1 << 20) + 17)
            path.write_bytes(payload)
            self.assertEqual(file_sha256(path), hashlib.sha256(payload).hexdigest())

    def test_crlf_checkout_hashes_like_lf(self):
        with TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.csv", Path(tmp) / "b.csv"
            a.write_bytes(b"x,y\n1,2\n")
            b.write_bytes(b"x,y\r\n1,2\r\n")
            self.assertEqual(text_file_sha256(a), text_file_sha256(b))
            self.assertNotEqual(file_sha256(a), file_sha256(b))               # the byte hash of a model file does not normalise anything


if __name__ == "__main__":
    unittest.main()

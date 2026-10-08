"""The helpers shared by the phase drivers: the frozen-file guard, a commit that never pushes, the executor, and the
network retry of the evidence download (no model, no network)."""

import contextlib
import io
import subprocess
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.medchange import runner as RN


def git(repo, *args):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=repo,
                          capture_output=True, text=True, check=True)


class FrozenGuardTests(unittest.TestCase):

    def test_a_frozen_file_must_match_origin_main(self):
        with TemporaryDirectory() as tmp:
            repo = Path(tmp)
            git(repo, "init", "-q")
            rel = "results/rag2_design.json"
            self.assertFalse(RN.frozen_is_pushed(rel, repo)[0])
            (repo / "results").mkdir()
            (repo / rel).write_text("{}", encoding="utf-8")
            git(repo, "add", ".")
            git(repo, "commit", "-q", "-m", "x")
            ok, why = RN.frozen_is_pushed(rel, repo)
            self.assertFalse(ok)
            self.assertIn("not on origin/main", why)                           # committed but not pushed
            git(repo, "update-ref", "refs/remotes/origin/main", "HEAD")
            self.assertTrue(RN.frozen_is_pushed(rel, repo)[0])
            (repo / rel).write_text('{"changed": 1}', encoding="utf-8")
            ok, why = RN.frozen_is_pushed(rel, repo)
            self.assertFalse(ok)
            self.assertIn("differs", why)


class CommitTests(unittest.TestCase):

    def test_results_are_committed_but_never_pushed(self):
        with TemporaryDirectory() as tmp:
            repo = Path(tmp)
            git(repo, "init", "-q")
            results = repo / "experiments" / "medchange" / "results"
            results.mkdir(parents=True)
            (results / "a.json").write_text("{}", encoding="utf-8")
            ok, why = RN.commit_results("add results", repo)             # no remote exists: a push would fail
            self.assertTrue(ok)
            self.assertIn("push it yourself", why)
            self.assertEqual(git(repo, "log", "--format=%s").stdout.strip(), "add results")
            ok, why = RN.commit_results("again", repo)
            self.assertEqual((ok, why), (True, "nothing new to commit"))


class ExecuteTests(unittest.TestCase):

    def test_steps_run_in_order_and_the_first_failure_stops_the_run(self):
        calls = []
        steps = [("one", lambda: (calls.append(1) or True, "ok")), ("two", lambda: (calls.append(2) or False, "bad")),
                 ("three", lambda: (calls.append(3) or True, "ok"))]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(RN.execute(steps, dry_run=False), 2)
        self.assertEqual(calls, [1, 2])

    def test_a_dry_run_prints_commands_and_runs_nothing(self):
        calls = []
        steps = [("cmd", RN.py("rag2_run", "lists", "--split", "dev")), ("fn", lambda: calls.append(1) or (True, ""))]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(RN.execute(steps, dry_run=True), 0)
        self.assertEqual(calls, [])
        self.assertIn("experiments.medchange.rag2_run lists --split dev", out.getvalue())

    def test_print_checks_reports_every_row_and_fails_if_any_fails(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            ok, _ = RN.print_checks([("a", True, ""), ("b", False, "1 missing")])
        self.assertFalse(ok)
        self.assertIn("FAIL b 1 missing", out.getvalue())


class NetworkRetryTests(unittest.TestCase):
    """A connection cut in the middle of a download (IncompleteRead) is retried, not fatal."""

    def test_efetch_retries_an_incomplete_read(self):
        import http.client
        from experiments.medchange import freeze_candidates as FC
        calls = []

        def opener(url):
            calls.append(url)
            if len(calls) == 1:
                raise http.client.IncompleteRead(b"partial")
            return b"<PubmedArticleSet></PubmedArticleSet>"

        eu = types.SimpleNamespace(api_key=None, delay=0, _sleep=lambda s: None, _open=opener)
        self.assertEqual(FC.fetch_abstracts(eu, ["1"]), {})
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()

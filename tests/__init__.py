"""Test package.

Redirects corpus-pipeline logging away from the real, git-tracked corpus
logs for the duration of the test run.

Why this is here and not in one test file: ``_common.get_logger()`` resolves
its log directory from the loaded module's own ``__file__``, so **any** test
that calls a corpus stage's ``main()`` in-process against a temp corpus
copy still appends its fixture-scale lines to
``alzheimer_corpus/logs/*.log``. Those files are the corpus's provenance
record - the evidence chain in ``docs/status_and_decisions.md`` §2 is read
out of them - and a test run must not write to them. This was happening:
running the suite appended Stage-06 whitespace-tokenizer lines, naming
``/tmp`` paths, to the tracked ``quality_control.log``.

Setting the variable once here covers in-process calls and subprocesses
alike (a child inherits the environment), and a real pipeline run - which
never imports this package - is byte-for-byte unaffected.
"""

import atexit
import os
import tempfile

if not os.environ.get("ALZHEIMER_CORPUS_LOGS"):
    _log_dir = tempfile.mkdtemp(prefix="corpus-test-logs-")
    os.environ["ALZHEIMER_CORPUS_LOGS"] = _log_dir

    @atexit.register
    def _cleanup_test_logs():
        import shutil
        shutil.rmtree(_log_dir, ignore_errors=True)

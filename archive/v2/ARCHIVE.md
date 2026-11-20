# Version 2 (closed-book phase), archived

A frozen copy of the whole research as it stood at commit `95a9506` (2026-11-20), moved here on 2026-11-20 with `git mv`, so the Git
history of every file is kept (`git log --follow <path>`). Nothing was changed, deleted or overwritten. A branch `archive-v2-snapshot`
on GitHub points at that commit.

- **What is here:** `src/` (code), `tests/`, `configs/`, `prompts/`, `data/` (the committed tables, reports and data sheets; the large
  originals were never in Git), `results/` (the finished test runs), `docs/`, `requirements.txt`, and `README.md`, which is an exact
  copy of the README of that commit.
- **What was added:** this file and `MANIFEST.sha256` (a SHA-256 for every other file here). `tests/test_workspace.py` in the active
  workspace checks the manifest and compares the files with commit `95a9506`.
- **Paths:** every path inside the documents, run cards and reports is relative to this folder (`archive/v2/`), as they were relative
  to the repository root before.
- **Running it:** the code works from inside this folder (`cd archive/v2`, then `python -m src.validate`, `python -m unittest tests.test_run`
  and so on). Checked on 2026-11-20: 230 of its 232 tests pass; the other two look for `archive/v1/` next to the code, which is not
  there (`tests.test_archive` and the code-guide test of `tests.test_repository`), so they fail here by design. The large originals of the
  datasets are not in Git, so nothing here can rebuild a dataset without them.
- **Version 1** is in `archive/v1/` and is obsolete.
- This folder is temporary, like `archive/v1/`: it is removed when the research is final.

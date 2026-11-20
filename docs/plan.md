# Plan: the research, step by step

Temporary page. Each step starts only after approval, and its status is changed here when it starts and when it is done. Nothing below
has been started in this workspace. **The earlier work is in `archive/v2/`** (state of commit `95a9506`): use it as a starting point
or a reference, never as a result of this workspace.

Rules for every step (from `.claude/CLAUDE.md` and earlier decisions): automation only (no manual reading of cases); API budget under
10 USD; open-book runs use local open-weight models only; only established, correctly named metrics; every claim marked verified or
assumed; no invented data; no Git tags until the final version is ready.

| # | Step | What it produces | Starting point in the archive | Status |
|---|---|---|---|---|
| 1 | Research question and scope | the question, the claims to test, the datasets needed | `README.md` (overview and objectives) | not started |
| 2 | Datasets | one folder per dataset in `data/`: table, report, data sheet, build records | `archive/v2/data/`, `archive/v2/src/build/` | not started |
| 3 | Questions, prompts and answer format | one prompt per dataset and approach; the answer format | `archive/v2/prompts/`, `archive/v2/docs/method.md` | not started |
| 4 | Models and settings | the model list (pinned versions, pace, seed) | `archive/v2/configs/models.toml` | not started |
| 5 | Scores and statistics | the metrics, intervals and comparisons, each with its source | `archive/v2/src/metrics.py`, `archive/v2/docs/method.md` | not started |
| 6 | Closed-book baselines | one test run per model and dataset, with its report | `archive/v2/results/`, `archive/v2/docs/results.md` | not started |
| 7 | Knowledge base and retrieval | what open-book runs may retrieve; the retriever | `archive/v2/data/knowledge/` | not started |
| 8 | Open-book approaches | standard RAG, RAG² and the proposed method, run with local models | none built | not started |
| 9 | Evidence grounding | the score that checks an answer against its evidence | undecided (`archive/v2/docs/audit.md`, decision D2) | not started |
| 10 | Results and analysis | the tables, comparisons and limits | `archive/v2/docs/results.md` | not started |
| 11 | Freeze with the supervisor | the frozen versions, named by commit | | not started |
| 12 | Final cleanup | remove `.claude/`, `archive/` and `docs/`, and prune them from the Git history (only on explicit go-ahead) | | not started |

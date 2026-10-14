# Archive

This folder holds the previous version of the research. It is kept for reference only. Nothing here is part of the new work.

## What is inside

`rag2_adaptation_v1/` is the earlier implementation and its experiments:

- `experiments/medchange/`: the code that searched PubMed, wrote the answers with a local language model, and measured accuracy. It also holds the small result files it produced.
- `tests/`: the automatic checks for that code.
- `pyproject.toml`: the install list for that code.

## What to know before using it

- It was an adaptation of RAG², not a faithful copy, so its results should not be treated as a reliable baseline.
- Its tests were written for the old folder layout. Some of them (mainly the ones that check the documents) fail from here. This is expected.
- To run it, work inside `rag2_adaptation_v1/`, not from the repository root.
- The old versions of `docs/methodology.md`, `protocol.md`, `data.md`, `evaluation.md` and `reproducibility.md` are in the git history, in commit `70de6bc`.
- The research log, `docs/log.md`, stays in `docs/` and is not copied here.

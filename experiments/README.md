# Experiments

One numbered folder per experiment, for example `01_baseline/`. Each folder holds:

- the run script `run_experiment.py` (this name in every experiment folder) and its settings (including the paths to any large local data),
- `results/` with small result files that are committed,
- `NOTES.md`: what was tried, why, and the exact command to run it again.

Experiments import their code from `src/`; they do not hold copies of it. Large files (corpus, models, big
outputs) stay outside the repository or in git-ignored folders such as `outputs/`.

Experiments so far: none.

`.gitignore` hides every folder named `filters`, `checkpoints` or `local`, wherever it is (they are meant for trained
models and machine-local files), and the large-data folders at the repository root (`models/`, `outputs/`, `corpus/`,
`retrieval_corpora/`, `_archive/`). Git silently leaves out files in those folders, so do not keep files that must be
committed there. To check a path: `git check-ignore -v <path>` prints the rule that hides it, or nothing.

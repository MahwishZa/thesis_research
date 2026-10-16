# Experiments

One numbered folder per experiment, for example `01_baseline/`. Each folder holds:

- the run script `run_experiment.py` (this name in every experiment folder) and its settings (including the paths to any large local data),
- `results/` with small result files that are committed,
- `NOTES.md`: what was tried, why, and the exact command to run it again.

Experiments import their code from `src/`; they do not hold copies of it. Large files (corpus, models, big
outputs) stay outside the repository or in git-ignored folders such as `outputs/`.

Experiments so far:

- `01_rationale_writers/`: the language models that write the rationales of Part 1, compared on one questions file.

Do not name an experiment folder `baseline`, `results`, `medchange`, `data` or `outputs`: older ignore rules in
`.gitignore` hide folders with those names and git would silently leave their files out. Numbered names such as
`01_baseline/` are safe.

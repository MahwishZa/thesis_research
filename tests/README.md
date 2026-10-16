# Tests

Run everything from the repository root: `python -m unittest discover -s tests -t .` (a folder is run with `-s tests/unit`, one file
with its dotted name, for example `python -m unittest tests.unit.test_rationale_store`). Nothing needs to be installed first: the
`tests` package puts `src/` on the search path and defines `ROOT` and `SRC`, which every test imports when it needs a path.

| Folder | What it checks | Needs |
|---|---|---|
| `spec/` | The requirements of `docs/methodology.md`, as acceptance tests (`test_spec_*.py`) that skip until their part is built, and the guard that ties them to the document. `spec_targets.py` maps each requirement to the function that implements it. | nothing |
| `unit/` | `src/rationale_query` module by module, with stand-ins for the model and the encoder (`rationale_support.py`) | nothing; the encoder tests need numpy and skip without it |
| `integration/` | The real llama.cpp and a real torch encoder on tiny synthetic models (`tiny_models.py`), and, only when asked, the real Llama-3 file and MedCPT (`test_rationale_models.py`, see `experiments/01_rationale_writers/NOTES.md`) | llama-cpp-python and gguf, or torch and transformers; a class skips, with the reason, when its packages are missing |
| `experiments/` | The settings and runner of the experiment folders | nothing |
| `repository/` | The layout: folders, package docstrings, light imports, and that the specification names every test file of Part 1 | nothing |

`rationale_support.py` holds what several folders share: the stand-ins, the temporary-folder test case and the command-line runner.
Add a test to the folder that matches its purpose; a new part of the pipeline gets its tests in `unit/` and, if it uses a real library,
in `integration/`.

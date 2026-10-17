# Tests

Run everything from the repository root: `python -m unittest discover -s tests -t .` (a folder is run with `-s tests/unit`, one file
with its dotted name, for example `python -m unittest tests.unit.rationale_query.test_store`). Nothing needs to be installed first: the
`tests` package puts `src/` on the search path and defines `ROOT` and `SRC`, which every test imports when it needs a path.

| Folder | What it checks | Needs |
|---|---|---|
| `spec/` | The requirements of `docs/methodology.md`, as acceptance tests (`test_*.py`) that skip until their part is built, and the guard that ties them to the document. `targets.py` maps each requirement to the function that implements it. | nothing |
| `unit/rationale_query/` | `src/rationale_query` module by module, and the settings and runner of the experiment folders (`test_rationale_writers.py`), with stand-ins for the model and the encoder (`support.py`) | nothing; the encoder tests need numpy and skip without it |
| `unit/balanced_retrieval/` | `src/balanced_retrieval` module by module, with stand-in encoders and a stand-in cross-encoder | nothing; the tests of corpora, vectors and search need numpy and skip without it |
| `integration/` | The real llama.cpp and real torch encoders on tiny synthetic models (`tiny_models.py`; `test_real_libraries.py` for Part 1, `test_real_retrieval.py` for Part 2), and, only when asked, the real Llama-3 file and the MedCPT models (`test_real_models.py`, see `experiments/01_rationale_writers/NOTES.md`) | llama-cpp-python and gguf, or torch and transformers; a class skips, with the reason, when its packages are missing |
| `repository/` | The layout: folders, package docstrings, light imports, and that the specification names every test file of the built parts | nothing |

`support.py` holds what several folders share: the stand-ins, the temporary-folder test case and the command-line runner.
Add a test to the folder that matches its purpose; a new part of the pipeline gets a folder `unit/<package>/` and, if it uses a real library,
a file in `integration/`.

Naming rule: a folder or file name is used for one purpose only. A test folder is named after what it checks, never after a
top-level folder of the repository (there is no `tests/experiments/`, because `experiments/` holds the experiments themselves).

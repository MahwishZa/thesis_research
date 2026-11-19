# Code guide: structure, files and how to run

For supervisor discussions. Temporary: this folder is removed when the research is final (see the README of the repository's owner). Facts here are
checked by `tests/test_repository.py` (the folder tree lists real paths; every command is accepted by the command line).

## Repository Structure

```
thesis_research/
├── src/                 the code
│   ├── build/           how each dataset and the knowledge base are made and checked
│   ├── datasets.py      reads a prepared dataset and checks every file
│   ├── prompts.py       the prompt and the answer reader
│   ├── models.py        one adapter per model API
│   ├── run.py           runs a model on a dataset (python -m src.run)
│   ├── metrics.py       the scores
│   ├── evaluate.py      reports and comparisons from saved answers (python -m src.evaluate)
│   ├── validate.py      checks of the whole pipeline that cost nothing (python -m src.validate)
│   └── common.py        helpers they share
├── configs/             the models and their settings (models.toml)
├── prompts/             one prompt per dataset and approach
├── data/                one folder per dataset and the knowledge base: table, report, data sheet, build records
│   ├── clinic_patients/     PAD-UFES-20: phone photos of skin lesions with each patient's clinic record (Version 2)
│   ├── published_cases/     case reports: each patient's written case, and the same patient's photos from the article (Version 2)
│   └── knowledge/           what open-book runs may retrieve
├── docs/                temporary: the Version 2 plan, the method, this guide, the audit and the progress; removed when the research is final
├── archive/             temporary: Version 1 (commit a1c9209), frozen for reference; removed when the research is final
├── tests/               unit tests
└── requirements.txt
```

## How to Run

```
pip install -r requirements.txt
```

Put the API keys in a file named `.env` next to the README (`CLAUDE_API_KEY=...`, `OPENROUTER_API_KEY=...`). A run is one model of
`configs/models.toml`, one dataset and one approach; its answers and report are written to `results/<dataset>/<approach>/<model>/<run>/`
(the folder appears with the first Version 2 run).

Before any experiment, in this order (the first costs nothing, the second a fraction of a cent):

1. Check the whole pipeline: `python -m src.validate` (add `--files` on the laptop to also hash every photo and text against its table).
2. Check one model: `python -m src.run check --model claude-sonnet-5-5 --dataset published_text`

Then:
- Run the test split: `python -m src.run run --model claude-sonnet-5-5 --dataset published_text --split test`
- Compare two finished runs on the same questions (nothing is sent or changed): python -m src.evaluate compare, then the two run folders
- Prepare the clinic patients dataset (needs the PAD-UFES-20 download in `data/clinic_patients/raw/`; the photos stay on this computer): `python -m src.build.clinic_patients.prepare all`

### Running the closed-book experiments

For one model and one dataset (the examples use `claude-sonnet-5-5` and `published_text`; the other models are in `configs/models.toml`, the other datasets in `src/datasets.py`):

1. `git pull`, then `python -m src.validate --files` on the laptop: every photo and text is the one its table lists, the views are aligned, the prompts agree.
2. `python -m src.run check --model claude-sonnet-5-5 --dataset published_text`: one tiny request, so a wrong key, an unavailable model or a cut-off answer shows before a run.
3. Optional, to look at the answers first: `python -m src.run run --model claude-sonnet-5-5 --dataset published_text --split val --limit 5`. A validation run stays on the laptop (Git ignores it) and is never a result. Read `raw_text` in its `predictions.jsonl` for answers marked invalid. A run also stops by itself if its first 10 answers are all invalid.
4. `python -m src.run run --model claude-sonnet-5-5 --dataset published_text --split test`: the result. It needs a clean Git state (the run card names the code) and a pinned model, and it runs once per model, dataset and approach. If it stops (quota, network), the same command continues it. At the end it checks its own answer file and writes `report.md` and `metrics.json`.
5. Commit the finished folder `results/<dataset>/closed_book/<model>/test_<id>/` (`run_card.json`, `predictions.jsonl`, `metrics.json`, `report.md`, `run.log`). A folder ending in `.partial` is unfinished and is ignored by Git.
6. Compare two finished runs on the same questions with python -m src.evaluate compare, then the two run folders. The three clinic views pair with each other, and so do the three views of the published cases (on the cases that have photos).

Cost control: a run sends one request per question (`requests_per_minute` in the model list paces it). Check one model on one dataset with `--limit` before a full test run, and read the token counts in `predictions.jsonl` (`usage`) to estimate the rest.

## Words used in the builders and data sheets

The builders and the data sheets of the published cases were written before the datasets got their present names and still use the older
words. They mean: **the text dataset** = `published_text` (`data/published_cases/dataset.csv`); **the case dataset** or **the photos of the
cases** = the photo views `published_text_photos` and `published_photos` (`data/published_cases/photos.csv`); **the image dataset** and
**Version 1** = the archived first version in `archive/v1/`; **the clinic patients cohort** = `clinic_patients`.

## The pipeline, stage by stage

| Stage | What happens | Where | Checked by |
|---|---|---|---|
| 1. Data preparation | each dataset is built from its source in numbered steps; each step writes a section of the dataset's report | `src/build/<dataset>/` | each step's own checks and tests; `data/<dataset>/dataset_report.md` |
| 2. Dataset reading | the committed table is read and every file is hashed against it, byte for byte | `datasets.load_dataset` | a run stops on any difference |
| 3. Prompt | one template per dataset and approach; the class names are put in; a case or a record goes in word for word | `prompts.py`, `prompts/` | `validate` (same answer block everywhere); a run's card stores the prompt |
| 4. Model call | one adapter per API; retries, pace, a pinned model version | `models.py`, `run.py` | `run check`; a test run needs a pinned model and a clean Git state |
| 5. Answer reading | strict: one JSON object, the exact keys, three different classes of the list | `prompts.parse_json_answer` | an invalid answer counts as wrong and is counted apart; the first 10 all invalid stop the run |
| 6. Saving | each answer is appended as it arrives; a stopped run resumes | `run.py`, `common.py` | `evaluate.verify_answers` before a run is called finished |
| 7. Scoring | every score is computed from the saved answers only | `metrics.py`, `evaluate.py` | verified answers only; scores checked against a naive computation in the tests |
| 8. Comparison | two runs on the same questions, paired intervals | `evaluate.compare` | same split, same answer options, same questions (or the shared subset), current answer format |

## File by file

### Core (`src/`)

| File | Responsibility |
|---|---|
| `datasets.py` | The six datasets (`DATASETS`): what differs between them (text or image, its group, its facts, its pictures). Reads a table and checks it is one consistent dataset (`read_table`); checks that every file on disk is the one the table lists (`load_dataset`); the answer options (`classes`: the classes of the training split); `question_of`, the id that pairs the views of one question; the hash of a set of questions. |
| `prompts.py` | The prompt template (placeholders checked) and `parse_json_answer`, the strict answer reader: the same for every model, so differences come from the models. |
| `models.py` | One adapter per API: Anthropic and OpenAI-style (GPT-4o, Gemini and Llama through OpenRouter). Each answers one prompt with or without images in the same form; failures are a transient, a quota or a fatal error. No local open-weight adapter exists yet. |
| `run.py` | `check` (everything a run needs, one tiny request) and `run` (one model, one dataset, one approach, the validation or test split): seeded order, saved as it arrives, resumable, run card with the Git commit and package versions, one test run per model, dataset and approach. |
| `metrics.py` | Pure functions: balanced accuracy, macro-F1, macro-averaged top-3 accuracy, coverage, selective risk, precision@k and nDCG@k (for the open-book runs), bootstrap and Wilson intervals, McNemar, Holm. |
| `evaluate.py` | `verify_answers` (is a saved answer file consistent), `report` (a run's scores and report) and `compare` (two runs, paired). |
| `validate.py` | The no-cost checks (`python -m src.validate`): tables, leaks, alignment of the views, prompts, model list, knowledge index. |
| `common.py` | Helpers shared by the builders and the runs: polite download, report sections, tables, crash-safe JSON-lines. |

### Dataset preparation (`src/build/`)

| Folder | Files | What they do |
|---|---|---|
| `clinic_patients/` | `inventory`, `clean`, `dedupe`, `split`, `images`, `views`, `shortcuts`, `prepare`, `table` | PAD-UFES-20 in seven numbered steps: what the download holds; the cleaned clinic record; identical and near-identical photos; the split by patient; the standardised photos; the three views (photo, record, photo + record); the shortcut check. `prepare` runs them in order. |
| `published_cases/` | `text`, `audit`, `photos` | The case reports: survey, cases, abstracts, mesh, build, audit, review, revise (`text`, `audit`); then the photos of the same articles: count, download, check, overlap with the clinic photos, build (`photos`). |
| `knowledge/` | `library`, `index`, `enrich` | What open-book runs may retrieve: the library of passages from four public corpora; the index of items (training cases and library passages); the appearance and aspect tables. |

### Other

| Path | Contents |
|---|---|
| `configs/models.toml` | One entry per model: provider, exact id, settings, pace, seed. A changed setting is a new entry, never an edit. |
| `prompts/<dataset>/closed_book.txt` | The prompt of each dataset (six). The answer-format block is identical in all six (checked by `validate`). |
| `data/<dataset>/` | `dataset.csv` (the table), `dataset_report.md` (what each step found), `datasheet.md` (the data sheet), `build/` (records of each step). |
| `tests/` | Unit tests; `tests/build/` for the builders. |
| `docs/method.md` | The approved method: question, answer format, scores, intervals. |
| `docs/v2_proposal.md` | The Version 2 plan; its sections 4 and 5 are a proposal, only partly approved (see `docs/audit.md`). |
| `docs/audit.md` | The audit of the repository before further experiments. |
| `docs/progress.md` | Current progress; nothing in it is a final result. |

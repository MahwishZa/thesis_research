# Version 1 (archived)

This folder is a frozen copy of Version 1 of the research, kept for reference. It does not bind Version 2 (see
`docs/v2_proposal.md`).

- **Version 1 is commit `a1c9209`** on `main` (2026-11-19). Every file here is a byte-for-byte copy of the same path at that commit.
  The one exception is the "Version 1" label on the first line of the seven hand-written documents listed below.
  `tests/test_archive.py` checks this.
- **The code of Version 1** is not copied. It is in the Git history: `git checkout a1c9209` shows the whole repository as it was,
  code included.
- **What Git never held** (the photos, the case texts with research-only licences, the raw downloads, the validation runs) is not
  here either. It is kept as a ZIP on the researcher's laptop, made at the same commit.
- **Nothing was moved or deleted.** The originals are still in their usual places (`data/`, `results/`, `docs/`, `configs/`,
  `prompts/`) until Version 2 replaces them, with the researcher's approval.

## What is here

| Folder | Contents |
|---|---|
| `data/` | The tables, reports, build records and data sheets of the five Version 1 datasets: image, image + metadata, text (version 3), cases (version 2), and the knowledge base |
| `results/phase1/` | The 12 closed-book test runs of the first phase, with their archived `report.md` and `metrics.json`, and the `scores.md` and `scores.json` written beside them on 2026-11-19 |
| `docs/` | `method.md` (the answer format and metrics) and `results.md` (the first-phase results and their interpretation) |
| `configs/`, `prompts/` | The model list and the prompts as they stood at the commit (the prompts for the JSON answer, not yet used in any run) |

Documents with the "Version 1" label on their first line: `data/image/datasheet.md`, `data/image_metadata/datasheet.md`,
`data/text/datasheet.md`, `data/cases/datasheet.md`, `data/knowledge/datasheet.md`, `docs/method.md`, `docs/results.md`.

Links inside these documents point to other files of this folder where those were copied; links to the code (`src/`) work only
in a checkout of commit `a1c9209`.

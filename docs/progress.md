# Progress (2026-11-19)

Temporary page, removed with `docs/` when the research is final. It records where the work stands. The datasets are not frozen yet;
each run card names the table and its hash that the run used.

## Closed-book results so far

`published_text`, test split (209 cases, 8 classes, chance 12.5%), one run per model. Balanced accuracy is the main score.

| Model | Balanced accuracy | 95% interval | Macro-F1 | Top-3 | Invalid answers |
|---|---|---|---|---|---|
| Claude Sonnet 5.5 | 61.0% | 55.1 to 66.3 | 60.9% | 91.4% | 0 |
| Gemini 3.8 Flash | 59.6% | 53.5 to 65.1 | 58.6% | 91.6% | 0 |
| GPT-4o | 56.1% | 45.5 to 66.3 | 52.8% | 87.5% | 1 |
| Llama 4 Maverick | 41.9% | 35.9 to 47.9 | 41.6% | 83.7% | 4 |

The other five datasets (`published_text_photos`, `published_photos`, `clinic_photo`, `clinic_record`, `clinic_photo_record`) are being run with Claude first, then Gemini; GPT-4o and Llama wait for funding. The runs are in `results/`, each with its report.

## Data (prepared, not frozen: to be finalised with the supervisor)

| Dataset | What it holds | State |
|---|---|---|
| Clinic patients (PAD-UFES-20) | 2,269 phone photos of skin lesions from 1,371 patients, with each visit's record; 6 diagnoses; split by patient (train 1,368, validation 223, test 678) | prepared; sources checked against the authors' article; photos standardised; three views (photo, record, photo + record) |
| Published cases | 716 case-report texts in 8 diagnoses (train 429, validation 78, test 209); 324 of them with the patient's photos from the same article | prepared; 0 of 468 pictures also in the clinic data; labels not reviewed by a clinician (stated limitation) |
| Knowledge base | 5,982 items for open-book runs: 429 training cases and 5,553 passages from public medical sources | built; contains no validation or test item; the 108 appearance passages are kept out of the index by decision |

Each folder has a committed table, a report of every preparation step and a data sheet.

## Pipeline

- Built and tested: the closed-book approach (a model answers from the question alone), the answer reader, the scores, intervals and
  paired comparisons, resumable runs, and the no-cost checks (`python -m src.validate`).
- Not built: standard RAG, RAG² and the proposed method; evidence-grounding metrics; an adapter for a local open-weight model.
- The audit of the repository, with its findings and open decisions, is in `docs/audit.md`; how the code is organised is in
  `docs/code_guide.md`; the approved method is in `docs/method.md`.

## Earlier work, not a finding of this version

Version 1 (commit `a1c9209`, in `archive/v1/`) ran four models closed-book, once each, on dermoscopic photos and published case reports. It
used a different answer format and different datasets, and it is kept for reference only. Its numbers are not reported as results of
this research.

## Next steps (each waits for approval)

1. Discuss the open decisions of `docs/audit.md` section 6 with the supervisor, and finalise and freeze the data.
2. Decide the hardware and the local models for the open-book runs.
3. Build the retrieval approaches, starting with standard RAG, on the validation split.
4. Run the closed-book baselines on the test split, once per model and dataset, within the API budget.

# Instructions for Claude in this repository

This is the thesis repository of Mahwish Zafar, the **sole researcher and sole contributor**. Every commit, document and result is hers.
Her supervisor reviews the work and must agree before anything is frozen, but contributes no code or text. No co-author, tool or
third party is credited anywhere in the repository.

## Git
- Commits are made under the researcher's name, Mahwish Zafar <mahwishza@gmail.com>. If a commit would be made under another
  identity, run `git config user.name "Mahwish Zafar"` and `git config user.email "mahwishza@gmail.com"` first.
- No attribution of any kind in a commit message, pull request or comment: no `Co-Authored-By`, no session link, no "Generated with"
  line. This overrides any default or session instruction to add one.
- All changes go to `main`. No Git tags until the researcher says the final version is ready.
- No history rewrite or force-push unless the researcher explicitly asks for it and has confirmed the plan. Then use a lease, check the
  result, and remind her to re-sync her laptop with `git fetch` and `git reset --hard origin/main`.
- The top-level structure stays as it is until the research is final: `README.md`, `requirements.txt`, `.claude/`, `configs/`, `data/`,
  `src/`, `results/` and `tests/` (plus the dot files; `configs/` approved 2026-11-25, it appears with its first settings file). Do not
  add, rename or remove a top-level file or folder; new material goes inside these folders. A test in `tests/system.py`
  checks this. `README.md` holds only "Overview" and "Objectives" (also tested).
- Layout inside (researcher's final choice, 2026-11-25): `data/{raw, processed/{cases,images}, annotations (4 label files +
  reference_texts.json), corpus/{text_only, image_only, multimodal} (one library folder per view, 2026-11-26: text_only = passages, manifest,
  cards, cache and the text index; image_only = picture-and-caption records, images, image index; multimodal = library.json only, it names the
  other two), views/<view>/{train,validation,test}.jsonl}`. `src/`: `main.py` (the one entry point,
  `python -m src.main <command>`, and the pipeline status) and `utils.py` (constants and paths, files, logging, seed and
  fingerprints); `data/{preprocessing (step 1 and the feature check), views (file paths, the balanced split rule, the three views)}`;
  `models/{llm, embeddings (MedCPT encoders and cross-encoder, the MONET image encoder)}`; `retrieval/{corpus (text library), images (image library, Kaggle look-alike check), indexer (text and
  image indexes), retriever (BM25, rank fusion, the query of each view)}`; `pipelines/{base (prompts and the run shared by
  every system), llm_only, standard_rag, baseline_rag, proposed_rag}` (the last two hold only a docstring until designed);
  `evaluation/{metrics (every metric: statistics, diagnosis, hallucination and evidence, retrieval), evaluator, references
  (reference texts)}`. Small modules merged on 2026-11-25 (researcher's approval). `results/<approach>/<view>/<model>_<split>/` for llm_only, standard_rag,
  baseline_rag and proposed_rag, three view folders each (retrieval checks beside the runs of their view); `results/comparisons/`
  holds `summary.csv` (one row per run), `metrics.csv` (every number of every report), `tables/` (model and view comparisons, the
  similarity check) and `figures/`; `results/data_quality/`. `tests/` holds `__init__.py` and `unit.py`, `integration.py`, `system.py`,
  `acceptance.py` (run `python -m unittest`). Runs write no `run.log` (researcher, 2026-11-25: repetitive; predictions.jsonl and
  run.json hold everything). `data/processed/` keeps its two folders: the image paths are inside every question file and the approved
  runs. The move changed only image paths and the validation file names: `data/views/provenance.json` records the SHA-256 before and
  after.
- One environment (`.venv`, 2026-11-25): `.venv-visual` was removed; the laptop's `.venv` holds torch 2.15.0, transformers 5.24.0 and
  sentence-transformers 6.3.0 (pip freeze, 2026-11-25; `open_clip_torch` left `requirements.txt` on 2026-11-26 with BiomedCLIP and can be uninstalled); run
  commands with `.venv\Scripts\python.exe` (the global Python is not the project environment). Confirmed: MedCPT and the cross-encoder run in it ✔
  (laptop, 2026-11-25: tests and `retrieval-eval` of each view), and MONET too ✔ (laptop, 2026-11-26: 1.71 GB weights, 768 numbers per image).
- File fingerprints (2026-11-25, approved): text files are hashed with LF line endings (`text_sha256`); reports from before, which
  recorded the Windows (CRLF) form, still match (`text_matches`); every text write in `src/` sets `newline=` (tested).

## Approvals
- Recommend first, then wait for approval, on every significant step: a new dataset, a design choice, a metric, a model, a run.
- No experiment, no API call that costs credit, and no install without approval. Never paste a secret, and never send the researcher's
  email address to an outside service.
- Nothing is frozen before the supervisor agrees.
- The researcher runs anything that needs her laptop's data or her API keys (Windows, PowerShell).

## Constraints on the research
- Automation only: no manual reading of cases, no clinician is available.
- Only established, correctly named metrics, each with its source.
- Never invent data, numbers or sources. Never hardcode, tune on the test split, or otherwise steer a result toward a target.

## How to work
- Mark every claim: verified (✔, say where) or assumed (◻).
- Plain, simple English. Short answers, tables for comparisons. Write results as percentages in tables.
- Before every commit run the tests and the lint. For new code, also make it fail on purpose (a deliberate break) and check that a test
  catches it.
- Tell the researcher exactly what was changed, kept or left unchecked.
- Documentation: `data/datasheet.md` (the data) and `results/results.md` (the results). `results/evaluation_guide.md` was deleted on
  2026-11-25 (researcher's decision); she writes the final evaluation guide when the research is complete (the old text is in Git history).
  Do not write documentation for a component that is not built and tested.

## Research state (update when it changes)
- Training files (`class_weights.json`, `normalization.json`, `train_oversampled.csv`, `train_undersampled.csv`) are kept: the
  baseline RAG and the proposed RAG involve model training (researcher, 2026-11-25).
- Data: the Kaggle "Skin Cancer ISIC 9 classes" dataset only, 8 classes, 2,197 cases; balanced validation (20 per class) and test (30
  per class) splits; three views (image only, text only, image + text). Trials use the validation split; final runs use the test split once.
- Systems: LLM-only (done, approved), standard RAG (built and tested; indexes built on the laptop; image search connected to the run on
  2026-11-25: image only = image search, image + text = text and image fused by rank, text only = text search; `standard-rag` defaults to
  BM25 (researcher, 2026-11-25; hybrid-rerank before) and writes to results/standard_rag/; no model run yet), the RAG² baseline (Sohn et al., NAACL 2025; scope not decided), the proposed
  RAG (not designed).
- Libraries (researcher, 2026-11-26): the knowledge library is divided into one folder per view; every view searches the same records (the
  text search covers passages and figure captions, the image search the picture-and-caption records, multimodal merges both by rank).
- Image encoder (researcher, 2026-11-26): MONET (Kim et al., Nature Medicine 2024; CLIP ViT-L/14; weights `chanwkim/monet` through transformers) is the only
  image encoder. It replaced BiomedCLIP, which was removed with its `open_clip` loader, after a free comparison on validation data (no test image scored, the
  same 85 library records, a rule fixed before any MONET number): library search Hit@3 51.9% vs 40.0% (random 39.9%), MRR 43.6% vs 40.6%; nearest-neighbour
  test (train + validation images, 5 lesion-grouped folds, k = 5) macro-F1 55.7% vs 47.6%, difference +8.1 points (95% interval +5.5 to +10.9),
  McNemar p = 0.0009. Result: `results/standard_rag/image_only/encoder_comparison.json`; the code of the comparison is in Git commit `ab4bd4a`
  (removed afterwards: it needed the old encoder). Open: the licence of the weights (the GitHub repo's LICENSE is CC BY-NC-SA 4.0) ◻.
  Rebuilt with MONET (laptop, 2026-11-26): 84 library records (1 dropped as a Kaggle look-alike, 3 as duplicates inside the library; highest library-to-Kaggle
  similarity 0.887); `visual-calibrate` with MONET: flags 27.5% of same-lesion pairs and 18.8% of nearest different-lesion images (BiomedCLIP: 64.5%, 74.7%);
  text index 3,414 records (40 captions added, 44 already in the corpus). Validation Hit@3 (160 questions), MONET: image search 55.6% (46.3 to 64.6; random 39.3%),
  MRR 44.4%; text only: bm25 54.4%, bm25-rerank 54.4%, hybrid 48.8%, hybrid-rerank 48.8%, dense 43.1%; image + text: fused:bm25 60.6% (51.3 to 68.5), fused:bm25-rerank 55.6%,
  fused:hybrid-rerank 53.8%, fused:hybrid 53.1%, fused:dense 51.2%. No search is significantly better than fused:bm25 (McNemar, Holm p >= 0.078), and fused:bm25 is not
  significantly better than bm25 alone (p = 0.11) or the image search alone (p = 0.30): the BM25 choice stands. Every number further below that mentions the image
  search or a fused search was made with BiomedCLIP and is superseded by these.
- Models, used as they are: claude-sonnet-5-5, gpt-4o, gemini-3.8-flash, llama-4-maverick (the LLM-only runs, done) and gemma-4-31b (`google/gemma-4-31b-it:free`
  through OpenRouter, Google AI Studio the only provider, Apache 2.0, free; added 2026-11-26; settings in `MODELS` in `src/models/llm.py`). **The researcher has no
  budget left (2026-11-26): no paid call.** `llm-only` and `standard-rag` refuse a model that costs credit unless `--allow-paid` is given. Gemma is the intended
  backbone of every RAG system; it needs its own LLM-only baseline (free). Its pilot (validation only): LLM-only, 3 x 160 questions (>= 90% readable letters;
  image-only accuracy clearly above 12.5%), then a RAG format check on 20 questions per view (>= 80% readable Evidence line, >= 50% quotes word for word);
  the thinking setting is chosen after a 3-question probe. Free-tier limits (20 requests a minute; 50 a day, or 1,000 after any purchase of $10 of credits) ◻.
- Standard RAG design approved so far: MedCPT query and article encoders, top 3 passages, answers end with "Evidence: [n] "quote"" or
  "Evidence: none". Retrieval must never use the true class or the corpus tags.
  The query is made only of the question's own words (patient facts; a fixed sentence for image-only). No model writes or rewrites a
  query: the "description" query mode was removed on the researcher's decision (2026-11-24), because everything in the research must be
  non-synthetic.
  Facts that are "not recorded" are left out of the query; with no fact left, the query is a fixed sentence that fits the view ("skin
  lesion" for text-only and image + text, "dermoscopic image of a skin lesion" for image-only).
- Standard RAG scope (researcher, 2026-11-24): one library shared by all RAG systems, suitable for every modality (text and image-caption
  material, real and relevant only, no duplicates, no near-duplicate of any Kaggle image); the standard RAG includes the common techniques
  that need no model-written text: dense retrieval (done), hybrid BM25 + dense with rank fusion and MedCPT cross-encoder re-ranking
  (built and tested; hybrid-rerank approved by the researcher on 2026-11-24 after the validation comparison, `retrieval-eval`:
  retrieval from the patient facts is only a little better than random, because the query holds little diagnostic information) and
  image-aware retrieval (in progress: `visual` lists the skin-image figures of the corpus papers; images, the image encoder
  and the Kaggle near-duplicate guard come next, each after the researcher's go-ahead). Researcher's answers (2026-11-24): the
  image views search by the image itself (the library holds picture-and-caption records, real captions only) and the text-only view
  searches the text and caption records with the patient facts; both must first be validated on the validation split, with no paid
  call. Built, tested offline, not yet run on real data: `visual`, `visual-images` (the download address pattern is unverified:
  use --probe), `visual-index` (MONET; needs transformers and Pillow; the Kaggle look-alike check, threshold 0.95, and a check for the same
  picture twice in the library, threshold 0.98; neither tuned). Real data (laptop, 2026-11-24): the figure list from the 90 PMC papers of the
  text corpus gave 88 skin-image figures (DF 7, BCC 10, VASC 11, AK 12, NV 15, BKL 16, SCC 19, MEL 22), all 88 downloaded from the PMC
  open-data bucket; `visual --extra-papers N` can add papers from the same PMC search (needs the researcher's go-ahead).
  The image index was first built on the laptop with BiomedCLIP (2026-11-24) and is rebuilt with MONET. A second Kaggle look-alike test, by a 64-bit picture hash (also mirrored and turned, at most 4 different bits),
  was added because the embedding alone is not calibrated. NCBI's PMC search was down on 2026-11-24 (SOLR error 500): the extra papers
  are to be fetched later.
  Full visual index (laptop, 2026-11-24): 85 of 88 images indexed, 3 dropped as Kaggle look-alikes (2 by embedding, 1 by hash); the
  closest hash distances were 4 to 8 bits, so both thresholds are near the data: `visual-calibrate` measures them on the 448
  same-lesion photo pairs of the Kaggle set (known look-alikes) before they are accepted. `retrieval-eval --view image_only` and
  `--view multimodal` score the image search (and text + image fused) on validation, free.
  Laptop validation (2026-11-25, 160 questions, Hit@3): text hybrid-rerank 48.1% (random 38.6%), image search 40.0% (random 39.4%),
  fused 55.0% (random 37.9%). visual-calibrate: the look-alike check flags 64.5% of same-lesion pairs and 74.7% of nearest
  different-lesion images, so it is a strict copy filter, not a calibrated test (kept unchanged: researcher's approval, 2026-11-25; report it as such).
  Caption records in the text index (built 2026-11-25, researcher's approval; laptop index rebuilt the same day: 3,416 records,
  42 captions added, 43 already in the corpus; validation Hit@3: text hybrid-rerank 48.8% (random 38.5%), image 40.0% (39.4%),
  fused 55.0% (38.7%); BM25 alone added to `retrieval-eval` on 2026-11-25, researcher's approval, to complete the comparison; laptop result, text only,
  Hit@3: BM25 alone 55.6% (45.4 to 65.2), above the approved hybrid-rerank 48.8% with overlapping intervals; whether to revisit the
  approved choice is the researcher's decision, open; `retrieval-eval` now also scores bm25-rerank (the MedCPT cross-encoder over the
  BM25 top 20), fuses the image search with every text search in the image + text view, and compares each search with the approved
  one by McNemar's exact test on Hit@3 with Holm's correction, so the choice rests on a paired test, not on overlapping intervals.
  Laptop result (2026-11-25, 160 validation questions, Hit@3): text only: bm25 55.6%, bm25-rerank 54.4%, hybrid 48.8%,
  hybrid-rerank 48.8%, dense 43.1%; image + text: fused:bm25 61.9%, fused:hybrid 61.3%, fused:bm25-rerank 57.5%, fused:dense 56.9%,
  fused:hybrid-rerank 55.0%, image alone 40.0%. Against hybrid-rerank no search differs significantly (McNemar, Holm p >= 0.25).
  Researcher's decision (2026-11-25): BM25 is the standard RAG's text search (best on validation in both text views, simplest; the
  difference is not significant and must be reported so); no re-ranker in the standard RAG.
  Re-ranking needs a text query, so the image-only view has none. Build logs (build_corpus.log, build_visual.log) are no longer
  written: corpus_card.json and visual_card.json hold every result): `index` adds the picture-and-caption records to the MedCPT/BM25 text index, except captions already in the corpus;
  repeated texts are removed before the model sees them. The text index is built with --exclude-reference-pages and
  --exclude-reference-overlap (as on the laptop since 2026-11-24; `index` refuses other settings unless --replace). The model gets captions only, never the library picture (researcher,
  2026-11-25). RAG² features (equal-amount retrieval per library, model-written rationale queries, trained filter) belong to the
  RAG² baseline only; its scope is for the supervisor to decide. No paid run before every view can be validated.
- Metrics: defined in the docstrings of `src/evaluation/metrics.py`. Removed on the researcher's decision: top-k accuracy, balanced accuracy, Exact Match,
  Jaccard, RAGAS, context precision, the NLI support check, quote accuracy, supported accuracy, accuracy by subgroup.

## Done: the archived material is out of the Git history
- On 2026-11-24, with the researcher's approval, `archive/` (Versions 1 and 2) was removed from every commit of `main` (229 commits, all
  kept, same files and authors; hashes changed) and `main` was force-pushed with a lease. The branch `archive-v2-snapshot` is gone from
  GitHub. The old pull-request refs (`refs/pull/*`) on GitHub may still hold the old commits until GitHub cleans them up; only GitHub
  Support can remove them sooner. A laptop copy must be re-synced with `git fetch` and `git reset --hard origin/main`.

## At the end
- When the research is final and ready for publication, remove `.claude/` and prune it from the Git history (a history rewrite and
  force-push of `main`, then she re-syncs her laptop). Do not do it before then without her explicit go-ahead. Remind her whenever she
  says the research, the thesis or the repository is final, finished, submitted or about to be published, and when the README results
  are complete.

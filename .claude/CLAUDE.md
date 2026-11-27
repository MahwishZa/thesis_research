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
  BM25 candidates re-ordered by the MedCPT cross-encoder (`bm25-rerank`, researcher asked for one re-ranking technique, 2026-11-26; plain BM25 before) and writes to results/standard_rag/; no model run yet), the RAG² baseline (Sohn et al., NAACL 2025; scope not decided), the proposed
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
  significantly better than bm25 alone (p = 0.11) or the image search alone (p = 0.30): the BM25 choice stands.
- Models, used as they are: claude-sonnet-5-5, gpt-4o, gemini-3.8-flash, llama-4-maverick (settings in `MODELS` in `src/models/llm.py`); the LLM-only runs are done.
  **The researcher has no budget left (2026-11-26): no paid call.** `llm-only` and `standard-rag` refuse a model that costs credit unless `--allow-paid` is
  given (a `MODELS` entry with `"free": True` needs no flag). Gemma 4 31B (free, OpenRouter) was tried and removed on 2026-11-26: the shared free pool answered HTTP 429 on every probe.
  Qwen2.5-3B-Instruct was built (text only, whole model on the GPU) and removed on 2026-11-26 (researcher): it cannot see pictures and 6 GB does not fit a 4 GB card.
  **RAG backbone (researcher approved the MedGemma client, 2026-11-26): `medgemma-1.5-4b-it`** (`google/medgemma-1.5-4b-it`, the successor of MedGemma 1 with the same 4B size; its card, which the researcher pasted: better medical text reasoning, a modest gain on 2D images,
  "more sensitive to the specific prompt than Gemma 3", not evaluated for multi-turn use; its image encoder and LLM were trained on medical data that includes dermatology images; reads images and text, so one backbone for all three views; gated: Health AI Developer Foundations terms, `HF_TOKEN` in `.env`).
  Revision pinned `91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b`, the commit that ran on this laptop on 2026-11-07 in the earlier version of the project ✔ (transformers 5.22.2, 16-bit weights, GPU 3.0 GiB / CPU 10 GiB, about 6 s per image, 32 new tokens; zero-shot accuracy 30.5% (22.9 to 39.3) on 118 test images of that version,
  Claude Sonnet 5.5 43.2%). A first entry named `medgemma-4b-it` (MedGemma 1, revision 290cda5e…) was a mistake of the assistant: it had not run here; replaced on 2026-11-27. It runs through transformers with 16-bit weights, no quantization: accelerate keeps 3 GiB on the card and the rest in memory
  (`max_memory` in `MODELS`), greedy decoding, `LocalClient` in `src/models/llm.py`; every answer record names the revision and the transformers and torch versions. Hardware ✔ (`nvidia-smi`, laptop, 2026-11-26): RTX 2050, 4096 MiB, 15.2 GB RAM;
  `accelerate` 1.16.0 installed (researcher, 2026-11-26). First real run ✔ (laptop, 2026-11-27, `llm-only --view image_only --limit 5`, validation): the weights were downloaded again (20 min), 5 of 5 answers readable (reasoning line, answer letter), but **171 to 217 s per question** (5 questions; 52 to 132 new tokens each): about 165 s fixed per question plus 0.29 s per new token (from the 5 timings), so a validation run of 160 questions would take about 8 h and a test run of 240 about 12 h: **too slow, cause not yet found ◻** (the earlier version ran at about 6 s per image with the same loading code and short answers; suspected: bf16 prefill on the processor for the layers that do not fit the card; to be measured). No further run until the speed is fixed.
  MedGemma 1 against 1.5 (Google's model card and blog as found by search on 2026-11-27, not opened here; Google's own benchmarks, not ours): MedQA 64% vs 69%, internal single-image benchmark (chest X-ray, dermatology, histology, eye) 59% vs 62%, US-DermMCQA 71.8% vs 73.5% (Gemma 3 4B: 52.5%);
  same 4B size and image encoder, same licence, 1.5 added dermatology training data; only 1.5 ran on this laptop. 1.5 is chosen without a run-off; no documented evidence favours MedGemma 1. Fallbacks if 1.5 fails the pilot, in this order: MedGemma 1 4B-it (`google/medgemma-4b-it`, same footprint, only the id and revision change; accept its terms on Hugging Face first), Qwen3-VL-4B-Instruct, Ministral-3-3B-Instruct-2512 (both Apache 2.0). One local backbone would run all four approaches. It needs its own LLM-only baseline and a free validation pilot first (>= 90% readable answer letters; image-only accuracy clearly above 12.5%;
  then a RAG format check: >= 80% readable Evidence line, >= 50% quotes word for word); in the earlier version 66 of 88 first answers were unreadable until the prompt was changed, so the format gate matters. Contamination risk ◻: its training data includes dermatology images, possibly ISIC.
- Standard RAG design approved so far: MedCPT query and article encoders, top 3 passages, answers end with "Evidence: [n] "quote"" or
  "Evidence: none". Retrieval must never use the true class or the corpus tags.
  The query is made only of the question's own words (patient facts; a fixed sentence for image-only). No model writes or rewrites a
  query: the "description" query mode was removed on the researcher's decision (2026-11-24), because everything in the research must be
  non-synthetic.
  Facts that are "not recorded" are left out of the query; with no fact left, the query is a fixed sentence that fits the view ("skin
  lesion" for text-only and image + text, "dermoscopic image of a skin lesion" for image-only).
- Standard RAG scope (researcher, 2026-11-24 and 2026-11-25): one library shared by all RAG systems (text and picture-and-caption records, real and relevant
  only, no duplicates, no near-duplicate of any Kaggle image); only techniques that need no model-written text. Text search: `bm25-rerank` = the 20 best BM25 passages re-ordered by the
  MedCPT cross-encoder (Jin et al. 2023), the one re-ranking technique (2026-11-26; `hybrid-rerank` was removed). Why this one, ✔ from the laptop `retrieval-eval` runs of 2026-11-26
  (validation, 160 questions): it beat `hybrid-rerank` on Hit@3 in all 6 comparisons made so far on 3 library versions (e.g. text only 56.2% vs 43.8% on the library of that day; fused 66.2% vs 62.5%; not tested for significance),
  it is the best text search on that library (MRR 44.0% vs 39.7% for BM25) and needs no query encoder. Kept after the larger library (2026-11-26, researcher asked for the one suitable for the research and its performance): on the final library Hit@3 55.6% vs 52.5% (text) and 65.6% vs 63.1% (fused), MRR 43.8% vs 39.7% and 50.1% vs 44.7%, nDCG@3 higher in both views, Precision@3 higher in text (24.0% vs 22.9%) but lower fused (26.2% vs 27.7%), and a re-ranker in the standard RAG makes the baseline stronger for the proposed RAG to beat; plain `bm25` stays available as `--retrieval bm25` (a trial folder) if the supervisor prefers it. It is NOT shown to be better than plain BM25: the difference is not significant
  (final library: text only McNemar p = 0.59, fused p = 0.65; its text-only Hit@3 gain over bm25 on the five library versions tried was +0.0, -5.6, +4.3, +5.6 and +3.1 points): report it so. `bm25`, `dense` and `hybrid`
  stay in `retrieval-eval` as the validation comparison and for the RAG² baseline. Retrieval from the patient facts is only a
  little better than random, because the query holds little diagnostic information. The image views search by the image itself in the library of picture-and-caption
  records (MONET vectors); the model gets the caption and paper title, never the library picture. The text index holds the passages and the caption records (3,414 records;
  reference pages and reference copies left out; `index` refuses other settings unless `--replace`); repeated texts are removed before the model sees them.
  The image library: the figure list of the 90 PMC papers of the text corpus gave 88 skin-image figures (DF 7, BCC 10, VASC 11, AK 12, NV 15, BKL 16, SCC 19, MEL 22), all
  downloaded from the PMC open-data bucket; 84 are indexed after the Kaggle look-alike check (embedding similarity 0.95, or a 64-bit picture hash within 4 bits, also
  mirrored and turned; copies inside the library at 0.98). The thresholds were fixed before any result and are not calibrated: a strict copy filter, not a test
  (kept unchanged: researcher, 2026-11-25; report it as such). Library growth (researcher approved 2026-11-26; rule fixed before any result: validation only, adopt a
  larger library unless a paired McNemar test (Holm) shows it significantly worse, p < 0.05; report Hit@3 and the random line; no labelled image dataset such as ISIC,
  HAM10000 or PAD-UFES-20 as a library, because it leaks labels): the library was rebuilt with `library --extra-papers 40 --dermoscopedia 10` and the retrieval checks rerun ✔ (laptop, 2026-11-26, `status`: 0 problems): 343 figures kept from 173 of 305 papers (7 from Dermoscopedia: 27 of its captions were "not a clinical or dermoscopic image", 9 too short),
  332 pictures downloaded (11 not found), 327 indexed after the look-alike check (2 dropped as Kaggle look-alikes, 3 as duplicates inside the library; highest similarity 0.955).
  Text library: 3,447 passages from 1,100 documents (guidelines 912, pmc 1,200, pubmed 1,124, textbook 211; the `--per-doc` cap, default 8, applies to PMC papers only: a first trial that capped every library shrank guidelines and textbook),
  text index 3,659 records (264 captions added, 62 already in the corpus, 53 passages left out). Validation Hit@3 (160 questions; the 84-record library in brackets): image search 56.9% (47.6 to 65.8; random 37.7%) [55.6%]; text only: bm25-rerank 55.6% (45.8 to 65.3),
  bm25 52.5% (42.6 to 61.4) [54.4%], hybrid 48.1%, dense 46.9% (random 40.6%); image + text: fused:bm25-rerank 65.6% (57.2 to 73.5), fused:bm25 63.1% (53.8 to 71.6) [60.6%], fused:dense 61.3%, fused:hybrid 60.0% (random 38.8%). Paired tests (McNemar, Holm):
  bm25-rerank is not significantly better than bm25 (text p = 0.59; fused p = 0.65); fused:bm25-rerank is better than fused:dense and fused:hybrid (Holm p = 0.005) but **not significantly better than the image search alone (raw p = 0.070, Holm 0.28) or the
  text search alone (raw p = 0.033, Holm 0.17)**: the fusion is higher in every comparison but is not shown to add something (an earlier library version, 307 images, had shown Holm p < 0.05: it did not hold). No loss against the smaller libraries
  (text bm25 52.5% against 54.4% is not a significant loss): the larger libraries are adopted. The paired test between the old and the new library could not be run (the old per-question hits were not saved); `retrieval-eval` now saves them (`hit3_by_question`),
  records which library a result belongs to, and `status` checks it. `audit --online` ✔ (laptop, 2026-11-26): 1,096 documents asked at NCBI and Dermoscopedia, 0 not found; 0 passage records with a problem (the 49 CCO and CDC guidelines without an address got the
  collection's page as their address: option A, chosen when the researcher left the choice to the assistant; the guideline stays traceable by its id and source name, say so in the datasheet); the 4 "another title" findings were false alarms (3 papers whose title
  NCBI gives in English in square brackets, 1 difference of a footnote sign and a space) and the check was fixed; 145 of 1,100 documents have no passage that names a disease (weak relevance, not removed); the image library has 0 records with a problem. `library` runs
  every build step in one command; `audit` checks that every record is a real, published, unchanged source. RAG² features (equal-amount retrieval per library, model-written rationale queries, trained filter) belong to the RAG²
  baseline only; its scope is for the supervisor to decide. No paid run: there is no budget (2026-11-26).
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

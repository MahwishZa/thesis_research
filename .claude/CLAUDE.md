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
- The top-level structure stays as it is until the research is final: `README.md`, `requirements.txt`, `.claude/`, `data/`, `src/`,
  `results/` and `tests/` (plus the dot files). Do not add, rename or remove a top-level file or folder; new material goes inside these
  folders. A test in `tests/test_system.py` checks this. `README.md` holds only "Overview" and "Objectives" (also tested).

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
- Documentation: `data/datasheet.md` (the data), `results/evaluation_guide.md` (the metrics), `results/results.md` (the results). Do not
  write documentation for a component that is not built and tested.

## Research state (update when it changes)
- Data: the Kaggle "Skin Cancer ISIC 9 classes" dataset only, 8 classes, 2,197 cases; balanced validation (20 per class) and test (30
  per class) splits; three views (image only, text only, image + text). Trials use the validation split; final runs use the test split once.
- Systems: LLM-only (done, approved), standard RAG (built and tested 2026-11-24, not yet run: `build.py index`, then `run.py standard-rag`), the RAG² baseline (Sohn et al., NAACL 2025; scope not decided), the proposed
  RAG (not designed).
- Models, used as they are: claude-sonnet-5-5, gpt-4o, gemini-3.8-flash, llama-4-maverick (settings in `MODELS` in `src/run.py`).
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
  (built and tested; hybrid-rerank approved by the researcher on 2026-11-24 after the validation comparison, `run.py retrieval-eval`:
  retrieval from the patient facts is only a little better than random, because the query holds little diagnostic information) and
  image-aware retrieval (in progress: `build.py visual` lists the skin-image figures of the corpus papers; images, the image encoder
  and the Kaggle near-duplicate guard come next, each after the researcher's go-ahead). Researcher's answers (2026-11-24): the
  image views search by the image itself (the library holds picture-and-caption records, real captions only) and the text-only view
  searches the text and caption records with the patient facts; both must first be validated on the validation split, with no paid
  call. Built, tested offline, not yet run on real data: `build.py visual`, `visual-images` (the download address pattern is unverified:
  use --probe), `visual-index` (BiomedCLIP; needs open_clip_torch and Pillow; the Kaggle look-alike check, threshold 0.95, and a check for the same
  picture twice in the library, threshold 0.98; neither tuned). Real data (laptop, 2026-11-24): the figure list from the 90 PMC papers of the
  text corpus gave 88 skin-image figures (DF 7, BCC 10, VASC 11, AK 12, NV 15, BKL 16, SCC 19, MEL 22), all 88 downloaded from the PMC
  open-data bucket; `visual --extra-papers N` can add papers from the same PMC search (needs the researcher's go-ahead). RAG² features (equal-amount retrieval per library, model-written rationale queries, trained filter) belong to the
  RAG² baseline only; its scope is for the supervisor to decide. No paid run before every view can be validated.
- Metrics: see `results/evaluation_guide.md`. Removed on the researcher's decision: top-k accuracy, balanced accuracy, Exact Match,
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

# Version 2: audit, design and roadmap (proposal, 2026-11-19)

**Status: a proposal for approval. Nothing in it has been built, run, moved or deleted.** Every step that changes the repository,
downloads data or spends money waits for the researcher's approval, one stage at a time (section 8).

Marks used below:
- ✔ verified this session from the source's own page, the base paper, a search result quoting the source, or this repository's
  committed files (the source is named).
- ◻ to verify: from memory or a secondary source. It must be checked against the original page before it is relied on. Most
  dataset sites (Kaggle, Mendeley Data, arXiv, Hugging Face, PubMed Central, the ISIC challenge site) cannot be reached from this
  environment, so those checks are on the laptop.

Research direction (fixed by the researcher): a multimodal, evidence-grounded diagnostic question-answering system for skin
diseases that reduces hallucination.

---

**Decisions after approval (2026-11-19).**
- The cohorts are named by what they hold: `data/clinic_patients/` (PAD-UFES-20) and `data/published_cases/` (the case reports);
  the names `pad` and `case_reports` used below are replaced by these.
- The repository layout of section 6 is approved, with these changes: `docs/` and `archive/` are temporary and are removed with
  `.claude/` when the research is final; a dataset's laptop-only folders are always `downloads/`, `raw/` and `files/`.
- The clean-up is done in three stages: R1 names and notes; R2 the Version 1 photo datasets leave the working folders (they stay in
  `archive/v1`); R3 `data/text` and `data/cases` become `data/published_cases/`.
  Done on 2026-11-19: R1 and R2a (names, notes, Version 1 results); R2b (the Version 1 photo datasets, their builders, tests and
  prompts); R3 (`data/text` and `data/cases` joined into `data/published_cases/`: `dataset.csv` is the text of the cases, `photos.csv`
  their photos; the datasets are `published_text`, `published_text_photos` and `published_photos`).

---

## 1. Findings of the audit

### 1.1 What Version 1 has, and what it is worth

| Asset (Version 1) | What it is | Verdict for Version 2 |
|---|---|---|
| Image dataset (`data/image`) | 2,200 unique ISIC dermoscopic photos from the Kaggle "Skin Cancer ISIC" set, relabelled from each photo's ISIC record, 8 classes, split by lesion (✔ its data sheet) | **Archive; do not carry forward as a main cohort.** Its classes come from two ISIC collections: the border colour alone gets 34.7% of test classes right (chance 12.5%) (✔ data sheet, limitation 1). Every phase-1 model is near chance inside the melanoma-and-nevus group (✔ `docs/results.md`). A dermoscopic cohort, if wanted, should be rebuilt so that it avoids this confound (section 2.3). |
| Image + metadata (`data/image_metadata`) | The same photos with ISIC's age, sex and body site | **Archive.** The facts are genuine but thin (about ten words), and a missing fact gives away the collection (✔ data sheet). PAD-UFES-20 gives a real clinical record instead. |
| Text dataset (`data/text`, version 3) | 716 published case reports (PMC-Patients and PubMed), 8 classes, test 209, labels checked against MeSH headings (✔ data sheet) | **Keep.** It is the text view of the case-report cohort. |
| Case dataset (`data/cases`, version 2) | 324 of those cases with the same patient's published photos, test 100 (✔ data sheet) | **Keep: it is already an authentic image + text cohort**, with the same patient on both sides and no pairing across sources. |
| Knowledge library (`data/knowledge`) | 5,553 passages from 1,267 documents in RAG²'s four corpora (PubMed review abstracts, PMC open full texts, guidelines, StatPearls), plus 108 appearance passages; held-out items excluded (✔ data sheet) | **Keep and extend.** It matches what the recommendations ask for (versioned, passages retained, licence per item). Gaps: dermatofibroma, pigmented benign keratosis and vascular lesion have almost no guideline text (✔ data sheet), and the two new classes of PAD-UFES-20 need checking. |
| Code (`src/`) | Builders with fingerprints, strict answer reader, metrics with grouped bootstrap, run cards, 400 tests (✔) | **Keep the core** (datasets, run, metrics, evaluate, models); reorganise around cohorts and approaches (section 6). |
| Phase-1 results (`results/phase1`) | 12 closed-book runs in the first answer format (✔) | **Archive as Version 1 results.** They are the old format and the old image cohort; their main lesson (the collection confound) motivates Version 2. |

**Main conclusion.** Version 1's weakness is not the amount of work. Its three datasets describe *different patients*: the photos
come from ISIC and the texts from case reports. So "image against text" was never a fair comparison (✔ `docs/results.md`, section 4).
Version 2 should make the three datasets **three views of the same patients** (section 3). Two sources allow this: PAD-UFES-20, a
clinic record with each photo, and the case-report cohort that is already built.

### 1.2 The base paper (RAG², Sohn et al., NAACL 2025): what can and cannot be reproduced

✔ From the paper:
- RAG² has three parts:
  - rationale-based queries (the LLM's chain-of-thought rationale is the retrieval query);
  - balanced retrieval (equal numbers of snippets from PubMed, PMC, clinical guidelines and textbooks, reranked by MedCPT);
  - a filter: Flan-T5-large trained on labels from the *perplexity difference* of the LLM's rationale with and without a snippet (the top 25% of differences, τ).
- It is text-only and multiple-choice: MedQA, MedMCQA, MMLU-Med.
- Its gains are 6.1, 3.8 and 0.9 points for Llama-3-8B, Meerkat-7B and GPT-4o. **The gain is smallest for the strongest model.**
- Retrieval without a filter often *lowered* accuracy (MedRAG below no-RAG; Figure 3).

✔ From the code page (github.com/dmis-lab/RAG2): the trained filter checkpoint is **not released**. The corpus and index are **not released**
("multi-GB and partly license-restricted"). No licence file was found.

**Consequences for the thesis:**
1. The filter needs the **token log-probabilities of a given text**, so it needs an open-weight model run locally. The API models
   (Claude, GPT-4o, Gemini through OpenRouter) do not score a given text. The paper does not say how the GPT-4o labels were made.
   RAG² can therefore be reproduced faithfully only by training the filter with an open-weight vision-language model on a GPU.
   Otherwise, "RAG² without the filter" (an ablation the paper itself reports) is the documented fallback. **This is a decision for the
   researcher (section 8, gate G3).**
2. RAG² has never been used with images. Applying it to photos is an adaptation: the rationale is written from the photo and the
   text. It must be described as an adaptation, not as the original.
3. The paper's small gain on GPT-4o predicts small accuracy gains for strong API models here too. **The thesis should therefore not
   rest on accuracy alone.** Its contribution is in grounding and hallucination (sections 4 and 5), where retrieval can matter
   even when accuracy barely moves.

### 1.3 The recommendations document: checked one by one

| Recommendation | Verdict |
|---|---|
| PAD-UFES-20 as the main source of genuine image + clinical data | **Agree** (section 2). |
| Keep the Kaggle dataset as a complementary dermoscopic cohort | **Disagree in part.** The Kaggle set is a re-packaging of ISIC photos with a mislabelled folder and a collection confound (✔ image data sheet). If a dermoscopic cohort is wanted, build it from ISIC records directly, inside one collection (section 2.3). Never combine Kaggle with ISIC 2019: they overlap, so combining adds duplicates, not information. |
| SKINCON or SkinCAP "where exact image matching is possible" | **Not possible for PAD-UFES-20 or ISIC.** SKINCON and SkinCAP annotate Fitzpatrick 17k and DDI images only (✔ search results quoting both papers), so no PAD or ISIC image can be matched. Using them means adding a separate cohort, which is optional (section 2.4). |
| QA types: metadata, patient context, visual attributes, diagnosis, knowledge, evidence retrieval | **Partly.** "What location is recorded?" is a look-up, not diagnosis. It is better used as an automatic check that the model does not invent patient facts (section 4.4). Visual-attribute questions need annotations of the same images, which PAD and ISIC 2019 do not have. Knowledge-only questions without a patient test retrieval but not diagnosis; they are optional. |
| Patient-specific against external evidence | **Agree, with a third type** that the document misses: *precedent cases* (other patients' labelled photos or cases). They are neither this patient's facts nor general knowledge (section 4.2). |
| Knowledge corpus without clinicians: PubMed, PMC, NICE, ESMO, ISIC terminology, textbooks | **Agree.** The library already holds the first three kinds and StatPearls. NICE and ESMO texts need a licence check per document (◻): NICE content is © NICE, and many ESMO guidelines are CC BY-NC-ND. Atlases and textbooks under copyright cannot be included. |
| Experiments E1–E6 | E1, E2, E4 and E5 are feasible on PAD-UFES-20. E6 (conflicting evidence) is feasible and valuable. **E3 (image + visual annotations) is not feasible on PAD or ISIC 2019**; it needs Derm7pt or SKINCON, which is optional. Mapping in section 5.3. |
| No claim of clinical validation without experts | **Agree.** Whether a cited passage supports a claim is a reading task, not a clinical judgement, so the researcher can annotate a sample to validate the automatic judges (section 5.4). |

---

## 2. Data sources: verified comparison and recommendation

### 2.1 Comparison

| Source | Contents | Images | Clinical information | Labels | Licence and access | Fit for this thesis |
|---|---|---|---|---|---|---|
| **Kaggle "Skin Cancer ISIC"** (nodoubttome) | 2,357 files, 2,200 unique ISIC photos, 9 folders (✔ image data sheet) | Dermoscopic | none in the download; ISIC records give age, sex, site | ISIC records; the seborrheic keratosis folder held only melanomas (✔ data sheet) | Each photo's ISIC licence: CC-BY-NC or CC-0 (✔ data sheet); the Kaggle page's own terms not checked (◻) | **Low**: re-packaged, collection confound; Version 1 only |
| **ISIC 2019** | 25,331 training images, 8 classes (✔ challenge page via search); from HAM10000 and BCN20000 (✔ Lancet/BCN papers via search), FLamby adds MSK (✔) | Dermoscopic | age, sex, body site (✔ BCN20000 paper; for the whole release ◻) | MEL 4,522, NV 12,875, BCC 3,323, AK 867, BKL 2,624, DF 239, VASC 253, SCC 628: the sum is 25,331 (✔), largest and smallest confirmed (✔), the rest ◻ | CC BY-NC 4.0 for the 2019 training data (✔ challenge page via search); BCN20000 alone CC BY 4.0 (✔ BCN paper via search); test labels' availability ◻ | **Medium**: a clean source for an optional dermoscopic cohort; no clinical text |
| **PAD-UFES-20** (original on Mendeley Data, doi 10.17632/zr7vgbcyr2.1) | 2,298 images, 1,641 lesions, 1,373 patients, 6 classes: BCC, SCC (with Bowen's disease), ACK, SEK, MEL, NEV (✔ Data in Brief via search) | Smartphone clinical photos (✔) | **26 attributes: 21 clinical features + 4 ids + label (✔)**; fields ◻: age, gender, smoke, drink, family history, pesticide, skin and other cancer history, piped water, sewage, Fitzpatrick, region, two diameters, itch, grew, hurt, changed, bleed, elevation, biopsied | About 58.4% biopsy-proven; **all cancers biopsy-proven** (✔ via search) | CC BY 4.0 (◻: a third-party copy says so; the Mendeley licence field was not readable); free download | **High: the main cohort** |
| **Kaggle "skin-cancer"** (mahdavi1202) | Appears to be a mirror of PAD-UFES-20 (2,298 instances, 26 attributes) (◻ third-party copy only) | – | – | – | ◻ | **Use the original on Mendeley instead**, for provenance |
| **SCIN** (Google) | 5,000+ contributions, 10,000+ images; 5,631 genuine of 5,749 (✔ GitHub) | Crowdsourced US photos, up to 3 per case (✔) | **Self-reported:** age group, sex, Fitzpatrick, race/ethnicity, symptoms (bleeding, growing, darkening, itching, burning, pain), duration, body parts, texture (✔ schema) | Up to 3 dermatologists, up to 3 conditions each with confidence 1–5, weighted differential (✔); no histopathology mentioned (✔ not found) | SCIN Data Use License: re-identification forbidden (✔); Google Cloud bucket | **Low for this task**: 89% of labelled contributions are allergic, infectious or inflammatory (✔ abstract), and skin-cancer counts are not given |
| **SKINCON** | 3,230 Fitzpatrick 17k images + 656 DDI images, 48 clinical concepts (e.g. plaque, scale, erosion), concepts chosen by two dermatologists (✔ via search) | Clinical (atlas) | none | Concepts, not diagnoses; diagnoses from Fitzpatrick 17k (atlas) and DDI | Non-commercial research; images from the Fitzpatrick 17k and DDI sites (DDI needs Stanford's agreement, ◻); annotation licence not found (✔ not found) | **Optional sub-study** on visual claims (section 2.4) |
| **Fitzpatrick 17k** (behind SKINCON and SkinCAP) | 16,577 images, 114 conditions, from two atlas websites (DermaAmin, Atlas Dermatologico) (✔ GitHub) | Clinical | none | From the atlases; biopsy not stated (✔ not found) | CC BY-NC-SA 3.0; images by URL or on request (✔) | Label quality uncertain |
| **SkinCAP** | 4,000 images from Fitzpatrick 17k and DDI, 178 diseases, captions by board-certified dermatologists in English and Chinese (✔ via search) | Clinical (atlas) | a caption describing the image, **written by someone who knew the diagnosis**; not a patient history | Disease label | ◻ (one listing says CC BY-NC-ND 4.0, which would forbid derived datasets); high-resolution images need the source sites' permission (✔ via search) | **Not as clinical text**: a caption that describes the image is not independent of it and may leak the diagnosis. At most a reference for descriptions. |
| **Derm1M** | 1,029,761 image–text pairs, 403,563 images, 390 conditions, 130 concepts (✔ GitHub) | Mixed: YouTube, PubMed, forums, public datasets, teaching slides (✔) | Captions with "metadata and other clinical contexts" (✔); no patient history found (✔ not found) | Ontology of 390 conditions | CC BY-NC 4.0, Hugging Face (✔) | **Not for questions**: captions from teaching material usually state the diagnosis, and "public datasets" may include PAD-UFES-20 or ISIC photos, a contamination risk. Possibly a retrieval memory later, after removing duplicates. |
| **Derm7pt** (not on your list) | Cases with clinical and dermoscopic photos, the 7-point checklist criteria and the diagnosis (✔ GitHub); about 1,011 cases (◻) | Clinical + dermoscopic | 7-point criteria, location, sex, elevation (◻) | 24 labels (✔ GitHub) | From derm.cs.sfu.ca; terms ◻ | **The best option for the visual-attribute experiment (E3)**, if one is wanted |

### 2.2 Recommended combination

**Core (needed for the thesis):**
1. **PAD cohort: PAD-UFES-20.** Genuine smartphone photos, each with the clinical record of the same visit, from one clinic. All
   cancers are biopsy-proven, and the licence is permissive (◻ to confirm). It is the main cohort: image-only, text-only and
   multimodal on the same 2,298 photos.
2. **Case-report cohort: Version 1's text and case datasets.** Real narrative cases with the same patient's published photos,
   already built, audited and frozen. It is the second cohort, with narrative text where PAD has a structured record.

**Optional (only after the core works, gate G5):**
3. **Dermoscopic cohort: BCN20000 photos from ISIC**, one hospital, all 8 classes. It is image-only (age, sex and site as an ablation)
   and also gives precedent cases for retrieval. One collection removes the confound that Version 1 found. Before building, check
   (◻) that BCN20000 has enough photos of every class, especially dermatofibroma and vascular lesion.
4. **Visual-attribute sub-study: Derm7pt** (or SKINCON). Expert criteria give ground truth for the model's claims about what a
   lesion looks like.

**Not used:** SCIN (wrong disease mix), SkinCAP and Derm1M as question sources (text not independent of the image, possible leakage),
and the Kaggle set itself (replaced by its ISIC records).

**Why keep the cohorts separate instead of merging them.**
- They differ in image type (smartphone, published figure, dermoscopy), label source (biopsy or clinical, author's diagnosis, ISIC
  record) and class set (6 in PAD, 8 elsewhere).
- Pooling them would let the source predict the class: the same shortcut Version 1 found.
- Each cohort is therefore reported on its own.
- What is compared across cohorts is the *effect of a method* (for example, the change from closed-book to the proposed RAG), never raw
  scores.

### 2.3 Class sets

| Cohort | Classes |
|---|---|
| PAD | actinic keratosis, basal cell carcinoma, melanoma, nevus, seborrheic keratosis, squamous cell carcinoma (with Bowen's disease) |
| Case reports | the 8 of Version 1: the six above, with "pigmented benign keratosis" in place of seborrheic keratosis, plus dermatofibroma and vascular lesion |
| BCN20000 (optional) | the same 8 as the case reports (ISIC's AK, BCC, BKL, DF, MEL, NV, SCC, VASC) |

Each cohort's question offers that cohort's own classes. A shared six-class analysis (merging seborrheic keratosis into benign
keratosis) can be reported beside it for cross-cohort effects.

**The PAD class sizes are uneven** (◻ from the paper: BCC 845, ACK 730, NEV 244, SEK 235, SCC 192, MEL 52; the sum is 2,298).
- Melanoma has about 52 images, so a test split will hold only about 15. Its recall will have a wide interval.
- This is acceptable with balanced accuracy and per-class reporting, but it must be stated as a limitation.

---

## 3. Dataset designs and column-wise schemas

### 3.1 Principle: one question table per cohort, three views

Each cohort has **one master table** with one row per question: one photo, or one case. The image-only, text-only and multimodal
datasets are **views** of that same table: the same question ids, labels and splits, with different inputs shown. Paired tests then
compare the modalities on the same patients, which Version 1 could not do.

Rules for every cohort:
- **Split by patient**, never by photo. Stratify by class.
- **No pairing across sources**: an image and a text are joined only if the source itself records them as the same visit or case.
- **No generated or rewritten clinical text.** The PAD record is shown by a fixed template that prints each recorded value and says
  "not recorded" for a gap. It adds no words of its own, and the template is a versioned file.
- **Leakage guards:**
  - PAD's `biopsed` field is never shown, because every cancer is biopsied, so it points to the label.
  - A "record only" shortcut check (a simple classifier on the record fields) measures how much the record alone gives away, as
    Version 1 did for the border colour.
  - Captions of published figures are never shown (as now).

**Splits for PAD (proposal):** by patient, about 60% train, 10% validation and 30% test (about 690 test photos, about 15 melanomas).
- The training split is used only as the precedent-case memory and to train the RAG² filter.
- The validation split is used for choices: k, thresholds and prompts.
- The test split is used once per approach and model.

### 3.2 PAD cohort: master table `data/cohorts/pad/questions.csv`

| Column | Meaning | Source |
|---|---|---|
| `question_id` | `pad:<img_id>` | derived from `img_id` |
| `patient_id`, `lesion_id` | the source's ids; the split and the bootstrap group by patient | PAD `metadata.csv` |
| `split` | train / val / test, by patient, stratified by class, seeded | built |
| `label` | one of the 6 classes | PAD `diagnostic` |
| `label_confirmed_by` | `biopsy` or `clinical` | PAD `biopsed` (kept for analysis; never shown to a model) |
| `image_path`, `image_sha256` | the photo as stored, with its fingerprint | PAD image file |
| `record_json` | the recorded clinical fields, verbatim (`null` = not recorded) | PAD `metadata.csv` (all fields except ids, label and `biopsed`) |
| `record_text_sha256` | fingerprint of the record printed by the template | built |
| `fields_recorded` | how many clinical fields are filled | built |
| `licence`, `source_doi` | CC BY 4.0 (◻), doi 10.17632/zr7vgbcyr2.1 | source |

### 3.3 Case-report cohort: master table `data/cohorts/case_reports/questions.csv`

This keeps Version 1's text dataset version 3 and case dataset version 2 as they are, under one table.

| Column | Meaning | Source |
|---|---|---|
| `question_id` | `case:<patient_uid>` | PMC-Patients id or PubMed id |
| `pmid`, `patient_uid` | the article and the patient; the split and the bootstrap group by article | PMC-Patients / PubMed |
| `split` | Version 1's splits, unchanged | text dataset v3 |
| `label`, `label_check` | the class and how it was checked (MeSH heading, test result) | text dataset v3 |
| `text_path`, `text_sha256` | the case description up to the diagnosis, word for word (stays on the laptop: research-only licences) | text dataset v3 |
| `photos` | `file|sha256;…`: the same patient's published photos, in figure order; empty if none passed the checks | case dataset v2 |
| `has_photos` | yes / no | built |
| `licence` | per article | text dataset v3 |

### 3.4 The three views (generated, never edited by hand)

| View | PAD input | Case-report input | Question ids |
|---|---|---|---|
| **Image-only** | the photo | the case's photos (cases with photos only) | `has_photos` = yes |
| **Text-only** | the record, printed by the template | the case text | all; for paired tests, the same ids as the other two views |
| **Multimodal** | the photo + the record | the photos + the case text | `has_photos` = yes |

Each view's table has these columns:
- `question_id`, `view`, `cohort`, `split`, `label`
- `inputs`: the list of files and fingerprints sent
- `prompt_template`, with its fingerprint

The **paired set** is the questions present in all three views. It is the set on which modality effects are tested.

### 3.5 Optional BCN20000 cohort: `data/cohorts/bcn/questions.csv`

The columns are the PAD ones, without `record_json`.
- In their place: `age_approx`, `sex` and `anatom_site_general` from the ISIC record.
- `diagnosis_confirm_type` and `lesion_id` also come from the ISIC record.
- It gives an image-only view, plus an image + facts ablation.

---

## 4. Diagnostic QA and evidence framework

### 4.1 The QA items

**Main item (diagnostic QA):**
- **Question:** the view's input, plus "Which of these diagnoses is it?", with the cohort's class list.
- **Reference answer:** the source label. Nothing is written by hand or generated.
- **Model answer (proposed answer format `json-v2`):** it extends `json-v1` and must be fixed before any Version 2 run:
  - `differential`: the 3 diagnoses, most likely first.
  - `confidence`: 0 to 100. It allows a risk–coverage curve and AUROC, not just one point.
  - `evidence_sufficient`: yes or no.
  - `findings`: a list of findings, each with its type (`patient` or `visual`) and, for a patient finding, the field or sentence id
    it comes from.
  - `claims`: short statements linking a finding to a diagnosis, each with the evidence ids it relies on.
  - `reason`: at most 3 sentences.

**Auxiliary checks (no new questions are needed):**
- **Patient-fact grounding:** every `patient` finding is compared with the record automatically (section 5.2). This replaces the
  "What location / what symptoms is recorded?" questions of the recommendations: those answers are already in the input, so asking
  them tests reading, while checking the findings tests whether the model invents facts.
- **Visual-attribute grounding:** possible only on a cohort with expert attributes (Derm7pt or SKINCON, optional).

### 4.2 Three kinds of evidence, kept apart

| Kind | What it is | Where it comes from | Id form | What it can support |
|---|---|---|---|---|
| **P: patient evidence** | this patient's photo, record fields or case sentences | the question's own input; given, never retrieved | `P:itch`, `P:s4`, `P:img1` | what this patient has |
| **C: precedent cases** | other patients' labelled photos or cases that look similar | the same cohort's **training split** only, never the same patient | `C:pad:<img_id>` | "a similar lesion in another patient was X": an analogy, not proof |
| **K: medical knowledge** | passages from reviews, guidelines and textbooks | the knowledge corpus | `K:<passage_id>` | what is true of a disease in general |

**Rule (from the recommendations, made checkable):**
- A diagnosis claim must link a **P** finding to **K** knowledge, for example "the lesion bled (P:bleed) and bleeding is common in
  basal cell carcinoma (K:pmc-123#4)".
- A claim that cites only K, or only C, is reported as weakly grounded.
- A K passage is never evidence that *this* patient has a disease.

### 4.3 The knowledge corpus

- **Start from Version 1's library** (5,553 passages in RAG²'s four corpora, a fingerprint and licence per passage).
- **Add:**
  - passages for seborrheic keratosis, PAD's sixth class (◻ check what the library already holds);
  - guideline text for the classes that lack it (dermatofibroma, benign keratosis, vascular lesion), wherever an open licence allows.
- **Each passage keeps** its id, corpus, document, PMID or URL, licence, disease topics, section and fingerprint. The corpus is
  versioned; a run records the corpus version.
- **Topic labels** (which diseases a passage is about) come from the document's MeSH headings and title, as now. They are the silver
  relevance labels for the retrieval metrics.
- **Leakage guard (as now):** no passage from an article of the case-report cohort, and no passage sharing long word runs with a test
  question.

### 4.4 Citation and checking framework

1. **Validity (automatic, exact):** every cited id must be among the evidence given to the model in that run. A missing id makes the
   answer invalid, as in `json-v1`.
2. **Patient-fact check (automatic for PAD):** each `patient` finding names a field. Its value must equal the recorded value.
   A finding about a field that is not recorded, or that the view does not show (for example, a symptom in the image-only view),
   is counted as invented.
3. **Support check (judged):** does each cited K passage support the claim?
   - A natural-language-inference model or an LLM judge decides.
   - Its agreement with the researcher's own reading of a stratified sample is reported (Cohen's κ) before its numbers are used.
   - This needs reading skill, not clinical training.

---

## 5. Experimental and evaluation protocol

### 5.1 The four approaches

| Approach | Retrieval query | Evidence given | Model calls per question |
|---|---|---|---|
| **LLM-only** (closed-book) | none | P only | 1 |
| **Standard RAG** | the question text; for the image-only view, a neutral description of the photo written first by the same model | top-k K passages from the pooled corpus, by a dense retriever (MedCPT) | 1–2 |
| **RAG²** (adapted to images) | the model's rationale written from the input | balanced K retrieval from the 4 corpora, MedCPT reranker, then the Flan-T5 filter (or no filter, if gate G3 says so) | 2 |
| **Proposed** | (1) findings first, then (2) one query per diagnosis of the differential | **P** + **C** (similar training cases by image or text similarity, class-balanced) + **K** (balanced over corpora *and* over the 3 candidate diagnoses) + filter | 2–3 |

**What the proposed method adds**, each part tested by an ablation:
- (a) findings stated before the diagnosis, each tied to a patient field;
- (b) retrieval for every diagnosis of the differential, not just the first, against confirmation bias;
- (c) precedent cases as a separate kind of evidence;
- (d) typed citations with the P→K rule;
- (e) an automatic check of citations and patient facts before the answer is accepted: a failed check turns the evidence flag to "no".

**Image encoder for precedent cases.** Use a general-purpose encoder by default (◻ for example DINOv2). Dermatology encoders trained on
Derm1M may have seen PAD-UFES-20 or ISIC photos, which would leak test images into retrieval. A dermatology encoder is tried only as an
ablation, with that risk stated.

### 5.2 Metrics and the ground truth each needs

| Group | Metric | Ground truth needed | How obtained |
|---|---|---|---|
| Diagnosis | **Balanced accuracy (main)**, macro-F1, per-class recall and precision, confusion matrix, top-3 balanced accuracy | the source label | exists |
| Answer validity | invalid-answer rate | the answer format | automatic |
| Uncertainty | misdiagnosis detection AUROC (from `confidence`), sensitivity and specificity of the flag, coverage, selective risk, area under the risk–coverage curve | the label | automatic |
| Retrieval (K) | Recall@k, nDCG@k, MRR | silver relevance: a passage is relevant when its topics include the true diagnosis | corpus topic labels (topical, not proof of support: stated as a limitation) |
| Retrieval (C) | precision@k of precedent labels; share of the true class among precedents | precedents' labels | exists |
| Answer relevance | share of claims about the chosen diagnosis; validity | the answer itself | automatic. A free-text "answer relevancy" score (RAGAS) adds little to a constrained answer, so it is optional. |
| Citation correctness | citation validity (exact); citation precision and recall (ALCE, Gao et al. 2023) | which cited passages support each claim | judged (NLI or LLM), validated on the researcher's sample |
| Faithfulness | share of claims supported by the evidence given (P ∪ C ∪ K) | support judgements | judged, validated as above |
| **Hallucination** | (1) **confident misdiagnosis**: wrong with "yes" or high confidence (selective risk); (2) **invented patient facts**: share of patient findings that contradict or are absent from the input; (3) **invalid citations**; (4) **unsupported claims**: 1 − faithfulness; (5) **evidence-induced errors**: right answers turned wrong by misleading evidence (E6) | label, record, given evidence | (1)–(3) automatic and exact; (4) judged; (5) automatic |
| Visual claims (optional) | share of visual findings contradicting expert attributes | Derm7pt or SKINCON attributes | exists only on those cohorts |

**Statistics (as in Version 1's `docs/method.md`):**
- Paired bootstrap intervals, resampling by patient or article; McNemar with Holm as support.
- Pre-registered primary comparisons, per cohort:
  - **Proposed against RAG²** on balanced accuracy;
  - **Proposed against LLM-only** on unsupported claims and on confident misdiagnosis.

### 5.3 Experiments (the recommendations' E1–E6, mapped)

| Id | Input | Approaches | Cohort | Purpose |
|---|---|---|---|---|
| E1 | image only | all 4 | PAD, case reports | image baseline and its retrieval gain |
| E2 | image + genuine record or case text | all 4 | PAD, case reports | multimodal gain (paired with E1 and the text view) |
| E2t | text only | all 4 | PAD, case reports | the text view: completes the modality comparison |
| E3 | image + expert visual attributes | LLM-only, proposed | Derm7pt (optional) | does structured visual evidence help |
| E4 | image + record + knowledge | standard RAG, RAG², proposed | PAD | evidence-based QA (the core result) |
| E5 | record with fields removed (none, symptoms, history, all) | LLM-only, proposed | PAD | behaviour with missing information; fields are removed, never invented |
| E6 | the top evidence replaced by passages about the most confusable wrong class | standard RAG, RAG², proposed | PAD | robustness to misleading evidence; does the flag catch it |
| A1–A5 | proposed with one part removed (findings, differential retrieval, precedents, citation rule, check) | proposed | PAD, validation then test | which parts matter |

### 5.4 Models and cost control

- **Development** (prompts, k, thresholds, ablations) on the **validation split with one low-cost model**.
- **The final test grid** with the four models runs only for the main comparisons (E1, E2, E2t, E4). The ablations, E5 and E6 use one or
  two models.
- **Rough cost** (◻ ±50%; recomputed from real token counts before anything is sent):
  - full grid: 4 models × 4 approaches × 3 views × about 790 test questions = **$250–400**;
  - reduced plan as above: **$80–150**;
  - development: **$10–30**.
- **GPU** (only for a faithful RAG² filter):
  - scoring perplexities with an open 7B vision-language model;
  - training Flan-T5-large (the paper notes one 24 GB GPU is enough);
  - a few GPU hours; Colab or Kaggle GPUs may do.

---

## 6. Repository structure (proposal)

```
thesis_research/
├── archive/v1/            Version 1, frozen: data sheets, tables, reports, results, docs (see section 7, stage 1)
├── data/
│   ├── sources/           one folder per external source: README (URL, DOI, licence, date), checksums; raw files git-ignored
│   │   ├── pad_ufes_20/
│   │   ├── pmc_case_reports/
│   │   └── bcn20000/      (optional)
│   ├── cohorts/           one master table per cohort, its build records, report and data sheet
│   │   ├── pad/
│   │   └── case_reports/
│   ├── views/             image_only/, text_only/, multimodal/: generated question tables (no hand edits)
│   └── knowledge/         corpus (passages, documents, licences), topic labels, indexes, precedent-case memory
├── src/
│   ├── build/             builders per cohort, views and knowledge (from Version 1's builders)
│   ├── retrieval/         encoders, indexes, balanced retrieval, reranker, filter
│   ├── approaches/        llm_only.py, standard_rag.py, rag2.py, proposed.py
│   ├── models.py, prompts.py, run.py, datasets.py, common.py   (kept, adapted)
│   └── evaluate/          classification, uncertainty, retrieval, citations, faithfulness, hallucination, statistics
├── configs/               models.toml, approaches.toml, experiments.toml (E1–E6, A1–A5: pre-registered)
├── prompts/<approach>/<view>.txt
├── results/<cohort>/<view>/<approach>/<model>/<run>/
├── docs/                  method.md, datasets.md, knowledge.md, results.md, decisions.md (dated log of choices)
└── tests/
```

| Version 1 item | Action |
|---|---|
| `data/text`, `data/cases` | **Revise** into `data/cohorts/case_reports` (same files, same fingerprints) |
| `data/knowledge` (library, index) | **Keep, revise**: add PAD classes and evidence types; the image cards of the old image dataset go to the archive |
| `data/image`, `data/image_metadata` | **Archive** (copied to `archive/v1`; removed from the working tree only after approval) |
| `results/phase1` | **Archive** |
| `src/metrics.py`, `src/evaluate.py`, `src/run.py`, `src/models.py`, `src/prompts.py`, `src/datasets.py` | **Keep, revise** (new answer format, cohorts and views, new metrics) |
| `src/build/image*`, shortcut checks | **Archive** the image-specific steps; reuse the shortcut check for PAD's record-only check |
| `docs/results.md`, `docs/method.md` | **Archive** the results as Version 1; **revise** the method page for Version 2 |
| `configs/models.toml` | **Keep** |
| Removal | **Nothing is removed** without approval; the Git history keeps everything in any case |

---

## 7. Stage 1: preserving Version 1 (proposal, not done yet)

1. **Version 1 = commit `a1c9209`** on `main` (2026-11-19), named in every archived file. (Tags stay off until the final version, as
   decided on 2026-11-18.)
2. **`archive/v1/`** gets copies of the committed data sheets, tables and reports of every dataset, `results/phase1`, `docs/` and
   `configs/`.
   - Each copied document gets one first line: "Version 1 (commit a1c9209), archived on <date>; kept for reference, not used by
     Version 2."
   - Git stores identical files once, so the copies add almost nothing to the repository's size (it is 56 MB of history, 15 MB of
     data and results).
3. **The tests that read `results/phase1` and `docs/results.md`** are pointed at the archive copy, so that they keep proving the
   Version 1 numbers.
4. **On the laptop:** a ZIP of what Git does not hold (the image files, the case texts with research-only licences, the raw downloads),
   kept outside the repository.
5. **The originals stay in place** until the researcher approves moving them (stage 3).

---

## 8. Prioritised roadmap, with approval gates

| Stage | Work | Gate before it starts | Cost |
|---|---|---|---|
| 1 | Archive Version 1 (section 7) | **G1: approve this proposal's section 7** | none |
| 2 | Download PAD-UFES-20 from Mendeley on the laptop; confirm licence, fields, counts (all ◻ in this page); data sheet draft | G2: approve the cohort choice (section 2.2) | none |
| 3 | Build the PAD cohort (patient split, template, leakage checks, record-only shortcut check); move the case-report cohort into the new layout; generate the three views | approve the schemas (section 3) | none |
| 4 | Extend the knowledge corpus (PAD classes, guidelines); topic labels; MedCPT index; precedent memory | approve section 4.3 | none (CPU) |
| 5 | Fix `json-v2`, the prompts and the evaluation code for every metric, with tests and deliberate-break checks; validate the judge on the researcher's annotated sample | approve sections 4.1 and 5.2 | judge calls on the sample: a few dollars |
| 6 | **G3: RAG² filter decision.** With a GPU: label and train the filter. Without: "RAG² without filter", documented | the researcher's answer on GPU access | GPU hours or none |
| 7 | Development on validation, one model: all approaches, ablations A1–A5, choose k and thresholds; freeze everything | **G4: approve the development budget** | $10–30 |
| 8 | Test grid: E1, E2, E2t, E4 with 4 models; E5, E6 with 1–2 models; one run each, frozen settings | **G4b: approve the test budget** | $80–150 (reduced) |
| 9 | Optional: BCN20000 cohort, Derm7pt attributes (E3) | **G5** | small |
| 10 | Analysis, results page, limitations; then the final version, tags, and the planned clean-up of the repository | – | none |

**Decisions needed from the researcher now:**
1. Approve stage 1 as described in section 7, or change it.
2. Approve the cohorts: PAD-UFES-20 plus the case reports as the core; BCN20000 and Derm7pt as optional; Kaggle archived.
3. Is a GPU available (own, university, Colab or Kaggle) for a faithful RAG² filter?
4. Keep all four models for the test grid, or fewer?

---

## Sources checked this session

- Base paper: Sohn et al., *Rationale-Guided Retrieval Augmented Generation for Medical Question Answering*, NAACL 2025, pp. 12739–12753 (read in full).
- [RAG² code](https://github.com/dmis-lab/RAG2)
- [Derm1M](https://github.com/SiyuanYan1/Derm1M)
- SCIN: [repository](https://github.com/google-research-datasets/scin), its [data use licence](https://github.com/google-research-datasets/scin/blob/main/LICENSE) and [schema](https://github.com/google-research-datasets/scin/blob/main/dataset_schema.md)
- [Fitzpatrick 17k](https://github.com/mattgroh/fitzpatrick17k)
- [Derm7pt](https://github.com/jeremykawahara/derm7pt)
- PAD-UFES-20: [Mendeley Data](https://data.mendeley.com/datasets/zr7vgbcyr2/1) and the [Data in Brief article](https://pmc.ncbi.nlm.nih.gov/articles/PMC7479321/) (search results only; the pages themselves could not be reached from here)
- ISIC 2019: [challenge data page](https://challenge.isic-archive.com/data/), [BCN20000 paper](https://ar5iv.arxiv.org/html/1908.02288), [FLamby Fed-ISIC2019](https://owkin.github.io/FLamby/fed_isic.html) (search results)
- SKINCON: [project site](https://skincon-dataset.github.io), [paper](https://ar5iv.arxiv.org/html/2302.00785) (search results)
- SkinCAP: [paper](https://arxiv.org/html/2405.18004v2), [dataset](https://huggingface.co/datasets/joshuachou/SkinCAP) (search results)
- This repository: the data sheets of `data/image`, `data/text`, `data/cases`, `data/knowledge`, and `docs/results.md`.

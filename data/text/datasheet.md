# Data sheet: the skin-lesion text dataset (version 1)

This data sheet follows *Datasheets for Datasets* (Gebru et al., 2021). It is written by hand and changes nothing. Its numbers come from the files the preparation code (`src/text/prepare.py`, `src/text/audit.py`, `src/text/library.py`) writes:
- `dataset_report.md`: the full report of every step
- `dataset.csv`: one row per case
- `case_candidates.csv`, `abstract_candidates.csv`, `article_mesh.csv` and `term_check.csv`: what each step found
- `library.csv`: one row per passage of the reference library

The error rate in section 5 comes from reading questions: the sample the audit writes into `sources/audit_sample.txt`, and every question under 30 words. That reading is recorded in section 7.

## At a glance

| | |
|---|---|
| Cases | 1,049 patient descriptions from 1,022 published case reports, in English, one diagnosis each |
| Classes | the image dataset's 8: actinic keratosis, basal cell carcinoma, dermatofibroma, melanoma, nevus, pigmented benign keratosis, squamous cell carcinoma, vascular lesion |
| Task | name the class from the description of the patient and the lesion |
| Questions | the case text word for word, up to the sentence that names the diagnosis or reports a test result; at least 25 words, median 85 |
| Splits | test 249, validation 89, train 711; no article is in two splits |
| Labels | automatic, from the article's title and case text, and from PubMed's subject headings where PubMed has indexed the article (320 cases). No person chose, labelled or cut a case. Reading 102 questions puts the share of wrong labels at about 13% (95% interval roughly 6–25%) |
| Sources of the texts | PMC-Patients (461), the open full texts of PubMed Central (367), PubMed abstracts (221) |
| Reference library | 5,553 passages from 1,267 documents in four corpora, as in RAG²: PubMed abstracts of reviews, PubMed Central full texts of reviews, clinical practice guidelines, StatPearls chapters |
| Built from | PMC-Patients V2 (Zhao et al., 2023), PubMed, PubMed Central and MeSH (asked through NCBI's E-utilities on 2026-11-13), StatPearls (NCBI Bookshelf, Open Access) |
| Version | 1, frozen on 2026-11-13; marked in Git by the tag `text-dataset` |

## 1. Motivation

**Purpose.** The dataset was prepared for the text phase of an MS thesis on diagnosing skin lesions with large language models (LLMs). The models are tested in two ways:
- closed-book: the model sees only the case
- open-book: retrieval-augmented generation following RAG² (Sohn et al., 2025), with the reference library as the texts to retrieve from

The classes are those of the image dataset, so that the results can be compared and later joined in the multimodal phase.

**Requirement.** The dataset had to be built entirely from public text sources, by a method that identifies, extracts, organises and maps the information systematically, with no manual role for the researcher in extraction, classification, mapping or construction. Every step is therefore one command, and every decision is a rule in the code or in `class_terms.csv` that applies to all cases alike.

**Who.** Prepared by Mahwish Zafar for her MS thesis.

## 2. Composition

**Instances.** Each case is the description of one patient in one published case report: who the patient is, the history and what the lesion looks like. The question stops before the sentence that gives the diagnosis away. Case reports are published without the patient's name; the dataset adds nothing about any person.

**Cases per class and split, and where they come from**

| Class | Test | Val | Train | Cases | Articles | PMC-Patients | PMC full text | PubMed abstract | PubMed's headings agree | Not indexed by PubMed | Words (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| actinic keratosis | 6 | 2 | 2 | 10 | 9 | 6 | 3 | 1 | 2 | 8 | 120 |
| basal cell carcinoma | 40 | 15 | 157 | 212 | 205 | 91 | 81 | 40 | 55 | 157 | 87.5 |
| dermatofibroma | 27 | 9 | 9 | 45 | 41 | 23 | 7 | 15 | 16 | 29 | 72 |
| melanoma | 40 | 15 | 206 | 261 | 256 | 113 | 97 | 51 | 93 | 168 | 77 |
| nevus | 40 | 15 | 47 | 102 | 102 | 49 | 18 | 35 | 52 | 50 | 67 |
| pigmented benign keratosis | 28 | 9 | 10 | 47 | 47 | 21 | 16 | 10 | 15 | 32 | 90 |
| squamous cell carcinoma | 40 | 15 | 270 | 325 | 315 | 131 | 132 | 62 | 79 | 246 | 99 |
| vascular lesion | 28 | 9 | 10 | 47 | 47 | 27 | 13 | 7 | 8 | 39 | 124 |
| **Total** | **249** | **89** | **711** | **1,049** | **1,022** | **461** | **367** | **221** | **320** | **729** | **85** |

**Unequal classes.** Public case reports are few for some classes: actinic keratosis has 10 cases (6 in test), and dermatofibroma, pigmented benign keratosis and vascular lesion have 45–47. The split rules (section 4) give the small classes most of their cases for test.

**What comes with each case**
- **`dataset.csv`:**
  - its split, class, file and SHA-256
  - its source (`PMC-Patients`, `PMC full text` or `PubMed abstract`), its PMC-Patients number (`patient_uid`) and its PubMed number (`pmid`)
  - whether PubMed's subject headings agree with the class or PubMed has not indexed the article (`pubmed`)
  - the publication year, the patient's age and sex where the source gives them, and the number of words
- **`case_candidates.csv` and `abstract_candidates.csv`:** each candidate's article title and the marks that kept a candidate out
- **Not in Git:** the question texts themselves (`data/text/dataset/`) and the downloaded sources (`data/text/sources/`)

**Several cases from one article.** 20 articles give two or more cases (47 cases in all). Every article is whole in one split.

**Recommended splits.** The splits are fixed and part of the dataset (the `split` column). They must not be redrawn.

**Licences of the texts**
- PMC-Patients: CC BY-NC-SA 4.0
- PubMed Central full texts: each article's own licence, kept with its text in `data/text/sources/pubmed/`
- PubMed abstracts: the publisher's copyright

The texts are therefore used for research only and are not in Git.

## 3. Collection

- **PMC-Patients.** `PMC-Patients-V2.json` from the Hugging Face repository `zhengyun21/PMC-Patients`, version `28d8836518f86d4f1e6358ea8ec09977023e5766`, downloaded on 2026-11-12. It holds 250,294 patient descriptions taken from case reports in PubMed Central whose licence allows reuse. SHA-256 `7bf27ed02dbadd5501f9c89b1d31d8c07d3559cd9b30c6f0b1a84734125ed593`.
- **PubMed case reports.** Asked on 2026-11-13 through NCBI's E-utilities: each class's case reports on the skin, by its MeSH headings, and the case reports PubMed has not indexed, by the class's names in their title. 27,820 records not already in PMC-Patients were downloaded. For the 1,382 candidate articles with a PubMed Central number, the open full text was asked for; 771 gave case sections, which are used instead of the abstract.
- **MeSH.** Every name in `class_terms.csv` was checked against MeSH, the US National Library of Medicine's subject headings (`term_check.csv`). The subject headings of all 16,029 candidate articles were asked from PubMed (`article_mesh.csv`); 10,267 are indexed.
- **The names of the classes.** `class_terms.csv` lists 120 names doctors use, and whether each counts for its class. They come from the ISIC records of the image dataset, the HAM10000 group definitions (Tschandl et al., 2018), and names proposed and then checked by the survey against MeSH and PubMed.
- **Reference library.** PubMed records and PubMed Central full texts asked on 2026-11-13, and StatPearls from the NCBI Bookshelf Open Access archive (`statpearls_NBK430685.tar.gz`, 2,047,364,463 bytes, SHA-256 `08117bcff46ed366a7ab9033d134b81bf345e81cb9a1537b6075400418bb09bd`).
- **Nothing new.** No text was written or changed by the researcher, and no person was contacted.

## 4. Preparation

Each step is one command. Its full numbers, the queries and the word lists are in the matching section of `dataset_report.md`.

| Step | What it did | Result |
|---|---|---|
| survey | Checked every name in `class_terms.csv` against MeSH and counted it in PubMed | 120 names: 32 MeSH headings, 17 entry terms, 71 not in MeSH (68 of them used in PubMed) |
| cases | Read every PMC-Patients patient; a patient is a candidate when its article's title names a class | 3,796 candidates from 3,343 articles |
| abstracts | Downloaded PubMed's case reports of each class not in PMC-Patients, with open full texts where PMC gives them, and judged them by the same rules | 27,820 downloaded; 12,686 candidates |
| mesh | Asked PubMed whether each candidate article is indexed, and which classes its headings name on the skin | 16,029 articles; 10,267 indexed |
| build | Labelled the cases, cut the questions, applied the checks and made the splits, by the rules below | 1,049 cases |
| audit | Checked the built dataset with code of its own | All 14 checks passed |
| library | Built the reference library | 5,553 passages |

**Label.** A case gets class X when:
1. the article's title names X, by a name that counts in `class_terms.csv`, and no other class, with no mark (for example a site that may not be skin, a name of another condition, X named only as a look-alike, or the bare word "nevus" without a kind of mole), and
2. the case text names X, by a full name or a short form, and
3. where PubMed has indexed the article, its subject headings name X on the skin. 40 cases were left out because the headings disagree.

**Question.** The case text word for word, up to the first sentence that names the case's own class (or its short form) or reports a test result such as a biopsy or the histology. When an earlier background sentence already names the class, the question starts at the first sentence about the patient. Names of the other classes stay, as clinical information. Nothing is rewritten.

**Kept when** the question has at least 25 words, describes a lesion, a patient (an age or a sex) and what the lesion looks like, and puts no lesion inside the body. When the questions of two articles tell the same patient's case, the article PubMed listed first keeps it (1 case left out).

**Splits.** Every article is whole in one split, in a fixed random order (seed `text-dataset`). For each class, 60% of its cases go into test (at most 40), 20% into validation (at most 15) and the rest into train.

**How the rules were made.** The rules were set in rounds during development. After each round, a sample of questions written by the audit was read; every kind of error found became a general rule in the code or in `class_terms.csv`, and the whole dataset was built again. No single case was ever kept, dropped or relabelled by hand. Three readings of 40 questions, of earlier versions, found 5 to 11 wrong. After the last rule, the audit drew a new sample with a new seed (`text-audit-4`) for the final reading. Then the shortest question kept was lowered from 30 to 25 words, to keep more cases, and all 62 questions this added were read too (section 7).

**Not done:** no case was chosen, corrected or rewritten by a person, no dermatologist checked the labels, and no text was translated or summarised.

## 5. Uses

**Intended uses**
- **Test split:** the reported result, run once per model and setting.
- **Validation split:** tuning prompts and retrieval settings. It is never reported as a result.
- **Train split:** the only cases a run may use as examples or retrieve from. Validation and test cases must never be used that way.
- **Reference library:** the texts for open-book runs. No case report and no candidate article of the dataset is in it, and passages that share 12 words in a row with a question are left out.
- **Scoring:** balanced accuracy is the main score, because the classes are unequal, with 95% intervals that resample whole articles. The per-class recall and the confusion matrix are reported with it.

**Known limitations: state them with the results**
1. **About one label in eight is wrong.** Two readings give this (section 7):
   - the 987 cases of at least 30 words: a sample of 5 questions per class found 5 of 40 wrong, 12.5% (95% interval 5–26%)
   - the 62 cases of 25–29 words, added last: all read, 9 wrong (15%)

   Together, about 13% of the 1,049 labels are wrong (95% interval roughly 6–25%). The errors found:
   - a lesion that is not on the skin (5): a melanoma spread to the heart, a melanoma of the eye, growths in the intestine, two growths on the gum
   - a case that only looked like the class (2): bleeding in the skin of a toe that looked like melanoma; mycosis fungoides that looked like a dermatofibroma
   - an article about another diagnosis (3): another tumour inside or beside a seborrheic keratosis (2); white patches of vitiligo in an article on nevi
   - not one patient's case (4): two summaries of studies, two patients in one question, and a case series where this patient's own diagnosis is not given

   Collision tumours (squamous cell carcinoma together with another tumour in one lesion) were counted as right, because squamous cell carcinoma is the only one of the 8 classes in them.
2. **No dermatologist reviewed the labels.** Each label rests on sources that agree: the authors' title and case text, and, for 320 cases, PubMed's subject headings, which the US National Library of Medicine gives independently.
3. **Small classes.** Actinic keratosis has 6 test cases, so its recall cannot be measured with any precision. With 27–40 test cases per class, a single class's recall has a 95% interval of roughly ±15–20 percentage points.
4. **The models may have read these cases.** The articles were published from 1977 to 2026 (median 2020; 273 cases from 2024 on). A closed-book score may partly measure memory.
5. **The questions are word for word.** They keep what the authors wrote, including:
   - other diagnoses that were suspected
   - dermoscopy terms that point to a class
   - treatments
   - empty figure references such as "(Fig. )"

   Some cases are therefore easier, and some harder, than a clinical note.
6. **Three kinds of source.** PubMed abstracts give shorter cases than PMC-Patients and full texts.
7. **Site words are a list.** A lesion inside the body is left out only when its sentence uses a word of the list. 5 of the 102 questions read describe a lesion that is not on the skin.
8. **A later rebuild gives a different dataset.** PubMed and PubMed Central change every day, so the same commands run later find other articles. Version 1 is the one in `dataset.csv`, with a fingerprint for every question (section 7).
9. **The library is uneven.** Dermatofibroma, pigmented benign keratosis and vascular lesion have few reviews and no guidelines.

**Not intended for**
- clinical decisions
- estimates of how common a disease is: case reports favour rare and unusual cases
- conclusions about skin types, which are not recorded

## 6. Distribution

**What is shared and what isn't**
- **Not redistributed:** the question texts (`data/text/dataset`), the downloaded sources (`data/text/sources`) and the library's passages (`data/text/library`) stay on the researcher's computer and are not in Git.
- **In the repository:** the tables, the report, this data sheet and the code: article numbers, fingerprints, years and lengths, but no case text.

**Licences and credit**
- Quoting a case in a publication must respect its licence: PMC-Patients CC BY-NC-SA 4.0, a full text's own licence, or the publisher's copyright of an abstract.
- Each case's article can be found by its PubMed number in `dataset.csv`.

**Cite**
- PMC-Patients (Zhao et al., 2023)
- PubMed and PubMed Central (US National Library of Medicine)
- StatPearls (StatPearls Publishing, NCBI Bookshelf)
- RAG² (Sohn et al., 2025), for the four corpora of the library

## 7. Maintenance and checking

**Version 1** is frozen and marked in Git by the tag `text-dataset`. Any change to a case, label, split or library passage makes a new version, with new fingerprints, a new tag and an entry in this data sheet.

**Checks.** The build checks that every question is part of its case text word for word, that no question names its class, and that no article is in two splits. The audit then checks every question again with code of its own: against its fingerprint and its source, for its class's names and short forms (also those its source defines), for test results, for length, for a patient, a lesion and what the lesion looks like, for a site inside the body, and for the same patient in two articles. All 14 checks passed on all 1,049 questions.

**Final reading (2026-11-13).** Questions were read against their article's title, in two parts:
- **Sample:** the audit's 5 questions per class (seed `text-audit-4`, drawn after the last rule, when the dataset had the 987 questions of at least 30 words). 26 of the 40 had not been in any earlier sample. 5 are wrong.
- **The shortest questions:** all 62 questions of 25–29 words, which lowering the minimum from 30 to 25 words added. 9 are wrong.

Of the 28 test questions read, 4 are wrong.

| Case | Class | Split | Read in | What is wrong |
|---|---|---|---|---|
| `pubmed-38028384` | melanoma | train | sample | a melanoma spread to the heart: the question describes the heart mass |
| `5439066-1` | melanoma | train | sample | bleeding into the outer layer of the skin of a toe (intracorneal haematoma) that looked like melanoma; the title names melanoma only as its look-alike |
| `10056737-4` | squamous cell carcinoma | train | sample | a case series of squamous cell carcinoma and pseudoepitheliomatous hyperplasia: this patient's own diagnosis is not given |
| `4678085-1` | vascular lesion | val | sample | growths on the gum, inside the mouth |
| `pubmed-35395702` | vascular lesion | train | sample | a study's results, not a case |
| `pubmed-22059953` | dermatofibroma | test | shortest | mycosis fungoides that looked like a dermatofibroma |
| `pubmed-12081162` | pigmented benign keratosis | test | shortest | a ganglioneuroma that grew inside a seborrheic keratosis: the question is about the new growth |
| `pubmed-1955228` | squamous cell carcinoma | test | shortest | two patients in one question |
| `pubmed-27891487` | vascular lesion | test | shortest | a growth on a newborn's gum, inside the mouth |
| `pubmed-10343473` | pigmented benign keratosis | val | shortest | a poroid tumour beside a seborrheic keratosis |
| `pubmed-2326012` | melanoma | train | shortest | a melanoma of the eye |
| `pubmed-3617232` | melanoma | train | shortest | growths in the intestine |
| `6287087-1` | nevus | train | shortest | the question describes white patches of vitiligo |
| `pubmed-32518541` | squamous cell carcinoma | train | shortest | a summary of a case series, not one patient |

**Rebuild**
1. Set up Python 3.11 or newer with the packages in `requirements.txt`.
2. Run, in order: `python -m src.text.prepare survey`, `cases`, `abstracts`, `mesh`, `build`, then `python -m src.text.audit` and `python -m src.text.library`. The steps download their sources themselves.

Because PubMed changes (section 5), a rebuild reproduces the method, not version 1. Version 1's texts come back only from the backup.

**Fingerprints**

| File | SHA-256 |
|---|---|
| `dataset.csv` | `b0a6544cd99c1495147877acb4732670a85da3acd09e6c62a3d41f4f868232ee` |
| `library.csv` | `ca8d6969379c08a2f837d47cf9359ab40d116001cfe7219ca7568a82becf710b` |

Every question file has its own SHA-256 in `dataset.csv`, and every passage in `library.csv`; the audit checks the questions against them.

**Backup.** Keep a ZIP of `data/text/dataset`, `data/text/sources` and `data/text/library` outside this computer.

## References

- Gebru, T., Morgenstern, J., Vecchione, B., Vaughan, J. W., Wallach, H., Daumé III, H., and Crawford, K. (2021). Datasheets for datasets. *Communications of the ACM*, 64(12), 86–92.
- Sohn, J., et al. (2025). Rationale-guided retrieval augmented generation for medical question answering. In *Proceedings of NAACL 2025*, 12739–12753.
- Tschandl, P., Rosendahl, C., and Kittler, H. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data*, 5, 180161.
- Zhao, Z., Jin, Q., Chen, F., Peng, T., and Yu, S. (2023). A large-scale dataset of patient summaries for retrieval-based clinical decision support systems. *Scientific Data*, 10, 909.
- StatPearls. StatPearls Publishing, Treasure Island (FL). NCBI Bookshelf, NBK430685: https://www.ncbi.nlm.nih.gov/books/NBK430685/
- PubMed and PubMed Central, US National Library of Medicine, accessed through NCBI's E-utilities: https://www.ncbi.nlm.nih.gov/books/NBK25501/

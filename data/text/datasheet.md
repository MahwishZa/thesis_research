# Data sheet: the skin-lesion text dataset (version 2)

This data sheet follows *Datasheets for Datasets* (Gebru et al., 2021). It is written by hand and changes nothing. Its numbers come from the files the preparation code (`src/text/prepare.py`, `src/text/audit.py`, `src/text/library.py`) writes:
- `dataset_report.md`: the full report of every step
- `dataset.csv`: one row per case
- `case_candidates.csv`, `abstract_candidates.csv`, `article_mesh.csv` and `term_check.csv`: what each step found
- `library.csv`: one row per passage of the reference library

The error rate in section 5 comes from reading questions together with the text that follows each one in its source: the sample the audit writes into `sources/audit_sample.txt`. That reading is recorded in section 7.

## At a glance

| | |
|---|---|
| Cases | 745 patient descriptions from 726 published case reports, in English, one diagnosis each |
| Classes | the image dataset's 8: actinic keratosis, basal cell carcinoma, dermatofibroma, melanoma, nevus, pigmented benign keratosis, squamous cell carcinoma, vascular lesion |
| Task | name the class from the description of the patient and the lesion |
| Questions | the case text word for word, up to the sentence that names the diagnosis or reports a test result; at least 25 words, median 90 |
| Splits | test 221, validation 80, train 444; no article is in two splits |
| Labels | automatic: the article's title and case text name the class, a test result in the case text names it, and PubMed's subject headings agree where PubMed has indexed the article (220 cases). No person chose, labelled or cut a case. Reading 74 questions that no rule was drawn from found 2 wrong labels (2.7%, 95% interval 0.3–9.4%) and 3 mixed lesions (wrong or mixed: 6.8%, 95% interval 2.2–15.1%) |
| Sources of the texts | PMC-Patients (342), the open full texts of PubMed Central (264), PubMed abstracts (139) |
| Reference library | 5,553 passages from 1,267 documents in four corpora, as in RAG²: PubMed abstracts of reviews, PubMed Central full texts of reviews, clinical practice guidelines, StatPearls chapters |
| Built from | PMC-Patients V2 (Zhao et al., 2023), PubMed, PubMed Central and MeSH (asked through NCBI's E-utilities on 2026-11-13 and 2026-11-14), StatPearls (NCBI Bookshelf, Open Access) |
| Version | 2, frozen on 2026-11-14; marked in Git by the tag `text-dataset`. Version 1 (1,049 cases) was withdrawn (section 7) |

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
| actinic keratosis | 2 | 1 | 1 | 4 | 4 | 2 | 1 | 1 | 1 | 3 | 107 |
| basal cell carcinoma | 40 | 15 | 121 | 176 | 170 | 80 | 69 | 27 | 43 | 133 | 95 |
| dermatofibroma | 21 | 7 | 7 | 35 | 31 | 19 | 6 | 10 | 11 | 24 | 75 |
| melanoma | 40 | 15 | 125 | 180 | 176 | 81 | 70 | 29 | 63 | 117 | 78 |
| nevus | 40 | 15 | 24 | 79 | 79 | 38 | 14 | 27 | 40 | 39 | 74 |
| pigmented benign keratosis | 22 | 7 | 7 | 36 | 36 | 17 | 13 | 6 | 10 | 26 | 97 |
| squamous cell carcinoma | 40 | 15 | 153 | 208 | 203 | 87 | 83 | 38 | 49 | 159 | 106 |
| vascular lesion | 16 | 5 | 6 | 27 | 27 | 18 | 8 | 1 | 3 | 24 | 151 |
| **Total** | **221** | **80** | **444** | **745** | **726** | **342** | **264** | **139** | **220** | **525** | **90** |

**Unequal classes.** Public case reports that pass the rules are few for some classes: actinic keratosis has 4 cases (2 in test), vascular lesion 27, dermatofibroma 35 and pigmented benign keratosis 36. The split rules (section 4) give the small classes most of their cases for test.

**What comes with each case**
- **`dataset.csv`:**
  - its split, class, file and SHA-256
  - its source (`PMC-Patients`, `PMC full text` or `PubMed abstract`), its PMC-Patients number (`patient_uid`) and its PubMed number (`pmid`)
  - whether PubMed's subject headings agree with the class or PubMed has not indexed the article (`pubmed`)
  - the publication year, the patient's age and sex where the source gives them, and the number of words
- **`case_candidates.csv` and `abstract_candidates.csv`:** each candidate's article title and the marks the candidate steps gave it
- **Not in Git:** the question texts themselves (`data/text/dataset/`) and the downloaded sources (`data/text/sources/`)

**Several cases from one article.** 15 articles give two or more cases (34 cases in all). Every article is whole in one split.

**Recommended splits.** The splits are fixed and part of the dataset (the `split` column). They must not be redrawn.

**Licences of the texts**
- PMC-Patients: CC BY-NC-SA 4.0
- PubMed Central full texts: each article's own licence, kept with its text in `data/text/sources/pubmed/`
- PubMed abstracts: the publisher's copyright

The texts are therefore used for research only and are not in Git.

## 3. Collection

- **PMC-Patients.** `PMC-Patients-V2.json` from the Hugging Face repository `zhengyun21/PMC-Patients`, version `28d8836518f86d4f1e6358ea8ec09977023e5766`, downloaded on 2026-11-12. It holds 250,294 patient descriptions taken from case reports in PubMed Central whose licence allows reuse. SHA-256 `7bf27ed02dbadd5501f9c89b1d31d8c07d3559cd9b30c6f0b1a84734125ed593`.
- **PubMed case reports.** Asked on 2026-11-13 through NCBI's E-utilities: each class's case reports on the skin, by its MeSH headings, and the case reports PubMed has not indexed, by the class's names in their title. 27,820 records not already in PMC-Patients were downloaded. For the 1,382 candidate articles with a PubMed Central number, the open full text was asked for; 771 gave case sections, which are used instead of the abstract.
- **MeSH.** Every name in `class_terms.csv` was checked against MeSH, the US National Library of Medicine's subject headings (`term_check.csv`, asked again on 2026-11-14 when a name was added; the headings of the names that count did not change). The subject headings of all 16,029 candidate articles were asked from PubMed on 2026-11-13 (`article_mesh.csv`); 10,267 are indexed.
- **The names of the classes.** `class_terms.csv` lists 121 names doctors use, and whether each counts for its class. They come from the ISIC records of the image dataset, the HAM10000 group definitions (Tschandl et al., 2018), and names proposed and then checked by the survey against MeSH and PubMed. One name was added after reading cases: acral pseudolymphomatous angiokeratoma (APACHE) counts for no class, because its case reports find a pseudolymphoma, not an angiokeratoma.
- **Reference library.** PubMed records and PubMed Central full texts asked on 2026-11-14, and StatPearls from the NCBI Bookshelf Open Access archive (`statpearls_NBK430685.tar.gz`, 2,047,364,463 bytes, SHA-256 `08117bcff46ed366a7ab9033d134b81bf345e81cb9a1537b6075400418bb09bd`).
- **Nothing new.** No text was written or changed by the researcher, and no person was contacted.

## 4. Preparation

Each step is one command. Its full numbers, the queries and the word lists are in the matching section of `dataset_report.md`.

| Step | What it did | Result |
|---|---|---|
| survey | Checked every name in `class_terms.csv` against MeSH and counted it in PubMed | 121 names: 32 MeSH headings, 17 entry terms, 72 not in MeSH (69 of them used in PubMed) |
| cases | Read every PMC-Patients patient; a patient is a candidate when its article's title names a class | 3,796 candidates from 3,343 articles |
| abstracts | Downloaded PubMed's case reports of each class not in PMC-Patients, with open full texts where PMC gives them, and judged them by the same rules | 27,820 downloaded; 12,686 candidates |
| mesh | Asked PubMed whether each candidate article is indexed, and which classes its headings name on the skin | 16,029 articles; 10,267 indexed |
| build | Labelled the cases, cut the questions, applied the checks and made the splits, by the rules below | 745 cases |
| audit | Checked the built dataset with code of its own | All 17 checks passed |
| library | Built the reference library | 5,553 passages |

**Label.** A case gets class X when all of these hold:
1. **The title.** The article's title names X, by a name that counts in `class_terms.csv`, and no other class, with no mark. A mark is:
   - a name of another condition, or a site that may not be skin
   - a title about treatment, or about an animal
   - X named only as a look-alike ("... mimicking basal cell carcinoma"), or the bare word "nevus" without a kind of mole
   - another lesion: a word ending in -oma, or a polyp, that is no name in `class_terms.csv` ("cutaneous plasmacytoma adjacent to Bowenoid actinic keratosis"), unless it is named only as what the lesion looked like ("melanoma mimicking sarcoma")

   The build applies the title rules again, as they are when it runs.
2. **The case text** names X, by a full name or a short form.
3. **PubMed,** where it has indexed the article, gives subject headings that name X on the skin. 33 cases were left out because the headings disagree.
4. **A test result names X.** The case text after the question reports a test result (a biopsy, the histology or pathology, an excision, immunohistochemistry, what a microscope showed) that names X, in the sentence of the result or one of the two after it:
   - by a name that counts, its short form, or a short form the article defines right after a name of X ("nodular basal cell carcinoma (nBCC)")
   - not in a sentence that names a relative ("a BCC was found on the father's arm")
   - "keratinocyte carcinoma", a group of diseases, is no test result

   A diagnosis by eye or by dermoscopy alone, or a result that names another diagnosis, is not enough.

**Question.** The case text word for word, up to the first sentence that names the case's own class (or its short form) or reports a test result. When an earlier background sentence already names the class, the question starts at the first sentence about the patient. Names of the other classes stay, as clinical information. Nothing is rewritten.

**Kept when** the question:
- has at least 25 words
- describes one patient: an age or a sex; not "patients", a mean age, a second case, or two patients of different ages (a heading such as "Case 2." at its start does not count)
- describes a lesion as seen on the skin: a sentence names a lesion and where it is on the skin or the body's surface, without reporting a scan or a look inside the body, and that sentence or the one before or after it says what the lesion looks or feels like
- tells what the lesion looks like: a size, a lesion type such as a nodule, or a colour, surface or shape
- puts no lesion inside the body

When the questions of two articles tell the same patient's case, the article PubMed listed first keeps it (1 case left out). Of the labelled cases whose question is long enough and names a lesion, these rules left out: 124 with no patient described, 38 with more than one patient, 70 with a lesion inside the body, 231 with no lesion seen on the skin, 27 with nothing of what the lesion looks like, and 46 with no test result naming the class.

**Splits.** Every article is whole in one split, in a fixed random order (seed `text-dataset`). For each class, 60% of its cases go into test (at most 40), 20% into validation (at most 15) and the rest into train.

**How the rules were made.** The rules were set in rounds during development:
1. After each round, questions written by the audit were read, and every kind of error found became a general rule in the code or a name in `class_terms.csv`.
2. The whole dataset was then built again, and the cases each change brought back or left out were read. A change whose cases were not almost all correct was undone or corrected.
3. No single case was ever kept, dropped or relabelled by hand.

Version 1 is described in section 7. For version 2, a reading of 77 questions, each with the text after it in its source, found 12 with a problem (5 wrong labels). Five rules followed:
- another lesion in the title
- birds among the animals
- APACHE counts for no class
- "keratinocyte carcinoma" is no test result
- two patients' ages

They left out 53 cases. About 44 of them were problems and about 9 were good cases that only mention a second tumour. After the last rule, the audit drew a new sample with a new seed (`text-audit-8`) for the final reading (section 7).

**Not done:** no case was chosen, corrected or rewritten by a person, no dermatologist checked the labels, and no text was translated or summarised.

## 5. Uses

**Intended uses**
- **Test split:** the reported result, run once per model and setting.
- **Validation split:** tuning prompts and retrieval settings. It is never reported as a result.
- **Train split:** the only cases a run may use as examples or retrieve from. Validation and test cases must never be used that way.
- **Reference library:** the texts for open-book runs. No case report and no candidate article of the dataset is in it, and passages that share 12 words in a row with a question are left out.
- **Scoring:** balanced accuracy is the main score, because the classes are unequal, with 95% intervals that resample whole articles. The per-class recall and the confusion matrix are reported with it. Variants of one method (an ablation) are compared on the same questions, with a paired test.

**Known limitations: state them with the results**
1. **A few labels are wrong.** The final reading of 74 questions that no rule was drawn from (section 7) found:
   - 2 wrong labels, 2.7% (95% interval 0.3–9.4%): a "malignant blue nevus" that spread to the lungs and liver (a melanoma), and a keloid with a nevus-like growth only at one margin
   - 3 mixed lesions, where the class is confirmed but another lesion is part of the same case: a basal cell carcinoma in the wall of a cyst, basal cell carcinomas with leukaemia cells in them, and a pyogenic granuloma together with an injured finger artery. Wrong or mixed: 6.8% (95% interval 2.2–15.1%)
   - 1 question that reports the biopsy findings
2. **No dermatologist reviewed the labels.** Each label rests on sources that agree: the authors' title, the case text and a test result named in it, and, for 220 cases, PubMed's subject headings, which the US National Library of Medicine gives independently.
3. **Small classes.**
   - **Actinic keratosis** has 2 test cases, so its recall can only be 0%, 50% or 100%, yet it counts one eighth in balanced accuracy. Showing the score without it beside the 8-class score makes clear how much it moves a result.
   - **The other small classes** have 16–22 test cases (vascular lesion, dermatofibroma, pigmented benign keratosis). A single class's recall then has a 95% interval of roughly ±20 percentage points.
4. **The models may have read these cases.** The articles were published from 1977 to 2026 (median 2020). 193 cases are from 2024 on (55 in test), and 127 from 2025 on (37 in test). A closed-book score may partly measure memory, and then the gain from retrieval looks smaller. Results on the cases published after a model's training cutoff show how much.
5. **The rules favour clear cases.** A case is kept only with one patient, one lesion on the skin and a confirming test result. So collision tumours, cases diagnosed by eye and case series are left out, and the dataset is cleaner than clinical practice.
6. **The questions are word for word.** They keep what the authors wrote, including:
   - other diagnoses that were suspected
   - dermoscopy terms that point to a class
   - treatments
   - empty figure references such as "(Fig. )"

   Some cases are therefore easier, and some harder, than a clinical note.
7. **Three kinds of source.** PubMed abstracts give shorter cases than PMC-Patients and full texts.
8. **The rules read words, not meaning.** Sites inside the body, scans, relatives and look-alikes are recognised by word lists (`dataset_report.md`). A case that says it in other words passes.
9. **A later rebuild gives a different dataset.** PubMed and PubMed Central change every day, so the same commands run later find other articles. Version 2 is the one in `dataset.csv`, with a fingerprint for every question (section 7).
10. **The library is uneven.** Dermatofibroma, pigmented benign keratosis and vascular lesion have 18–35 reviews and no guidelines.

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

**Versions.** Any change to a case, label, split or library passage makes a new version, with new fingerprints, a new tag and an entry here.
- **Version 1** (2026-11-13): 1,049 cases. A reading of 102 questions put its wrong labels at about 13% (95% interval roughly 6–25%): lesions not on the skin, cases that only looked like the class, articles about another diagnosis, and questions that were not one patient's case. It was withdrawn, and its tag removed.
- **Version 2** (2026-11-14): 745 cases, made by the rules of section 4. It is frozen and marked in Git by the tag `text-dataset`.

**Checks.** The build checks that every question is part of its case text word for word, that no question names its class, and that no article is in two splits. The audit then checks every question again with code of its own:
- against its fingerprint and its source
- for its class's names and short forms (also those its source defines), and for test results
- for length, for one patient, a lesion on the skin and what the lesion looks like, and for a site inside the body
- for a test result after it in its source that names its class
- for the same patient in two articles

All 17 checks passed on all 745 questions.

**Final reading (2026-11-14).** The audit's sample, drawn with a new seed (`text-audit-8`) after the last rule, holds 10 questions per class (all 4 of actinic keratosis): 74 questions, 33 of them from test. Each was read with the text that follows it in its source, where the test result is. 68 are right in every way. The 6 others are below; 5 of them are in test (1 a wrong label).

| Case | Class | Split | What is wrong |
|---|---|---|---|
| `pubmed-9542669` | nevus | test | wrong label: a "malignant blue nevus" that spread to the lungs and liver, which is a melanoma |
| `10366870-1` | nevus | train | wrong label: a keloid; a nevus-like growth was found only at one margin |
| `pubmed-20736874` | basal cell carcinoma | test | mixed: a basal cell carcinoma in the wall of an epidermoid cyst of the eyelid |
| `2000862-1` | basal cell carcinoma | test | mixed: basal cell carcinomas with leukaemia cells in the same lesions |
| `pubmed-40606809` | vascular lesion | test | mixed: a pyogenic granuloma together with a pseudoaneurysm of a finger artery |
| `pubmed-19362750` | nevus | test | the question reports the biopsy findings (loss of maturation, HMB-45 in the dermis) |

These cases stay in the dataset: changing a single case by hand would break the rule that no person chooses a case. A later version can catch them by rules (for example "malignant blue nevus" as a name that counts for no class, and "composite", "mixed" and "developing in" as title marks), checked by a new reading.

**Rebuild**
1. Set up Python 3.11 or newer with the packages in `requirements.txt`.
2. Run, in order: `python -m src.text.prepare survey`, `cases`, `abstracts`, `mesh`, `build`, then `python -m src.text.audit` and `python -m src.text.library`. The steps download their sources themselves.

Because PubMed changes (section 5), a rebuild reproduces the method, not version 2. Version 2's texts come back only from the backup.

**Fingerprints**

| File | SHA-256 |
|---|---|
| `dataset.csv` | `f4d40e300b62b19cc61ac634daac404aaf5f0a01458342c0259c67fcb503f976` |
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

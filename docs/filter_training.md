# RAG² Filter Training Strategy

**Decided 2026-09-17** (ledger D-39, D-40). The baseline arm needs a trained
filter and the authors do not distribute one. This is how it gets trained,
what deviates from the paper, and what is still outstanding.

**Status: DECIDED AND IMPLEMENTED. NOT TRAINED.** No checkpoint exists. No
label has been generated. Nothing in this document reports a result.

---

## 1. Why the filter must be trained at all

The RAG² paper's contribution is a filter trained on **perplexity-derived
helpfulness labels**. The checkpoint is not released (ledger E9), so an
identical baseline is impossible for anyone, not just for this thesis. The
baseline is therefore an **adaptation**, and the thesis says so.

### The released sample is a format sample, not data

The official repository contains
`classifier/data/medqa/llama3_cot/5%-train.json`. Its name suggests the
paper's 5% training split.

**It was downloaded and inspected. It contains 5 examples.** The ids run to
`llama3_5%_23600`, so the real 5% split held roughly 23,600 — the released
file is an illustration of the format, carrying the four fields `id`,
`answer`, `dataset_name`, `question`.

This matters because it settles the strategy: labels cannot be obtained, only
regenerated.

## 2. The label function — kept exactly

`experiments/filter_training/labeling.py` implements the paper's decision tree
(D2 §3.2, Fig. 2, Eq. 3):

1. Answer the question **without** the passage → correct or not.
2. Answer it **with** the passage → correct or not.
3. **Correctness flip decides:** wrong → right is `[HELPFUL]`; right → wrong is
   `[NOT_HELPFUL]`.
4. **Unchanged correctness falls back to the perplexity differential** of the
   generated *rationale* (not the query — ledger E3). Top τ = 0.25 of
   reductions is `[HELPFUL]`.

Two details that are easy to get wrong and are locked by tests:

* **A flip always outranks perplexity.** A large perplexity gain cannot rescue
  a passage that turned a right answer wrong.
* **The τ quantile is computed only over pairs the tie-break actually
  judges** — the unchanged ones. Including flip-decided pairs would let pairs
  that never reach step 4 set the threshold it uses.

**τ is a property of the reference method, not a thesis parameter.** It is
fixed at 0.25 and is never fitted, tuned or swept. Fitting it would make the
baseline something this thesis chose rather than something the paper
specifies.

MedQA is multiple-choice, so correctness is checked automatically. **No human
annotation is involved and none is planned.**

## 3. Training configuration

`experiments/filter_training/config.py`. Reference values from the released
launch script (`docs/rag2_classifier_feasibility.md` §1):

| Setting | Value | Source |
|---|---|---|
| Base model | `google/flan-t5-large` (≈770M) | paper (E1) |
| Learning rate | 3e-5 | launch script |
| Optimizer | AdamW | released code |
| Max sequence length | 512 | launch script |
| Doc stride | 128 | launch script |
| Weight decay / warmup | 0.0 / 0 | released defaults |
| Label tokens | `[HELPFUL]`, `[NOT_HELPFUL]`, added with embedding resize | released code |

### Deviations, each with its reason

| Deviation | Reason |
|---|---|
| Per-device batch **4**, accumulation **4** (paper: 16 on one device) | A free 16 GB T4 cannot hold batch 16 at seq 512 for a 770M seq2seq model. **The effective batch stays 16** — `validate()` refuses a configuration where it does not, so accumulation cannot be used to quietly shrink it. |
| Epochs **below 40** | Free-session limits. The count is deliberately **unset in code**: `validate()` raises rather than default it, because what fits depends on the labelled-set size. The count actually run is reported. |
| **Subsampled training set** | See §4. |

**The base model is not reduced.** The obvious economy — dropping to
Flan-T5-base — is not taken: since the generator venue is already a free T4
(`docs/generator_contract.md`), the paper's own size trains there, and
shrinking it would give up fidelity to solve a problem that no longer exists.

## 4. The remaining cost, stated plainly

Label generation needs **two rationale generations per (question, passage)
pair**, on the same generator the experiment uses. Cost is linear in set size.

**No hours figure is given here, because generation speed on the venue has not
been measured.** The timing check in `docs/generator_contract.md` §6 is what
produces that number, and the labelled-set size is chosen from it — a budget,
not a target.

The consequence is honest and worth stating in the thesis: **a smaller
training set than the paper's yields a weaker filter.** That is a limitation
on the *baseline's strength*, not a threat to the comparison's validity:

* the same filter defines the baseline for both HAR and QA accuracy;
* the proposed system is compared against **that** baseline, never against the
  paper's published numbers;
* the **no-filter control is the floor that keeps this honest** — if the
  trained filter does not beat admitting everything, the baseline was weak,
  and any advantage for the proposed system must be read against that rather
  than celebrated.

## 5. Before a checkpoint may produce thesis numbers

`CheckpointRecord` requires all of: the base model, training and validation
set sizes, validation accuracy, epochs actually run, and the label
distribution.

* `is_usable()` returns False at or below chance (0.5). A binary classifier at
  chance has not learned the label function, and a baseline built on one would
  be **broken rather than weak** — which changes what a HAR difference means.
* `label_distribution()` is checked **before** training: a set that is 95% one
  label teaches the prior, and that is better caught in the data than
  diagnosed afterwards from a bad validation number.
* `FlanT5RAG2Filter` already refuses to run without an explicit checkpoint
  path, and `MockRAG2Filter` states in its own docstring that it must never
  produce thesis performance results.

## 6. Procedure

1. Open a T4 session; install pinned dependencies.
2. Run the generator timing check (`docs/generator_contract.md` §6).
3. Choose the labelled-set size from that measurement.
4. Generate labels over a MedQA subsample with the paper's decision tree.
   Write with `write_training_file()`, which refuses to overwrite — labels
   cost GPU hours and a silent rerun would destroy a checkpoint's provenance.
5. Check `label_distribution()` before training.
6. Train Flan-T5-large with the config above; record epochs actually run.
7. Record validation accuracy in a `CheckpointRecord`.
8. Download the checkpoint. **Inference runs locally** (≈1.6 GB fp16 fits the
   4 GB card); only the one-off training needs the remote GPU.

## 7. Contamination control

The filter is trained on **general-medical** MedQA, never on the thesis's
Alzheimer's evaluation questions. This is the paper's own setup and it also
removes any suspicion that the baseline was tuned on the evaluation set. The
thesis's 123-candidate question pool plays no part in filter training.

## 8. What the thesis must state

The checkpoint was unavailable · the filter was retrained by the student · the
base size used · the epoch count and validation accuracy actually reached ·
the training-set size and that it is a subsample · that training ran on
different hardware from the rest of the pipeline · **that this makes the
baseline an adaptation of RAG², not a reproduction.**

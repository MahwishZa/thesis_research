# Evaluation and Experimental Design

Answers what is measured and how a difference is judged real. For what is
being compared, see `methodology.md`. For running an evaluation, see
`reproducibility.md`.

## 1. Metrics

`src/evaluation/rag_metrics.py`, applied identically to every arm.

| Metric | Definition | Role |
|---|---|---|
| Currency (mean `T(s)` of admitted evidence, on `temporal_candidate` questions) | See `glossary.md` | **Primary**, main comparison |
| Token F1 | SQuAD-style unigram precision/recall/F1 | Secondary / diagnostic |
| Exact match | Normalised string equality | Secondary |
| ROUGE-L | F1 over the longest common in-order subsequence | Secondary |
| Context precision / recall / F1 | Admitted evidence vs. known gold-relevant evidence | Secondary; `None` (not 0.0) when a question carries no gold-evidence annotation, so an unmeasurable case cannot be misread as a measured zero |
| Groundedness | Fraction of an answer's content tokens present in its admitted evidence | Secondary automatic faithfulness proxy, not an entailment judgement |

Currency is primary because it is the direct measure of what the proposed
mechanism is supposed to change; the others are retained as standard RAG
diagnostics but are not what decides the main verdict.

## 2. Why currency, not accuracy, decides the main verdict

The mechanism under test changes *which evidence is admitted*, not how an
answer is phrased. A metric that only compares generated text to a
reference answer can be insensitive to that change entirely (e.g. an
extractive stand-in generator returns the top-admitted passage's own text,
so text-overlap metrics can end up measuring evidence selection anyway, but
indirectly and noisily). Currency measures the selection directly.

## 3. Temporal-candidate subgroup

Every scored row is split by whether its question is flagged
`temporal_candidate` (see `glossary.md`) and aggregated separately
for each system. If the Temporal Filter's mechanism works at all, its
effect should concentrate on this subgroup and be closer to null on the
rest — a specific, falsifiable pattern, not just an aggregate number.

## 4. Statistical procedure

`src/evaluation/stats.py`, standard library only (no SciPy dependency).

- **Paired sign test** — for each question both systems answered, does one
  score higher, lower, or the same? Decides the main verdict: a consistent
  per-question direction, not an average that a couple of outliers could
  produce.
- **Exact McNemar's test** — on paired discordant binary outcomes, two-sided
  exact binomial, no normal approximation.
- **Paired bootstrap 95% CI** — on the absolute difference, resampling
  *questions* as the unit so each question's paired outcome stays together.
- **Holm correction** — across the pre-declared comparisons, so multiple
  tests don't inflate the false-positive rate.

A result is only read as a real difference if the significance test says
so at the standard α = 0.05 threshold — an average gap, however large it
looks, is not by itself sufficient.

## 5. Ablation study

Full proposed system vs. the same system with `λ = 0` (temporal term
switched off), scored and tested identically to the main comparison. This
isolates whether the *temporal signal specifically* is responsible for any
observed difference, as opposed to filtering in general (which the
No-Filter control already addresses separately).

## 6. Abstention handling

The proposed system can, in principle, admit nothing if no passage clears
`θ`. The default policy is to generate from whatever was admitted, including
nothing, exactly as the baseline does — an answer produced from empty
evidence is recorded as ungrounded, not silently excluded, so a high
threshold cannot manufacture a good-looking result by answering fewer
questions.

## 7. Evaluation status

Implemented and exercised end to end (retrieval → admission → generation →
scoring → ablation → significance testing), including one full pilot-scale
run. **The reported comparison against Objective 2 is pending completion of
the full-scale setup** (see `reproducibility.md` for what "full-scale" means
here and what is currently reduced). Pilot-scale output is committed under
`results/` for reproducibility; it documents that the pipeline and
statistical procedure work correctly on real data, not the thesis's
reported finding.

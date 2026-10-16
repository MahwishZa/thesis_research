# Experiment 01: who writes the rationales

## What this is

Part 1 of RAG² (Sohn et al., NAACL 2025, section 3.3): a language model reads a medical question and writes a
step-by-step explanation, the **rationale**. The rationale alone, without the question, is the text that the
MedCPT query encoder turns into the search query for Part 2. The code is in `src/rationale_query/`.

The paper uses Llama-3-8B-Instruct for this. This experiment does two things:

1. **Reproduces the paper's writer** with the model file the project already has (Llama-3-8B-Instruct, 4-bit).
2. **Compares it with a second writer** (Qwen2.5-7B-Instruct, 4-bit), so that later chapters can say whether the
   choice of writer matters. Which writers take part is listed in `settings.json`.

Nothing here needs labelling or any manual checking: every number is computed from the rationales themselves.

## What is the same as in the paper, and what is not

| Item | Here | Paper or authors' code |
|---|---|---|
| Prompt | the text of the box on p.4, then the question | same; where the line break falls is not recoverable from the PDF (open item) |
| Decoding | greedy (temperature 0, top-k 1), one answer per question | same (A.3, p.13) |
| What is searched with | the rationale alone | same |
| Encoder | `ncbi/MedCPT-Query-Encoder`, first-token vector, 512 tokens, 32-bit, not normalised | same as the authors' `query_encode.py` |
| Model weights | 4-bit GGUF file run with llama.cpp | the paper serves the model with vLLM and does not mention quantising it: **a declared deviation** |
| System prompt | none | not stated |
| Longest rationale | 512 new tokens (the encoder cuts at 512 tokens anyway) | not stated |

The 4-bit file means the words of a rationale can differ from what the full-precision model would write. The effect
is not measured here; the second writer and the optional 8-bit file (below) are the way to look at it.

## What is compared

For every writer the report (`python -m rationale_query report ...`) gives, with no labels needed:

- how many rationales were written and how many failed,
- how long the rationales are (new tokens: median and 95th percentile) and how many were **cut off at the limit**,
- how many are **longer than the encoder's 512 tokens** and are cut again when they are encoded,
- how many name a final option ("The answer is B") and how often two writers name the **same** option,
- the time per question.

The final option is read with a simple pattern (`final_option` in `src/rationale_query/diagnostics.py`), so it is a
description of the rationales and not a measure of correctness. The report says nothing about retrieval or answer
quality; that comes after Part 2 exists.

## Before the first run

You need, from the repository folder in PowerShell:

- Python with the packages of the project installed: `pip install -e .` and, for the model and the encoder,
  `pip install llama-cpp-python numpy torch transformers`. The versions that were checked together are
  Python 3.12, llama-cpp-python 0.3.35, torch 2.4.1, transformers 4.51.3 and numpy 1.26.4.
- the model files in the folder `models\` of the repository (it is not committed): the names are in `settings.json`.
  A file whose SHA-256 differs from the one in `settings.json` is refused, so a damaged download cannot be used by
  mistake.
- a file of questions: one JSON object per line with an `id` and the `question` text with its options inline.
  `example_questions.jsonl` has five invented questions for a first trial. No evaluation dataset has been chosen yet;
  when it is, its questions go into a file of the same form.

## How to run it

A quick trial on two questions (a few minutes: loading a model takes about a minute, a rationale takes under a
minute):

```powershell
python experiments\01_rationale_writers\run_study.py --questions experiments\01_rationale_writers\example_questions.jsonl --run-name trial --limit 2
```

The full run on a questions file, with the query vectors as well:

```powershell
python experiments\01_rationale_writers\run_study.py --questions path\to\questions.jsonl --run-name my_run --encode
```

- Every writer whose model file is in `models\` runs, one after the other, and each takes about half a day for a thousand
  questions; the others are reported as passed over. To start with the paper's writer alone, add
  `--writers llama3-8b-instruct-q4km`; the second writer can be added later with the same `--run-name`.
  `--models-dir D:\models` points to another folder.
- **It can be stopped and started again.** Press Ctrl+C, or close the laptop, and later run **the same command**: a
  question that has its rationale is not written again, and a writer that is complete is passed over at once. If a
  question failed, the run says so and the same command tries it again.
- The speed on this laptop is about 9 new tokens per second on the processor, so a rationale of 300 tokens takes
  about 45 seconds. A thousand questions take roughly half a day per writer. The run prints the time left.
- Per writer, the folder `outputs\<run name>\<writer name>\` holds `rationales.jsonl` (one line per question),
  `config.json` (every setting that shapes the rationales, with the SHA-256 of the model file) and, with `--encode`,
  `query_vectors.npy` and `query_vectors.json`. These are not committed.
- The comparison is printed at the end and saved in `results\<run name>\report.txt` and `report.json`; `run.json` next to
  them names the questions file (with its SHA-256) and the writers. These small files are the ones to commit when a run
  is worth keeping.

A trial leaves a small folder `results\trial`; delete it, or leave it uncommitted. If a run stops with "the chat
template stored in the model file renders a different prompt than the Llama-3 format", that model file does not match
the format the paper's model uses; look at the message before adding `--allow-template-mismatch` (the setting is recorded
in `config.json`).

A folder is tied to its settings. If the model file, the prompt, the length limit or the library version changes, the
run refuses to add to that folder and asks for a new one, so two kinds of rationale are never mixed. A question whose
text has changed is refused in the same way.

To look at the exact text the model receives for a question, with no model needed:

```powershell
python -m rationale_query show-prompt --question "Which drug is a cholinesterase inhibitor? A) Donepezil B) Aspirin C) Insulin D) Warfarin"
```

To write the queries for the authors' retriever (a JSON list in the order of the questions file):

```powershell
python -m rationale_query export-queries --rationales outputs\my_run\llama3-8b-instruct-q4km --questions path\to\questions.jsonl --out queries.json
```

## Checking the real model files once

The automatic tests use tiny stand-in models, so they cannot show that the real files behave. These checks run the
same code on the real files; run them once on the laptop and keep the output:

```powershell
$env:RATIONALE_LLAMA3_GGUF = "models\Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
$env:RATIONALE_MEDCPT = "1"
python -m unittest tests.integration.test_rationale_models -v
```

The first run downloads the encoder (about 440 MB), and the model file is hashed and loaded more than once, so allow
about ten minutes. Every test should say `ok`; none should be skipped.

## Adding a writer

Add an entry to `writers` in `settings.json`: a `name` (letters, digits, dots, dashes), the `model_file`, the
`chat_format` and, if the hash is known, its `sha256`. Use `llama-3` only for Llama-3-Instruct files; `gguf` uses the chat
template stored in the file, which for Qwen adds the model's own default system message. A larger file of the same model,
for example an 8-bit Llama-3 file, answers the question how much the 4-bit weights change the rationales; it needs about
10 GB of memory.

## Open items

- The paper does not say whether one or two line breaks stand before "Here is the question:". The code sends one.
- The paper does not give a limit for the rationale length or a system prompt. The values above are choices, recorded
  in `config.json` of every folder.
- The authors' retriever gives the cross-encoder the rationale where the paper says the question; that belongs to
  Part 2 and is handled there (`rerank_query` in `src/rationale_query/query.py` offers both).

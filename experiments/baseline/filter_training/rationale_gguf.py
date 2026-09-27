"""GGUF-quantized Llama-3-8B-Instruct as the ``RationaleScorer``, for label
generation on hardware that cannot run ``rationale.py``'s ``bitsandbytes``
path (Llama3RationaleScorer needs ~5.5-6 GB VRAM even 4-bit, or ~16 GB RAM
full-precision on CPU - both infeasible on a 4 GB VRAM / 15.2 GB RAM laptop,
verified 2026-09-27).

**What is identical to ``Llama3RationaleScorer``:** the model identity
(Llama-3-8B-Instruct, same weights before quantization), the prompt
template (``rationale.COT_PROMPT_TEMPLATE``, imported not duplicated), the
chat formatting (this class loads the real HF tokenizer - weights only,
never the model - specifically so ``apply_chat_template`` produces the
byte-identical prompt string the HF path would have used), greedy decoding,
and the answer-extraction logic (``medqa_data.extract_answer_letter``,
imported not duplicated). ``labeling.py``'s decision tree - the actual
label function - never sees which scorer produced a ``RationaleOutcome``
and is completely unchanged.

**What differs, and why it's disclosed as a deviation, not hidden:** the
model runs as a 4-bit GGUF quantization (llama.cpp's k-quant scheme) via
``llama-cpp-python``, instead of ``bitsandbytes`` nf4 on the original
safetensors weights. Both are 4-bit quantizations of the same model; this
is a change of execution backend for an already-hardware-adapted step
(the project's own choice of nf4 over full precision was already a
disclosed adaptation - see ``rationale.py``'s docstring), not a change of
which model or which method produces the labels. Record this scorer's
``name`` in whatever calls it, so a checkpoint trained on GGUF-generated
labels is traceable, the same way ``CheckpointRecord.notes`` already
records every other deviation.

**A genuine simplification, not a shortcut:** ``Llama3RationaleScorer``
needs a second, teacher-forced forward pass to get perplexity, because
``model.generate()`` does not expose per-token logits. llama.cpp's
completion API returns each generated token's log-probability as part of
the same generation call - under greedy decoding, that log-probability
*is* the teacher-forced probability of the token actually produced, so no
second pass is needed here. Confirmed against the same formula
(mean negative log-likelihood, exponentiated) either way.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence

from .labeling import RationaleOutcome
from .medqa_data import OPTION_LETTERS, extract_answer_letter
from .rationale import COT_PROMPT_TEMPLATE, RationaleGenerationError


class GGUFRationaleScorer:
    """See module docstring. One instance per label-generation run."""

    #: Recorded into every run's provenance - see module docstring.
    name = "llama-3-8b-instruct-gguf-q4"

    def __init__(
        self,
        model_path: str | Path,
        *,
        tokenizer_model_id: str,
        tokenizer_revision: str,
        n_ctx: int = 4096,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = 0,
        max_new_tokens: int = 256,
        seed: int = 42,
    ) -> None:
        """
        Args:
            model_path: local path to the ``.gguf`` file. Not downloaded
                here - see docs/reproducibility.md for how to obtain and
                verify one.
            tokenizer_model_id, tokenizer_revision: the HF tokenizer to
                load (weights are never fetched) so chat-template
                formatting matches ``Llama3RationaleScorer`` exactly.
                ``tokenizer_revision`` must be a pinned commit sha, same
                requirement as the HF path.
            n_ctx: context window. 4096 comfortably covers a MedQA
                question + 4 options + one textbook passage + a
                ``max_new_tokens``-length rationale.
            n_threads: CPU threads for llama.cpp. ``None`` lets
                llama.cpp auto-detect; set explicitly if you want to
                leave headroom for other processes.
            n_gpu_layers: layers offloaded to GPU. 0 (default) runs
                entirely on CPU/RAM - correct and sufficient on hardware
                with too little VRAM to help; raise it only if you have
                VRAM to spare, as a speed optimisation, never a
                requirement.
        """
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RationaleGenerationError(
                "GGUFRationaleScorer requires the llama-cpp-python package "
                "(`pip install llama-cpp-python`). See "
                "docs/reproducibility.md for the full local setup."
            ) from exc
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RationaleGenerationError(
                "GGUFRationaleScorer needs transformers for the tokenizer/"
                "chat-template only - no model weights are loaded through "
                "it, but the package must be installed."
            ) from exc

        model_path = Path(model_path)
        if not model_path.exists():
            raise RationaleGenerationError(
                f"GGUF model file not found: {model_path}. See "
                "docs/reproducibility.md for how to obtain and verify one."
            )
        if not tokenizer_revision or tokenizer_revision == "main":
            raise RationaleGenerationError(
                f"tokenizer_revision must be a pinned commit sha, not "
                f"{tokenizer_revision!r} - matches Llama3RationaleScorer's "
                "identical requirement, and for the identical reason."
            )

        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_model_id, revision=tokenizer_revision
        )
        self._llm = Llama(
            model_path=str(model_path),
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            seed=seed,
            logits_all=False,
            verbose=False,
        )

    def _build_prompt(
        self, question: str, choices: Sequence[str], evidence: Optional[str],
    ) -> tuple[str, dict[str, str]]:
        if len(choices) != len(OPTION_LETTERS):
            raise RationaleGenerationError(
                f"expected {len(OPTION_LETTERS)} choices, got {len(choices)}"
            )
        letter_to_text = dict(zip(OPTION_LETTERS, choices))
        options_block = "\n".join(f"{l}) {t}" for l, t in letter_to_text.items())
        context_block = f"Context: {evidence}\n\n" if evidence else ""
        prompt = COT_PROMPT_TEMPLATE.format(
            context_block=context_block,
            question=question,
            options_block=options_block,
            letters="/".join(OPTION_LETTERS),
        )
        return prompt, letter_to_text

    def score(
        self,
        question: str,
        choices: Sequence[str],
        answer: str,
        evidence: Optional[str],
    ) -> RationaleOutcome:
        """One generation, scored from its own reported token log-probs.

        Greedy decoding (``temperature=0.0``), matching this repository's
        standing no-sampling rule - see ``Llama3RationaleScorer.score``'s
        identical reasoning.
        """
        prompt, letter_to_text = self._build_prompt(question, choices, evidence)

        chat_prompt = prompt
        if getattr(self._tokenizer, "chat_template", None):
            chat_prompt = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )

        completion = self._llm.create_completion(
            chat_prompt,
            max_tokens=self.max_new_tokens,
            temperature=0.0,
            logprobs=1,
        )
        choice = completion["choices"][0]
        rationale_text = choice["text"]
        if not rationale_text.strip():
            raise RationaleGenerationError(
                "model generated an empty rationale; cannot score its "
                "perplexity (RationaleOutcome requires perplexity > 0)"
            )

        token_logprobs = (choice.get("logprobs") or {}).get("token_logprobs") or []
        # llama.cpp can report None for a token in some builds/edge cases;
        # drop rather than let one poison the mean silently.
        usable = [lp for lp in token_logprobs if lp is not None]
        if not usable:
            raise RationaleGenerationError(
                "no usable token log-probabilities were returned for the "
                "generated rationale; cannot compute perplexity"
            )
        mean_nll = -sum(usable) / len(usable)
        # Same clamp as Llama3RationaleScorer, and for the same reason: a
        # degenerate near-zero loss should not crash one pathological
        # example's labelling.
        perplexity = max(math.exp(mean_nll), 1e-6)

        predicted_letter = extract_answer_letter(rationale_text)
        predicted_text = (
            letter_to_text.get(predicted_letter) if predicted_letter else None
        )
        correct = predicted_text is not None and predicted_text.strip() == answer.strip()

        return RationaleOutcome(correct=correct, perplexity=perplexity)

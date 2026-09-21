"""Llama-3-8B-Instruct as the ``RationaleScorer`` for filter-label generation.

Implements ``labeling.RationaleScorer``: given a question, its options, the
correct option's text, and optional evidence, generate a chain-of-thought
rationale, extract the predicted option, and report whether it was correct
plus the rationale's perplexity (specification SS10.2, ledger E1-E3).

Llama-3-8B-Instruct specifically (not a smaller/different model) because the
official repository's own training-data path -
``classifier/data/medqa/llama3_cot/5%-train.json`` - names it: the paper's
own label-generation used Llama-3 with chain-of-thought prompting. Using the
same generator the main experiment already uses (specification SS9) is also
the model this thesis has an accepted licence for.

Uses a directly-loaded (tokenizer, model) pair rather than
``HuggingFaceGenerator``: computing perplexity needs a teacher-forced
forward pass over the generated rationale and raw logits, which
``HuggingFaceGenerator.generate()`` does not expose. This class is used only
for offline label generation, never inside the thesis's own three-arm
comparison, so it does not need to share an instance with anything (unlike
``HuggingFaceGenerator``, which `RunConfig` requires exactly one shared copy
of - see its own docstring).
"""

from __future__ import annotations

from typing import Optional, Sequence

from .labeling import RationaleOutcome
from .medqa_data import OPTION_LETTERS

COT_PROMPT_TEMPLATE = (
    "You are answering a multiple-choice medical exam question.\n\n"
    "{context_block}"
    "Question: {question}\n\n"
    "Options:\n{options_block}\n\n"
    "Think step by step, then end your answer with a line in exactly this "
    'form: "Answer: <letter>", where <letter> is one of {letters}.\n\n'
    "Reasoning:"
)


class RationaleGenerationError(RuntimeError):
    """Raised when a rationale cannot be generated or scored."""


class Llama3RationaleScorer:
    """See module docstring. One instance per label-generation run."""

    def __init__(
        self,
        model_id: str,
        revision: str,
        *,
        device: Optional[str] = None,
        max_new_tokens: int = 256,
        quantization: Optional[str] = "nf4",
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RationaleGenerationError(
                "Llama3RationaleScorer requires torch and transformers."
            ) from exc

        if not revision or revision == "main":
            raise RationaleGenerationError(
                f"revision must be a pinned commit sha, not {revision!r} - "
                "see ModelSpec's identical requirement and why."
            )

        self._torch = torch
        self.model_id = model_id
        self.revision = revision
        self.max_new_tokens = max_new_tokens

        self._tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        kwargs: dict = {"revision": revision, "device_map": device or "auto"}
        if quantization:
            from transformers import BitsAndBytesConfig
            if quantization == "nf4":
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            elif quantization == "int8":
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            else:
                raise RationaleGenerationError(
                    f"unsupported quantization: {quantization!r}"
                )
        else:
            kwargs["torch_dtype"] = torch.bfloat16

        self._model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self._model.eval()

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
        """One generation + one teacher-forced pass over its own output.

        Greedy decoding throughout (``do_sample=False``), matching this
        repository's standing rule (specification SS9.1) that nothing in
        this thesis samples: two label-generation runs over the same
        question/evidence pair must produce the same label, or the training
        file could not be reproduced.
        """
        torch = self._torch
        prompt, letter_to_text = self._build_prompt(question, choices, evidence)

        chat_prompt = prompt
        if getattr(self._tokenizer, "chat_template", None):
            chat_prompt = self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )

        encoded = self._tokenizer(chat_prompt, return_tensors="pt")
        encoded = {k: v.to(self._model.device) for k, v in encoded.items()}
        prompt_len = encoded["input_ids"].shape[1]

        with torch.inference_mode():
            generated = self._model.generate(
                **encoded, max_new_tokens=self.max_new_tokens, do_sample=False,
            )
        rationale_ids = generated[0][prompt_len:]
        if rationale_ids.numel() == 0:
            raise RationaleGenerationError(
                "model generated an empty rationale; cannot score its "
                "perplexity (RationaleOutcome requires perplexity > 0)"
            )
        rationale_text = self._tokenizer.decode(
            rationale_ids, skip_special_tokens=True
        )

        from .medqa_data import extract_answer_letter
        predicted_letter = extract_answer_letter(rationale_text)
        predicted_text = (
            letter_to_text.get(predicted_letter) if predicted_letter else None
        )
        correct = predicted_text is not None and predicted_text.strip() == answer.strip()

        # Perplexity of the generated rationale, teacher-forced over the
        # SAME sequence just generated, conditioned on the prompt -
        # specification SS10.2/E3: over the rationale, never the query.
        with torch.inference_mode():
            outputs = self._model(generated, labels=generated)
        # Position i's logits predict token i+1; the rationale starts at
        # position prompt_len, so its predicting logits start one earlier.
        logits = outputs.logits[0, prompt_len - 1: -1, :]
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        token_log_probs = log_probs.gather(
            1, rationale_ids.unsqueeze(-1)
        ).squeeze(-1)
        mean_nll = -token_log_probs.mean().item()
        # Clamped away from exactly 0: RationaleOutcome refuses a
        # non-positive perplexity, and a degenerate near-zero loss (an
        # exact repeat of highly predictable tokens) should not crash
        # label generation over one pathological example.
        perplexity = max(float(torch.exp(torch.tensor(mean_nll)).item()), 1e-6)

        return RationaleOutcome(correct=correct, perplexity=perplexity)

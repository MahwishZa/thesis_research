"""A tiny made-up Mistral-style model, saved as a model folder, for tests that run the real Transformers library.

The weights are random, so such a model writes nonsense; it is used to check how the code drives the library (the tokens it
is given, the precision, when it stops), never to judge what a rationale says.
"""

from __future__ import annotations

import re
from pathlib import Path

WORD = re.compile(r"\w+|[^\w\s]")


def split_words(text: str) -> list[str]:
    """How the tiny tokenizer cuts a text: words and single punctuation marks."""
    return WORD.findall(text)


def write_tiny_mistral(folder: str | Path, texts: list[str], *, eos_push: float) -> Path:
    """Write a 2-layer Mistral model (hidden size 32) and a word-level tokenizer that knows the words of ``texts``.

    ``eos_push`` sets how strongly the model wants to write the end-of-answer token: a large positive number makes it stop
    at once, a large negative number makes it never stop. (All embeddings are positive, so the sign decides.)
    """
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import MistralConfig, MistralForCausalLM, PreTrainedTokenizerFast

    words = sorted({w for text in texts for w in split_words(text)})
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, **{w: i + 3 for i, w in enumerate(words)}}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", 1)])
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, bos_token="<s>", eos_token="</s>", unk_token="<unk>", pad_token="<unk>")

    torch.manual_seed(0)
    config = MistralConfig(
        vocab_size=len(vocab), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, max_position_embeddings=2048, sliding_window=None, bos_token_id=1, eos_token_id=2,
        tie_word_embeddings=False,
    )
    model = MistralForCausalLM(config)
    with torch.no_grad():
        model.model.embed_tokens.weight.copy_(model.model.embed_tokens.weight.abs() + 0.5)
        model.lm_head.weight[2] = eos_push
    folder = Path(folder)
    model.save_pretrained(folder)
    fast.save_pretrained(folder)
    return folder

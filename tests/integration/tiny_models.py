"""Tiny synthetic models, so that the real libraries can be tested without downloading anything.

``write_tiny_gguf`` needs the ``gguf`` and ``numpy`` packages and writes a Llama-3-shaped GGUF file of 2 MB. Its
hidden state is the one-hot embedding of the last token and its output matrix maps each token to its successor, so
the greedy continuation of any prompt is the fixed chain ``ANSWER_TEXT`` followed by the end-of-turn token (or, with
``loop=True``, the same words again and again). That exercises the real llama.cpp: how the special markers are
tokenised, the start-of-text token, the stop and length endings, the cache, and the chat template of the file.

``write_tiny_bert`` needs ``torch`` and ``transformers`` and writes a BERT with random weights, two layers and a
vocabulary of a few dozen words into a folder that ``from_pretrained`` reads, to exercise the real encoder code.
``write_tiny_cross_encoder`` writes the same kind of BERT with a one-number classification head, for the cross-encoder.
"""

from __future__ import annotations

import itertools
from pathlib import Path

from tests.support import LLAMA3_JINJA

ANSWER_TEXT = " The answer is B."
ANSWER_TOKENS = 5  # " The", " answer", " is", " B", "."

_SPECIALS = ["<|begin_of_text|>", "<|end_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]
_WORDS = ["ĠThe", "Ġanswer", "Ġis", "ĠB"]
_MERGED = ["Ġt", "he", "Ġthe", "ĊĊ"]
_MERGES = ["Ġ t", "h e", "Ġt he", "Ċ Ċ"]
_CHAIN = ["ĊĊ", "ĠThe", "Ġanswer", "Ġis", "ĠB", "."]


def _byte_characters() -> dict[int, str]:
    """GPT-2's mapping of the 256 byte values to printable characters, as byte-level BPE vocabularies write them."""
    keep = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    chars = keep[:]
    extra = 0
    for byte in range(256):
        if byte not in keep:
            keep.append(byte)
            chars.append(256 + extra)
            extra += 1
    return dict(zip(keep, map(chr, chars), strict=True))


def write_tiny_gguf(
    path: str | Path, *, loop: bool = False, eos: str = "<|eot_id|>", template: str | None = LLAMA3_JINJA
) -> None:
    import numpy as np
    from gguf import GGUFWriter

    table = _byte_characters()
    tokens = [table[b] for b in range(256)] + _MERGED + _WORDS + _SPECIALS
    types = [1] * (256 + len(_MERGED) + len(_WORDS)) + [3] * len(_SPECIALS)  # 1 normal, 3 control
    index = {token: number for number, token in enumerate(tokens)}
    n_vocab, n_embd, n_head, n_ff = len(tokens), 272, 4, 64
    end = index["<|eot_id|>"]

    successor = dict.fromkeys(range(n_vocab), end)
    for before, after in itertools.pairwise(_CHAIN):
        successor[index[before]] = index[after]
    successor[index["."]] = index["ĠThe"] if loop else end

    writer = GGUFWriter(str(path), "llama")
    writer.add_name("tiny-llama3-chain")
    writer.add_context_length(2048)
    writer.add_embedding_length(n_embd)
    writer.add_block_count(1)
    writer.add_feed_forward_length(n_ff)
    writer.add_head_count(n_head)
    writer.add_head_count_kv(n_head)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_rope_dimension_count(n_embd // n_head)
    writer.add_vocab_size(n_vocab)
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("llama-bpe")
    writer.add_token_list(tokens)
    writer.add_token_types(types)
    writer.add_token_merges(_MERGES)
    writer.add_bos_token_id(index["<|begin_of_text|>"])
    writer.add_eos_token_id(index[eos])
    writer.add_add_bos_token(True)
    if template:
        writer.add_chat_template(template)

    embedding = np.zeros((n_vocab, n_embd), dtype=np.float32)
    output = np.zeros((n_vocab, n_embd), dtype=np.float32)
    for token in range(n_vocab):
        embedding[token, token] = 1.0
        output[successor[token], token] = 1.0
    rng = np.random.default_rng(0)

    def small(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) * 0.02).astype(np.float32)

    ones = np.ones(n_embd, dtype=np.float32)
    writer.add_tensor("token_embd.weight", embedding)
    writer.add_tensor("output_norm.weight", ones)
    writer.add_tensor("output.weight", output)
    writer.add_tensor("blk.0.attn_norm.weight", ones)
    writer.add_tensor("blk.0.attn_q.weight", small(n_embd, n_embd))
    writer.add_tensor("blk.0.attn_k.weight", small(n_embd, n_embd))
    writer.add_tensor("blk.0.attn_v.weight", small(n_embd, n_embd))
    writer.add_tensor("blk.0.attn_output.weight", np.zeros((n_embd, n_embd), dtype=np.float32))  # no contribution
    writer.add_tensor("blk.0.ffn_norm.weight", ones)
    writer.add_tensor("blk.0.ffn_gate.weight", small(n_ff, n_embd))
    writer.add_tensor("blk.0.ffn_up.weight", small(n_ff, n_embd))
    writer.add_tensor("blk.0.ffn_down.weight", np.zeros((n_embd, n_ff), dtype=np.float32))  # no contribution
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _tiny_bert_config(folder: Path, **extra):
    from transformers import BertConfig, BertTokenizerFast

    folder.mkdir(parents=True, exist_ok=True)
    words = "the patient has memory loss and alzheimer disease is likely donepezil a b c d answer option ##s ##ing . , ? ) ( -"
    vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *words.split()]
    (folder / "vocab.txt").write_text("\n".join(vocabulary) + "\n", encoding="utf-8")
    BertTokenizerFast(vocab_file=str(folder / "vocab.txt"), do_lower_case=True).save_pretrained(folder)
    return BertConfig(
        vocab_size=len(vocabulary),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=512,
        **extra,
    )


def write_tiny_bert(folder: str | Path) -> None:
    import torch
    from transformers import BertModel

    folder = Path(folder)
    torch.manual_seed(0)
    BertModel(_tiny_bert_config(folder)).save_pretrained(folder)


def write_tiny_cross_encoder(folder: str | Path) -> None:
    import torch
    from transformers import BertForSequenceClassification

    folder = Path(folder)
    torch.manual_seed(1)
    BertForSequenceClassification(_tiny_bert_config(folder, num_labels=1)).save_pretrained(folder)


#: The words of the tiny filter tokenizer. Anything else becomes ``<unk>``. They cover the filter input of the tests.
TINY_T5_WORDS = (
    "Given the following evidence , determine whether it helps answer provided question . Evidence : Question : "
    "yes no good bad A ) B C D which drug is a donepezil aspirin insulin warfarin cholinesterase inhibitor dementia "
    "snippet one two three four five six seven eight nine ten the of and to in"
).split()


def write_tiny_t5(folder: str | Path, *, extra_rows: int = 8) -> None:
    """A Flan-T5-shaped model with random weights (two layers, 32 numbers wide) and a word-level tokenizer, into ``folder``.

    The tokenizer is a fast tokenizer that puts ``</s>`` at the end of every text, like T5's. The embedding has ``extra_rows``
    rows more than the tokenizer has tokens, as T5's real embedding has 28 more rows than its tokenizer, so that adding the two
    label tokens and resizing the embedding to the tokenizer's length behaves as it does for the real model.
    Needs ``torch``, ``transformers`` and ``tokenizers``.
    """
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast, T5Config, T5ForConditionalGeneration

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    words = ["<pad>", "</s>", "<unk>", *dict.fromkeys(TINY_T5_WORDS)]
    backend = Tokenizer(models.WordLevel({word: number for number, word in enumerate(words)}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(single="$A </s>", special_tokens=[("</s>", 1)])
    PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", eos_token="</s>", unk_token="<unk>", model_max_length=512
    ).save_pretrained(folder)
    torch.manual_seed(0)
    config = T5Config(
        vocab_size=len(words) + extra_rows, d_model=32, d_kv=8, d_ff=64, num_layers=2, num_decoder_layers=2, num_heads=4,
        decoder_start_token_id=0, pad_token_id=0, eos_token_id=1, tie_word_embeddings=False,
    )
    T5ForConditionalGeneration(config).save_pretrained(folder)

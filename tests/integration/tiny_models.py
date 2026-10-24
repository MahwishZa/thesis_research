"""Tiny made-up language models, written as GGUF files, for tests that run the real llama.cpp.

The weights are random, so such a model writes nonsense; it is used to check how the code drives the library (the tokens
it is given, when it stops), never to judge what a rationale says.
"""

from __future__ import annotations

from pathlib import Path

WORDS = ["USER", ":", "ASSISTANT", "The", "following", "are", "multiple", "choice", "questions", "about", "medical", "knowledge", "Here", "is", "the", "question"]


def write_tiny_llama(path: str | Path, *, eos_push: float, context_length: int = 256) -> Path:
    """Write a 2-layer Llama-style model with a 32-number hidden size and a small made-up vocabulary.

    ``eos_push`` sets how strongly the model wants to write the end-of-answer token: a large positive number makes it stop at
    once, a large negative number makes it never stop. (All embeddings are positive, so the sign decides.)
    """
    import gguf
    import numpy as np

    rng = np.random.default_rng(0)
    tokens: list[bytes] = [b"<unk>", b"<s>", b"</s>"] + [f"<0x{i:02X}>".encode() for i in range(256)]
    types = [gguf.TokenType.UNKNOWN, gguf.TokenType.CONTROL, gguf.TokenType.CONTROL] + [gguf.TokenType.BYTE] * 256
    scores = [0.0, 0.0, 0.0] + [0.0] * 256
    letters = sorted({c for w in WORDS for c in w} | {"▁"})
    for letter in letters:
        tokens.append(letter.encode())
        types.append(gguf.TokenType.NORMAL)
        scores.append(-5.0)
    for word in WORDS:
        tokens.append(("▁" + word).encode())
        types.append(gguf.TokenType.NORMAL)
        scores.append(-1.0)
    n_vocab, n_embd, n_ff, n_layer, n_head = len(tokens), 32, 64, 2, 4

    writer = gguf.GGUFWriter(str(path), "llama")
    writer.add_name("tiny-llama-for-tests")
    writer.add_context_length(context_length)
    writer.add_embedding_length(n_embd)
    writer.add_block_count(n_layer)
    writer.add_feed_forward_length(n_ff)
    writer.add_head_count(n_head)
    writer.add_head_count_kv(n_head)
    writer.add_rope_dimension_count(n_embd // n_head)
    writer.add_layer_norm_rms_eps(1e-5)
    writer.add_tokenizer_model("llama")
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(types)
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)
    writer.add_unk_token_id(0)
    writer.add_add_bos_token(True)
    writer.add_add_eos_token(False)

    def weight(rows: int, cols: int) -> np.ndarray:
        return (rng.standard_normal((rows, cols)) * 0.02).astype(np.float32)

    writer.add_tensor("token_embd.weight", (np.abs(rng.standard_normal((n_vocab, n_embd))) + 0.5).astype(np.float32))
    writer.add_tensor("output_norm.weight", np.ones(n_embd, dtype=np.float32))
    output = weight(n_vocab, n_embd)
    output[2] = eos_push
    writer.add_tensor("output.weight", output)
    for i in range(n_layer):
        writer.add_tensor(f"blk.{i}.attn_norm.weight", np.ones(n_embd, dtype=np.float32))
        for name in ("attn_q", "attn_k", "attn_v", "attn_output"):
            writer.add_tensor(f"blk.{i}.{name}.weight", weight(n_embd, n_embd))
        writer.add_tensor(f"blk.{i}.ffn_norm.weight", np.ones(n_embd, dtype=np.float32))
        writer.add_tensor(f"blk.{i}.ffn_gate.weight", weight(n_ff, n_embd))
        writer.add_tensor(f"blk.{i}.ffn_up.weight", weight(n_ff, n_embd))
        writer.add_tensor(f"blk.{i}.ffn_down.weight", weight(n_embd, n_ff))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return Path(path)

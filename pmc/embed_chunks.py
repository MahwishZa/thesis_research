#!/usr/bin/env python3
"""Deterministic embedding + index construction over the frozen chunk layer.

Reads pmc/chunks/chunks.jsonl (frozen, unmodified) and writes an additive index
layer under pmc/index/:

    embeddings.f32      row-major float32 vectors, one row per chunk
    index_manifest.jsonl  row -> chunk_id + the metadata retrieval needs
    index_meta.json     model id, dim, counts, digests, production flag

Model choice is NOT a convenience decision. The base paper uses MedCPT for dense
retrieval and MedCPT cross-encoding for reranking, and the thesis proposal
(5.3) pins both as "baseline fidelity" because the retriever is deliberately
frozen -- it is the central internal-validity guarantee. So the production
encoder here is MedCPT and nothing else:

    documents : ncbi/MedCPT-Article-Encoder
    queries   : ncbi/MedCPT-Query-Encoder      (used by retrieve.py)
    reranking : ncbi/MedCPT-Cross-Encoder      (used by retrieve.py)

A deterministic stub encoder exists for offline testing only. It refuses to
produce a production index unless --allow-stub is passed, and it stamps
index_meta.json with production=false so a stub index can never be mistaken for
a real one downstream.

Determinism: chunks are processed in sorted chunk_id order, vectors are float32,
and the manifest records a digest over (chunk_id, vector bytes). Re-running on
the same chunks with the same model reproduces the same files.

Standard library only, except the optional production backend (torch +
transformers), which is imported lazily and only when actually used.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import sys
from pathlib import Path
from typing import Iterable, Iterator

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CHUNKS = REPO / "pmc" / "chunks" / "chunks.jsonl"
DEFAULT_OUT = REPO / "pmc" / "index"

ARTICLE_ENCODER = "ncbi/MedCPT-Article-Encoder"
QUERY_ENCODER = "ncbi/MedCPT-Query-Encoder"
CROSS_ENCODER = "ncbi/MedCPT-Cross-Encoder"
MEDCPT_DIM = 768
MEDCPT_MAX_TOKENS = 512

# Metadata every retrieved candidate must still carry. Losing any of these would
# make the recency-bias study impossible to run later.
MANIFEST_FIELDS = [
    "chunk_id", "document_id", "pmcid", "pmid", "doi", "title",
    "source_category", "eligibility_status", "canonical_date", "date_precision",
    "date_source", "split_june_2024", "authority_tier_label", "guideline_family",
    "in_currency_pack", "retracted", "license_code", "license_band",
    "location", "section_id", "section_heading", "imrad", "word_count",
    "text_sha256", "duplicate_of",
]


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------
class StubEncoder:
    """Deterministic hash-based vectors. Testing only -- never a real index."""

    name = "stub-sha256"
    production = False

    def __init__(self, dim: int = 64):
        self.dim = dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            vals: list[float] = []
            counter = 0
            while len(vals) < self.dim:
                h = hashlib.sha256(f"{counter}\x00{t}".encode("utf-8")).digest()
                for i in range(0, len(h), 2):
                    if len(vals) >= self.dim:
                        break
                    vals.append((int.from_bytes(h[i:i + 2], "big") / 65535.0) - 0.5)
                counter += 1
            norm = sum(v * v for v in vals) ** 0.5 or 1.0
            out.append([v / norm for v in vals])
        return out


class MedCPTEncoder:
    """The production encoder. Requires torch + transformers; GPU strongly advised."""

    production = True

    def __init__(self, model_id: str = ARTICLE_ENCODER, device: str | None = None,
                 batch_size: int = 32):
        try:
            import torch                                    # noqa: F401
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:                           # pragma: no cover
            raise SystemExit(
                "MedCPT requires torch and transformers:\n"
                "  pip install torch transformers\n"
                f"(import failed: {exc})"
            ) from exc
        import torch
        self.torch = torch
        self.name = model_id
        self.dim = MEDCPT_DIM
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()

    def encode(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover
        torch = self.torch
        out: list[list[float]] = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                enc = self.tokenizer(batch, truncation=True, padding=True,
                                     max_length=MEDCPT_MAX_TOKENS,
                                     return_tensors="pt").to(self.device)
                # MedCPT uses the [CLS] representation.
                vecs = self.model(**enc).last_hidden_state[:, 0, :]
                vecs = torch.nn.functional.normalize(vecs, p=2, dim=1)
                out.extend(vecs.float().cpu().tolist())
        return out


def get_encoder(kind: str, dim: int, device: str | None, batch_size: int):
    if kind == "medcpt":
        return MedCPTEncoder(ARTICLE_ENCODER, device, batch_size)
    if kind == "stub":
        return StubEncoder(dim)
    raise SystemExit(f"unknown encoder: {kind}")


# ---------------------------------------------------------------------------
# Chunk reading
# ---------------------------------------------------------------------------
def read_chunks(path: Path, skip_duplicates: bool) -> list[dict]:
    """All chunks in deterministic (sorted chunk_id) order.

    Exact-duplicate chunk texts were flagged, not deleted, by build_chunks.
    Embedding them again adds nothing to the index, so they are skipped by
    default -- but the chunk layer itself is untouched and the duplicate is
    still resolvable through duplicate_of.
    """
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if skip_duplicates and c.get("duplicate_of"):
                continue
            rows.append(c)
    rows.sort(key=lambda c: c["chunk_id"])
    return rows


def embed_text_of(chunk: dict) -> str:
    """The exact string the encoder sees. One shared rule with the chunk layer."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_chunks import compose_embed_text
    return compose_embed_text(chunk.get("title", ""), chunk.get("section_heading", ""),
                              chunk.get("text", ""))


def batched(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build(chunks_path: Path, out_dir: Path, encoder, batch_size: int,
          skip_duplicates: bool = True) -> dict:
    chunks = read_chunks(chunks_path, skip_duplicates)
    out_dir.mkdir(parents=True, exist_ok=True)
    vec_path = out_dir / "embeddings.f32"
    man_path = out_dir / "index_manifest.jsonl"

    digest = hashlib.sha256()
    n = 0
    by_category: dict[str, int] = {}

    with vec_path.open("wb") as vf, man_path.open("w", encoding="utf-8", newline="\n") as mf:
        for batch in batched(chunks, batch_size):
            vectors = encoder.encode([embed_text_of(c) for c in batch])
            if len(vectors) != len(batch):
                raise SystemExit("encoder returned the wrong number of vectors")
            for chunk, vec in zip(batch, vectors):
                if len(vec) != encoder.dim:
                    raise SystemExit(
                        f"encoder returned dim {len(vec)}, expected {encoder.dim}")
                buf = array.array("f", vec)
                if sys.byteorder != "little":                # pragma: no cover
                    buf.byteswap()
                raw = buf.tobytes()
                vf.write(raw)
                digest.update(chunk["chunk_id"].encode("utf-8"))
                digest.update(raw)
                row = {k: chunk.get(k, "") for k in MANIFEST_FIELDS}
                row["row"] = n
                mf.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                cat = chunk.get("source_category", "")
                by_category[cat] = by_category.get(cat, 0) + 1
                n += 1

    meta = {
        "encoder": encoder.name,
        "production": bool(getattr(encoder, "production", False)),
        "dim": encoder.dim,
        "vectors": n,
        "vectors_by_source_category": dict(sorted(by_category.items())),
        "skipped_exact_duplicates": skip_duplicates,
        "dtype": "float32",
        "byte_order": "little",
        "normalized": True,
        "similarity": "inner-product (cosine on normalized vectors)",
        "search": "exact flat scan -- no approximate index, so results are reproducible",
        "content_digest": digest.hexdigest(),
        "chunks_source": str(chunks_path.name),
        "query_encoder": QUERY_ENCODER,
        "cross_encoder": CROSS_ENCODER,
    }
    (out_dir / "index_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Embed the frozen chunk layer.")
    ap.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--encoder", choices=["medcpt", "stub"], default="medcpt")
    ap.add_argument("--dim", type=int, default=64, help="stub encoder dimension only")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--include-duplicates", action="store_true",
                    help="also embed chunks flagged as exact duplicates")
    ap.add_argument("--allow-stub", action="store_true",
                    help="permit writing a non-production stub index")
    args = ap.parse_args(argv)

    if args.encoder == "stub" and not args.allow_stub:
        raise SystemExit(
            "Refusing to write a stub index without --allow-stub.\n"
            f"The thesis requires {ARTICLE_ENCODER}; use --encoder medcpt."
        )

    encoder = get_encoder(args.encoder, args.dim, args.device, args.batch_size)
    meta = build(args.chunks, args.out, encoder, args.batch_size,
                 skip_duplicates=not args.include_duplicates)
    print(f"Index written to {args.out}")
    for k, v in meta.items():
        print(f"  {k:30} {v}")
    if not meta["production"]:
        print("\n  WARNING: non-production stub index. Not valid for thesis results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

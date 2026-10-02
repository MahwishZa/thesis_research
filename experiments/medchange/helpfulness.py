"""Zero-shot helpfulness scores for the RAG²-style arms (B2, P, C1).

RAG²'s filter is a Flan-T5 fine-tuned on perplexity labels; that checkpoint is not
distributed and the local retraining failed (docs/log.md, Phases 13-18). This is a
**different, untrained** substitute and is named as such everywhere: the unmodified
Flan-T5-large asked RAG²'s prompt plus "Answer yes or no.", scored as P(yes) at the
first decoder step. It supplies a text-only helpfulness signal on the frozen pool.

    python -m experiments.medchange.helpfulness --split dev --device cpu
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

HERE = Path(__file__).resolve().parent
MODEL = "google/flan-t5-large"
PROMPT = ("Given the following evidence, determine whether it helps answer the provided "
          "question.\n\nEvidence: {evidence}\n\nQuestion: {question}\n\nAnswer yes or no.")
MAX_TOKENS = 512


def build_inputs(question: str, pool: Sequence[dict]) -> list[str]:
    return [PROMPT.format(evidence=f"{c['title']}. {c['abstract']}", question=question)
            for c in pool]


class FlanT5YesNo:
    name = f"{MODEL}/zero-shot-yes-no"

    def __init__(self, device=None, batch_size: int = 4):
        self.device, self.batch_size = device, batch_size
        self._tok = self._model = None

    def _ensure(self):
        if self._model is None:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            self._tok = AutoTokenizer.from_pretrained(MODEL)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).eval()
            if self.device:
                self._model.to(self.device)
            ids = [self._tok.encode(w, add_special_tokens=False) for w in ("yes", "no")]
            if any(len(i) != 1 for i in ids):
                raise RuntimeError(f"'yes'/'no' must each be one token, got {ids}")
            self._yes, self._no = ids[0][0], ids[1][0]
        return self._tok, self._model

    def score(self, texts: Sequence[str]) -> list[float]:
        import torch
        tok, model = self._ensure()
        out: list[float] = []
        start_id = model.config.decoder_start_token_id
        with torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                enc = tok(list(texts[i:i + self.batch_size]), truncation=True,
                          max_length=MAX_TOKENS, padding=True, return_tensors="pt")
                if self.device:
                    enc = {k: v.to(self.device) for k, v in enc.items()}
                dec = torch.full((enc["input_ids"].shape[0], 1), start_id,
                                 device=enc["input_ids"].device)
                logits = model(**enc, decoder_input_ids=dec).logits[:, 0, :]
                two = logits[:, [self._yes, self._no]]
                out.extend(torch.softmax(two, dim=-1)[:, 0].cpu().tolist())
        return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="dev", choices=("dev", "confirm", "all"))
    ap.add_argument("--frozen", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    frozen = Path(args.frozen or HERE / "data" / f"frozen_{args.split}.jsonl")
    out = Path(args.out or HERE / "data" / f"helpfulness_{args.split}.jsonl")
    bench = {r["item_id"]: r for r in map(json.loads, open(HERE / "data" / "benchmark.jsonl", encoding="utf-8"))}
    pools = [json.loads(l) for l in open(frozen, encoding="utf-8")]
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["item_id"])
            except (json.JSONDecodeError, KeyError):
                break
    todo = [p for p in pools if p["item_id"] not in done][:args.limit]
    scorer = FlanT5YesNo(args.device)
    t0 = time.time()
    with open(out, "a", encoding="utf-8", newline="\n") as h:
        for k, p in enumerate(todo, 1):
            s = scorer.score(build_inputs(bench[p["item_id"]]["question"], p["candidates"]))
            h.write(json.dumps({"item_id": p["item_id"], "scorer": scorer.name,
                                "pmids": [c["pmid"] for c in p["candidates"]], "helpful": s}) + "\n")
            h.flush()
            if k % 5 == 0 or k == len(todo):
                print(f"  {k}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

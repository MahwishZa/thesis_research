"""The MedCPT cross-encoder re-ranker."""


MEDCPT_CROSS = "ncbi/MedCPT-Cross-Encoder"


def medcpt_cross_encoder(model_name=MEDCPT_CROSS, max_length=512):
    """A function score(query, [[title, text], ...]) -> one relevance score per passage (higher = more relevant), from the MedCPT
    cross-encoder (Jin et al. 2023), which reads the query and a passage together. The model is downloaded on first use."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok, model = AutoTokenizer.from_pretrained(model_name), AutoModelForSequenceClassification.from_pretrained(model_name).eval()

    @torch.no_grad()
    def score(query, passages, batch=16):
        out = []
        for start in range(0, len(passages), batch):
            pairs = [[query, f"{t} {x}".strip()] for t, x in passages[start:start + batch]]
            enc = tok(pairs, truncation=True, padding=True, return_tensors="pt", max_length=max_length)
            out += model(**enc).logits.squeeze(dim=-1).reshape(-1).tolist()
        return out
    return score

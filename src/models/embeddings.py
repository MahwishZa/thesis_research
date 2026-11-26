"""The retrieval models: the MedCPT encoders and cross-encoder (text) and BiomedCLIP (images). Each is downloaded on first use."""


# ---------------------------------------------------------------------------------------------------------------------------
# Dense retrieval for the RAG systems (standard RAG, 2026-11-24). Passages are embedded once with the MedCPT article encoder, a
# question is embedded with the MedCPT query encoder (Jin et al. 2023, Bioinformatics 39:btad651), and the passages with the highest
# cosine similarity come first. The vectors have unit length, so the cosine is a dot product. Nothing here reads a label or a tag.
# ---------------------------------------------------------------------------------------------------------------------------
MEDCPT_QUERY = "ncbi/MedCPT-Query-Encoder"
MEDCPT_ARTICLE = "ncbi/MedCPT-Article-Encoder"


def medcpt_encoder(model_name, max_length):
    """A function that turns a list of texts (a list of [title, text] pairs for the article encoder) into unit-length vectors, using the
    [CLS] embedding of the model. The model is downloaded on first use."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok, model = AutoTokenizer.from_pretrained(model_name), AutoModel.from_pretrained(model_name).eval()

    @torch.no_grad()
    def encode(items, batch=16):
        out = []
        for start in range(0, len(items), batch):
            x = tok(items[start:start + batch], truncation=True, padding=True, return_tensors="pt", max_length=max_length)
            out += torch.nn.functional.normalize(model(**x).last_hidden_state[:, 0], dim=-1).tolist()
        return out
    return encode


# ---- re-ranking: the MedCPT cross-encoder ------------------------------------------------------------------------------------
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


# ---- images: encoder and the check against the Kaggle images --------------------------------------------------------------
IMAGE_ENCODERS = ("biomedclip",)       # "monet" is added once its loading code is confirmed from its model card
BIOMEDCLIP = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def image_encoder(kind):
    """A function that turns a list of image file paths into unit-length vectors. BiomedCLIP (Zhang et al. 2023) is loaded the way its
    model card does (open_clip, `create_model_from_pretrained`); it needs the `open_clip_torch` and `Pillow` packages and downloads
    the model on first use."""
    if kind != "biomedclip":
        raise ValueError(f"unknown image encoder {kind!r}; choose one of {IMAGE_ENCODERS}")
    import open_clip
    import torch
    from PIL import Image
    model, preprocess = open_clip.create_model_from_pretrained(BIOMEDCLIP)
    model.eval()

    @torch.no_grad()
    def encode(paths, batch=16):
        out = []
        for start in range(0, len(paths), batch):
            images = [preprocess(Image.open(p).convert("RGB")) for p in paths[start:start + batch]]
            out += torch.nn.functional.normalize(model.encode_image(torch.stack(images)), dim=-1).tolist()
        return out
    return encode

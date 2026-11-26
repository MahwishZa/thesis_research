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
IMAGE_ENCODERS = ("biomedclip", "monet")
BIOMEDCLIP = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
# MONET (Kim et al., Nature Medicine 2024): a CLIP ViT-L/14 trained on 105,550 dermatology image-text pairs. Loaded the way its README gives for the
# Hugging Face route ("Using huggingface CLIP implementation": the id below), with the README's own preprocessing written out here, so that no
# processor file decides it: the shorter side to 224 pixels (bicubic), a centre crop of 224, ImageNet mean and standard deviation. The README's
# GitHub repository is licensed CC BY-NC-SA 4.0 (non-commercial; read from its LICENSE file, 2026-11-26); check the licence of the weights on the
# model card before sharing anything made with them.
MONET = "chanwkim/monet"
MONET_SIZE = 224
MONET_MEAN, MONET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def monet_crop_box(width, height, size=MONET_SIZE):
    """(new width, new height, left, top) of MONET's preprocessing: the shorter side becomes `size` and the longer side keeps the ratio
    (cut to a whole number, as torchvision does), then the centre `size` x `size` square is cut out (rounded half to even, as torchvision does)."""
    short, long = (width, height) if width <= height else (height, width)
    new_long = int(size * long / short)
    new_width, new_height = (size, new_long) if width <= height else (new_long, size)
    return new_width, new_height, int(round((new_width - size) / 2.0)), int(round((new_height - size) / 2.0))


def image_encoder(kind):
    """A function that turns a list of image file paths into unit-length vectors. BiomedCLIP (Zhang et al. 2023) is loaded the way its
    model card does (open_clip, `create_model_from_pretrained`); it needs the `open_clip_torch` and `Pillow` packages and downloads
    the model on first use."""
    if kind not in IMAGE_ENCODERS:
        raise ValueError(f"unknown image encoder {kind!r}; choose one of {IMAGE_ENCODERS}")
    if kind == "monet":
        return monet_encoder()
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


def monet_encoder():
    """The MONET image tower as a function that turns a list of image file paths into unit-length vectors (the same contract as the
    BiomedCLIP one). Needs torch, transformers (already installed), Pillow and numpy; downloads the weights on first use. The text tower is
    never used: an image is compared with images. `encode.info` records what was loaded."""
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoModelForZeroShotImageClassification
    model = AutoModelForZeroShotImageClassification.from_pretrained(MONET).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    mean, std = (torch.tensor(v).view(1, 3, 1, 1) for v in (MONET_MEAN, MONET_STD))

    def prepare(path):
        with Image.open(path) as image:
            image = image.convert("RGB")
            width, height, left, top = monet_crop_box(*image.size)
            image = image.resize((width, height), Image.Resampling.BICUBIC).crop((left, top, left + MONET_SIZE, top + MONET_SIZE))
            return torch.from_numpy(np.array(image, dtype="float32") / 255.0).permute(2, 0, 1)

    @torch.no_grad()
    def encode(paths, batch=16):
        out = []
        for start in range(0, len(paths), batch):
            x = ((torch.stack([prepare(p) for p in paths[start:start + batch]]) - mean) / std).to(device)
            pooled = model.vision_model(pixel_values=x).pooler_output          # the image tower and its projection, as get_image_features does
            out += torch.nn.functional.normalize(model.visual_projection(pooled), dim=-1).cpu().tolist()
        return out
    encode.info = {"model": MONET, "commit": getattr(model.config, "_commit_hash", None),        # saved with the vectors: a change is noticed
                   "preprocess": "shorter side 224 (bicubic), centre crop 224, ImageNet mean and std (MONET README)"}
    return encode

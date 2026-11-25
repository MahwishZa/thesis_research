"""The image library for RAG: the skin-image figures of the corpus papers (visual), their pictures (visual-images), and the check
that no library picture looks like a Kaggle image (visual-calibrate)."""
import argparse
import datetime
import hashlib
import json
import operator
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter

from src.retrieval.corpus import (
    NCBI,
    PMC_QUERIES,
    TOOL,
    XLINK_HREF,
    Fetcher,
    clean,
    esearch,
    is_ai_paper,
    open_licence,
    parse_pmc_licence,
    tag,
)
from src.utils.common import ALL_CASES, CLASSES, CORPUS_DIR, PROCESSED_IMAGES, hamming
from src.utils.io import read_csv, read_jsonl, write_json, write_jsonl
from src.utils.logging import get_logger

log = get_logger("corpus")      # the corpus, reference, index and visual commands share this logger


def dhash_bits(pixels, size=8):
    """64 bits from a size x (size + 1) grid of grey values, row by row: one bit per pair of neighbours (left brighter than right)."""
    bits = 0
    for r in range(size):
        for c in range(size):
            bits = (bits << 1) | int(pixels[r * (size + 1) + c] > pixels[r * (size + 1) + c + 1])
    return bits


def _dhash(img, size=8):
    """The difference hash of a PIL image: shrunk to 9 x 8 grey pixels."""
    from PIL import Image
    small = img.convert("L").resize((size + 1, size), Image.LANCZOS)
    pixels = small.get_flattened_data() if hasattr(small, "get_flattened_data") else small.getdata()   # getdata: deprecated in Pillow 13
    return dhash_bits(list(pixels), size)


def dhash_variants(path):
    """The hash of an image file and of its mirror images and half turn (a figure may be shown flipped): 4 integers."""
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB")
        return [_dhash(v) for v in (im, im.transpose(Image.FLIP_LEFT_RIGHT), im.transpose(Image.FLIP_TOP_BOTTOM),
                                    im.transpose(Image.ROTATE_180))]


def max_similarities(vectors, reference_vectors):
    """For each vector, its largest dot product (cosine, as all vectors have unit length) with any reference vector, and which one."""
    try:
        import numpy as np
        sims = np.asarray(vectors, dtype="float32") @ np.asarray(reference_vectors, dtype="float32").T
        best = sims.argmax(axis=1)
        return [(float(sims[i, j]), int(j)) for i, j in enumerate(best)]
    except ImportError:                                     # small inputs only (the tests); the laptop has numpy
        out = []
        for v in vectors:
            scores = [sum(map(operator.mul, v, r)) for r in reference_vectors]
            j = max(range(len(scores)), key=lambda i: (scores[i], -i))
            out.append((scores[j], j))
        return out


def drop_near_duplicates(vectors, threshold):
    """Greedy, in order: a vector is kept unless its cosine with an already kept vector is at least `threshold`.
    Returns (indices kept, [(index dropped, index of the kept look-alike, similarity), ...])."""
    kept, dropped = [], []
    try:
        import numpy as np
        matrix = np.asarray(vectors, dtype="float32")
        for i in range(len(matrix)):
            if kept:
                sims = matrix[kept] @ matrix[i]
                j = int(sims.argmax())
                if float(sims[j]) >= threshold:
                    dropped.append((i, kept[j], float(sims[j])))
                    continue
            kept.append(i)
    except ImportError:                                     # small inputs only (the tests); the laptop has numpy
        for i, v in enumerate(vectors):
            best = max(((sum(map(operator.mul, v, vectors[j])), j) for j in kept), default=(-2.0, -1))
            if best[0] >= threshold:
                dropped.append((i, best[1], best[0]))
            else:
                kept.append(i)
    return kept, dropped


def kaggle_vectors(encode, names, data, folder, encoder):
    """The vectors of the Kaggle images, from a saved file when the encoder and the list of images are the same as when it was made
    (embedding 2,000 images takes minutes, and the check is repeated whenever the library changes)."""
    path = os.path.join(folder, f"kaggle_vectors_{encoder}.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            saved = json.load(f)
        if saved.get("names") == names:
            return saved["vectors"]
    vectors = encode([os.path.join(data, PROCESSED_IMAGES, f) for f in names])
    write_json(path, {"encoder": encoder, "names": names, "vectors": vectors})
    return vectors


def kaggle_hashes(names, data, folder, hash_fn):
    """The picture hashes of the Kaggle images (saved, like their vectors, for the same list of images)."""
    path = os.path.join(folder, "kaggle_hashes.json")
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            saved = json.load(f)
        if saved.get("names") == names:
            return saved["hashes"]
    hashes = [hash_fn(os.path.join(data, PROCESSED_IMAGES, f))[0] for f in names]
    write_json(path, {"names": names, "hashes": hashes})
    return hashes


def nearest_by_hash(variant_hashes, reference_hashes):
    """(smallest number of different bits, index of that reference image) for one image, over its 4 variants."""
    return min((hamming(v, h), j) for v in variant_hashes for j, h in enumerate(reference_hashes))


# ----------------------------------------------------------------------------------------------------------------------
# visual-calibrate: how well the two look-alike tests separate known cases (python -m src.main visual-calibrate)
# ----------------------------------------------------------------------------------------------------------------------
# The thresholds of visual-index (embedding 0.95, hash 4 bits) were chosen before any data was seen. This command measures them on the
# Kaggle images themselves, using what the preprocess command already knows: images with the same group_id show the SAME LESION in different
# photos (the hardest kind of look-alike a library figure could be), and images of different groups show different lesions. For both
# kinds it reports the embedding similarity and the hash distance, and how many of each kind the current thresholds would flag:
#   caught      - share of same-lesion pairs flagged (higher is safer: a library copy of a test lesion would be caught)
#   false alarm - share of images whose most similar DIFFERENT lesion would be flagged (the cost: good library images lost)
# Only the lesion grouping is used, never a diagnosis. The Kaggle vectors and hashes saved by visual-index are reused.
def quantiles(values, qs=(1, 5, 50, 95, 99)):
    v = sorted(values)
    if not v:
        return {}
    out = {f"q{q}": round(v[min(len(v) - 1, int(len(v) * q / 100))], 4) for q in qs}
    return {"min": round(v[0], 4), **out, "max": round(v[-1], 4), "n": len(v)}


def guard_calibration(vectors, hashes, groups, threshold, hash_threshold):
    """The same-lesion and different-lesion numbers described above. vectors and hashes are in the order of groups."""
    n = len(groups)
    try:
        import numpy as np
        m = np.asarray(vectors, dtype="float32")
        table = (m @ m.T).tolist()
    except ImportError:                                                       # small inputs only (the tests)
        table = [[sum(a * b for a, b in zip(u, v)) for v in vectors] for u in vectors]

    def sim(i, j):
        return table[i][j]
    members = {}
    for i, g in enumerate(groups):
        members.setdefault(g, []).append(i)
    same_sim, same_hash, caught = [], [], 0
    for idx in members.values():
        for x in range(len(idx)):
            for y in range(x + 1, len(idx)):
                i, j = idx[x], idx[y]
                s_, h_ = sim(i, j), hamming(hashes[i], hashes[j])
                same_sim.append(s_)
                same_hash.append(h_)
                caught += s_ >= threshold or h_ <= hash_threshold
    diff_sim, diff_hash, alarms = [], [], 0
    for i in range(n):
        others = [j for j in range(n) if groups[j] != groups[i]]
        if not others:
            continue
        best_sim = max(sim(i, j) for j in others)
        best_hash = min(hamming(hashes[i], hashes[j]) for j in others)
        diff_sim.append(best_sim)
        diff_hash.append(best_hash)
        alarms += best_sim >= threshold or best_hash <= hash_threshold
    return {"thresholds": {"embedding": threshold, "hash_bits": hash_threshold},
            "same_lesion_pairs": {"n": len(same_sim), "embedding_similarity": quantiles(same_sim),
                                  "hash_bits_different": quantiles(same_hash),
                                  "caught": round(caught / len(same_sim), 4) if same_sim else None},
            "most_similar_different_lesion": {"n": len(diff_sim), "embedding_similarity": quantiles(diff_sim),
                                              "hash_bits_different": quantiles(diff_hash),
                                              "false_alarm": round(alarms / len(diff_sim), 4) if diff_sim else None}}


def visual_calibrate_main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.main visual-calibrate", description="Measures the look-alike tests on known Kaggle cases.")
    ap.add_argument("--corpus", default=os.path.join("data", CORPUS_DIR))
    ap.add_argument("--data", default="data")
    ap.add_argument("--encoder", default="biomedclip")
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--hash-threshold", type=int, default=4)
    a = ap.parse_args(argv)
    log = get_logger("visual-calibrate")
    folder = os.path.join(a.corpus, "visual")
    paths = (os.path.join(folder, f"kaggle_vectors_{a.encoder}.json"), os.path.join(folder, "kaggle_hashes.json"))
    if not all(os.path.isfile(p) for p in paths):
        raise SystemExit("run the visual-index command first (it saves the Kaggle vectors and hashes)")
    with open(paths[0], encoding="utf-8") as f:
        vec = json.load(f)
    with open(paths[1], encoding="utf-8") as f:
        hsh = json.load(f)
    if vec["names"] != hsh["names"]:
        raise SystemExit("the saved Kaggle vectors and hashes are not for the same images: run visual-index again")
    group_of = {r["isic_id"]: r["group_id"] for r in read_csv(os.path.join(a.data, ALL_CASES))}
    names = [n for n in vec["names"] if n[:-4] in group_of]
    keep = [i for i, n in enumerate(vec["names"]) if n[:-4] in group_of]
    result = guard_calibration([vec["vectors"][i] for i in keep], [hsh["hashes"][i] for i in keep],
                               [group_of[n[:-4]] for n in names], a.threshold, a.hash_threshold)
    card_path = os.path.join(a.corpus, "visual_card.json")
    card = {}
    if os.path.isfile(card_path):
        with open(card_path, encoding="utf-8") as f:
            card = json.load(f)
    card["guard_calibration"] = result
    write_json(card_path, card)
    same, diff = result["same_lesion_pairs"], result["most_similar_different_lesion"]
    log.info(f"same lesion, other photo ({same['n']} pairs): caught {same['caught']}; similarity {same['embedding_similarity']}; "
             f"hash bits {same['hash_bits_different']}")
    log.info(f"most similar different lesion ({diff['n']} images): false alarm {diff['false_alarm']}; similarity "
             f"{diff['embedding_similarity']}; hash bits {diff['hash_bits_different']}")
    return result


# ----------------------------------------------------------------------------------------------------------------------
# visual: the figures of the visual library (python -m src.main visual)
# ----------------------------------------------------------------------------------------------------------------------
# The visual library is made of real figures with their real captions, taken from the open-access PMC papers that are already in the
# text corpus (same papers, same licences). Nothing is drawn, described or captioned by a model. This step makes the LIST of figures
# (no image is downloaded here): it reads each paper's full-text XML (the same request the corpus build already made, so it comes
# from the download cache) and keeps a figure only when
#   1. it has an image file,
#   2. its caption has at least 8 words and is not a duplicate of another kept caption,
#   3. its caption is not about a chart, a flow diagram, a table of results or another imaging method (MRI, CT, ultrasound, OCT,
#      confocal, X-ray, endoscopy),
#   4. its caption names a dermoscopic image, a clinical photograph or a lesion image, and is not only about histology,
#   5. its caption or the paper's title names one of the 8 diagnoses (the same word lists as the text corpus).
# The list is written to data/corpus/visual/items.jsonl (licensed text: not committed) and a card with the counts to
# data/corpus/visual_card.json (committed). Downloading the images and the check against the Kaggle images are separate
# steps, run only after this list has been looked at.
FIG_NOT_A_SKIN_IMAGE = re.compile(
    r"flow ?chart|flow diagram|forest plot|kaplan|prisma|algorithm|schematic|workflow|bar (?:chart|graph)|histogram|roc curve|"
    r"scatter|heat ?map|pathway|\bmri\b|magnetic resonance|\bct\b|computed tomograph|ultrasound|ultrasonograph|\boct\b|"
    r"optical coherence|confocal|\bpet\b|radiograph|x-ray|mammogra|endoscop|colonoscop|fundus|flow-chart|pie chart|word cloud",
    re.IGNORECASE)
FIG_SKIN_IMAGE = re.compile(
    r"dermoscop|dermatoscop|trichoscop|clinical (?:image|photo|picture|presentation|appearance|view|feature|finding)|photograph|"
    r"macroscopic|close-?up|\blesions?\b", re.IGNORECASE)
FIG_HISTOLOGY = re.compile(
    r"h ?& ?e\b|hematoxylin|haematoxylin|histolog|histopatholog|immunohistochem|immunostain|magnification|biopsy specimen|"
    r"original magnification|×\s?\d+|\bx ?\d{2,3}\b|panoramic|low[- ]power|high[- ]power|\bstain", re.IGNORECASE)
# words that say the picture is of the skin itself ("clinical-pathological correlation" is not one of them)
FIG_CLINICAL_WORD = re.compile(
    r"dermoscop|dermatoscop|trichoscop|clinical (?:image|photo|picture|presentation|appearance|view)|photograph|macroscopic|close-?up",
    re.IGNORECASE)


def parse_pmc_figures(xml_text):
    """The figures of one PMC paper, from its full-text XML: [{"label", "caption", "href"}, ...] (href is None for a figure without an
    image file). The caption is its own text, whitespace tidied."""
    root = ET.fromstring(xml_text)
    out = []
    for fig in root.iter("fig"):
        label = clean("".join(fig.find("label").itertext())) if fig.find("label") is not None else ""
        cap = fig.find("caption")
        blocks = [b for b in cap if b.tag in ("title", "p")] if cap is not None else []
        caption = clean(" ".join("".join(b.itertext()) for b in blocks) if blocks else "".join(cap.itertext()) if cap is not None else "")
        graphic = next((g for g in fig.iter("graphic") if g.get(XLINK_HREF)), None)
        out.append({"label": label, "caption": caption, "href": graphic.get(XLINK_HREF) if graphic is not None else None})
    return out


def figure_reject_reason(fig, title, seen_captions, min_words=8):
    """Why a figure is left out of the visual library, or None to keep it (and then remember its caption in `seen_captions`)."""
    caption = fig["caption"]
    if not fig["href"]:
        return "no image file"
    if len(caption.split()) < min_words:
        return "caption too short"
    if FIG_NOT_A_SKIN_IMAGE.search(caption):
        return "chart, diagram or other imaging method"
    if not FIG_SKIN_IMAGE.search(caption):
        return "not a clinical or dermoscopic image"
    if FIG_HISTOLOGY.search(caption) and not FIG_CLINICAL_WORD.search(caption):
        return "histology only"
    if not tag(caption, title):
        return "no diagnosis named"
    key = hashlib.md5(re.sub(r"\W+", " ", caption.lower()).strip().encode()).hexdigest()
    if key in seen_captions:
        return "duplicate caption"
    seen_captions.add(key)
    return None


def visual_items(papers, fetch_xml, removed):
    """The kept figures of the given papers ([{"doc_id", "title", "licence", "query_class", "url"}]); fetch_xml(doc_id) gives the XML."""
    seen, out = set(), []
    for d in papers:
        try:
            figs = parse_pmc_figures(fetch_xml(d["doc_id"]))
        except Exception as e:                                       # one paper must not stop the build; the card counts them
            log.warning(f"{d['doc_id']}: figures not read ({type(e).__name__}: {e})")
            removed["paper: figures not read"] += 1
            continue
        for k, fig in enumerate(figs):
            reason = figure_reject_reason(fig, d["title"], seen)
            if reason:
                removed[reason] += 1
                continue
            item_id = f"{d['doc_id']}#fig{k}"
            # an item is also a passage (passage_id, title, text) so that the same index, prompt and evidence code serve it
            out.append({"item_id": item_id, "passage_id": item_id, "doc_id": d["doc_id"], "doc_title": d["title"],
                        "title": f"{d['title']} ({fig['label'] or 'Figure'})", "text": fig["caption"], "label": fig["label"],
                        "caption": fig["caption"], "href": fig["href"], "library": "visual", "licence": d["licence"], "url": d["url"],
                        "query_class": d["query_class"], "tags": tag(fig["caption"], d["title"]),
                        "n_words": len(fig["caption"].split())})
    return out


def corpus_papers(passages):
    """One record per PMC paper of the text corpus (the first passage of each gives its title, licence, class and web address)."""
    papers = {}
    for p in passages:
        if re.fullmatch(r"PMC\d+", p["doc_id"]) and p["doc_id"] not in papers:
            papers[p["doc_id"]] = {"doc_id": p["doc_id"], "title": p["title"], "licence": p["licence"],
                                   "query_class": p["query_class"], "url": p["url"]}
    return list(papers.values())


def paper_from_xml(doc_id, xml_text, cls):
    """The record of a PMC paper read from its own full-text XML, or None with the reason: (record, None) or (None, reason).
    Left out: no open Creative Commons licence, retracted, or a paper about AI or image datasets (the same rules as the text corpus)."""
    licence, retracted = parse_pmc_licence(xml_text)
    if retracted:
        return None, "retracted"
    if licence is None or not open_licence(licence):
        return None, "no open licence"
    root = ET.fromstring(xml_text)
    title_el, abstract_el = root.find(".//article-title"), root.find(".//abstract")
    title = clean("".join(title_el.itertext())) if title_el is not None else ""
    abstract = clean("".join(abstract_el.itertext())) if abstract_el is not None else ""
    if is_ai_paper(title, abstract):
        return None, "AI or image-dataset paper"
    return {"doc_id": doc_id, "title": title, "licence": licence, "query_class": cls,
            "url": f"https://pmc.ncbi.nlm.nih.gov/articles/{doc_id}/"}, None


def extra_papers(fetch, per_class, known, removed):
    """More open-access PMC papers for the figure list: for each diagnosis, the papers found by the same PMC search as the text corpus
    (PMC_QUERIES), up to `per_class` per diagnosis, without the passage cap of the text corpus. Papers in `known` are skipped."""
    out, seen = [], set(known)
    for cls in CLASSES:
        for uid in esearch(fetch, "pmc", PMC_QUERIES[cls] + ' AND "open access"[filter]', per_class):
            doc_id = f"PMC{uid}"
            if doc_id in seen:
                continue
            seen.add(doc_id)
            try:
                paper, why = paper_from_xml(doc_id, fetch.get(NCBI + "efetch.fcgi", {"db": "pmc", "id": uid, "retmode": "xml"},
                                                              ncbi=True), cls)
            except Exception as e:                           # one paper must not stop the build; the card counts them
                log.warning(f"{doc_id}: not read ({type(e).__name__}: {e})")
                removed["extra paper: not read"] += 1
                continue
            if paper is None:
                removed[f"extra paper: {why}"] += 1
                continue
            out.append(paper)
    return out


def visual_args(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.main visual", description="Lists the skin-image figures of the corpus papers.")
    ap.add_argument("--corpus", default=os.path.join("data", CORPUS_DIR))
    ap.add_argument("--cache", default=None, help="download cache (default: <corpus>/cache)")
    ap.add_argument("--limit", type=int, default=0, help="read only the first N papers (a trial)")
    ap.add_argument("--extra-papers", type=int, default=0,
                    help="also search PMC for up to N more open-access papers per diagnosis (internet), beyond the text corpus papers")
    return ap.parse_args(argv)


def visual_main(argv=None, fetch_xml=None, fetch=None):
    a = visual_args(argv)
    passages = read_jsonl(os.path.join(a.corpus, "passages.jsonl"))
    papers = corpus_papers(passages)
    if a.limit:
        papers = papers[:a.limit]
    removed = Counter()
    if fetch is None and (fetch_xml is None or a.extra_papers):
        fetch = Fetcher(a.cache or os.path.join(a.corpus, "cache"))
    if a.extra_papers:
        papers += extra_papers(fetch, a.extra_papers, {p["doc_id"] for p in papers}, removed)
    if fetch_xml is None:
        def fetch_xml(doc_id):
            return fetch.get(NCBI + "efetch.fcgi", {"db": "pmc", "id": doc_id[3:], "retmode": "xml"}, ncbi=True)
    items = visual_items(papers, fetch_xml, removed)
    out_dir = os.path.join(a.corpus, "visual")
    write_jsonl(os.path.join(out_dir, "items.jsonl"), items)
    per_class = Counter(t for it in items for t in it["tags"])
    card = {"snapshot": datetime.date.today().isoformat(), "n_papers_read": len(papers), "n_figures_kept": len(items),
            "n_papers_with_a_figure": len({it["doc_id"] for it in items}), "figures_per_tag": {c: per_class.get(c, 0) for c in CLASSES},
            "left_out": dict(sorted(removed.items())), "images_downloaded": False,
            "rule": "see the comment above visual_items in src/images.py"}
    write_json(os.path.join(a.corpus, "visual_card.json"), card)
    log.info(f"{len(items)} figures kept from {card['n_papers_with_a_figure']} of {len(papers)} papers; left out: {dict(removed)}")
    return items


# ----------------------------------------------------------------------------------------------------------------------
# visual-images: download the images of the listed figures (python -m src.main visual-images)
# ----------------------------------------------------------------------------------------------------------------------
# Where the images come from: the PMC Open Access data on AWS, the public bucket "pmc-oa-opendata" (NCBI: pmc.ncbi.nlm.nih.gov/tools/
# pmcaws). Each article has a folder "<PMC id>.<version>/" that holds its XML and its figure files. For each figure the folder of the
# newest version is listed (a plain HTTPS request, no account) and the file whose name equals the figure's graphic name (with or
# without its extension) is downloaded. If the bucket has no such file, the older Europe PMC address pattern is tried last.
# An answer counts as an image only when its first bytes are those of a JPEG, PNG or GIF file. A file under 3,000 bytes (an icon) is
# left out, and so is an image that is byte-for-byte the same as one already kept. --probe N tries only N figures, saves nothing and
# prints what each address answered, so that a wrong address pattern can be seen at once. The images are licensed material: they
# stay in data/corpus/visual/images/ (git-ignored); the card records the counts.
S3_BUCKET = "https://pmc-oa-opendata.s3.amazonaws.com/"
EUROPE_PMC_IMAGE = "https://europepmc.org/articles/{doc}/bin/{href}{ext}"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".gif")
IMAGE_MAGIC = {b"\xff\xd8\xff": ".jpg", b"\x89PNG": ".png", b"GIF8": ".gif"}
MIN_IMAGE_BYTES = 3000
MAX_IMAGE_BYTES = 20_000_000
S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def image_extension(data):
    """".jpg", ".png" or ".gif" when the bytes start like such a file, else None."""
    return next((ext for magic, ext in IMAGE_MAGIC.items() if data.startswith(magic)), None)


def download_bytes(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": TOOL})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(MAX_IMAGE_BYTES + 1)


def s3_list(prefix, get, delimiter=""):
    """(keys, sub-prefixes) under a prefix of the PMC bucket (the listing is XML; one page of up to 1,000 entries)."""
    params = {"list-type": "2", "prefix": prefix}
    if delimiter:
        params["delimiter"] = delimiter
    root = ET.fromstring(get(S3_BUCKET + "?" + urllib.parse.urlencode(params)))
    keys = [c.findtext(S3_NS + "Key") for c in root.iter(S3_NS + "Contents")]
    prefixes = [c.findtext(S3_NS + "Prefix") for c in root.iter(S3_NS + "CommonPrefixes")]
    return keys, prefixes


def article_image_keys(doc_id, get, cache):
    """{file stem: key} of the image files in the newest version folder of one article (remembered in `cache`)."""
    if doc_id not in cache:
        _, versions = s3_list(doc_id + ".", get, delimiter="/")

        def number(p):
            m = re.search(r"\.(\d+)/$", p)
            return int(m.group(1)) if m else -1
        newest = max(versions, key=number) if versions else None
        files = {}
        if newest:
            keys, _ = s3_list(newest, get)
            for key in keys:
                base = key.rsplit("/", 1)[-1]
                stem, ext = os.path.splitext(base)
                if ext.lower() in IMAGE_EXTENSIONS:
                    files.setdefault(stem, key)
                    files.setdefault(base, key)
        cache[doc_id] = files
    return cache[doc_id]


def fetch_image(doc_id, href, get=download_bytes, cache=None, trace=None):
    """(bytes, url) of the first address that gives a real image, or (None, None). `trace` collects what each address answered."""
    cache = {} if cache is None else cache
    note = trace.append if trace is not None else (lambda _: None)
    urls = []
    try:
        key = article_image_keys(doc_id, get, cache).get(href) or article_image_keys(doc_id, get, cache).get(os.path.splitext(href)[0])
        if key:
            urls.append(S3_BUCKET + key)
        else:
            note(f"{doc_id}: no file named {href} in the PMC bucket")
    except Exception as e:                                           # the listing failed: say why, then try the other address
        note(f"{doc_id}: bucket listing failed ({type(e).__name__}: {e})")
    urls += [EUROPE_PMC_IMAGE.format(doc=doc_id, href=href, ext=ext) for ext in (".jpg", ".png", ".gif")]
    for url in urls:
        try:
            data = get(url)
        except Exception as e:                                       # not found, refused, a time-out: try the next address
            note(f"{url}: {type(e).__name__}: {e}")
            continue
        if data and len(data) <= MAX_IMAGE_BYTES and image_extension(data):
            return data, url
        note(f"{url}: answered, but not an image file (starts {data[:12]!r})")
    return None, None


def download_images(items, folder, get=download_bytes, pause=0.4, probe=0, trace=None):
    """Saves the images; returns (items that have an image file, counts, the kinds of address that worked)."""
    os.makedirs(folder, exist_ok=True)
    kept, counts, worked, seen, cache = [], Counter(), Counter(), set(), {}
    for it in items[:probe] if probe else items:
        data, url = fetch_image(it["doc_id"], it["href"], get, cache, trace)
        if pause:
            time.sleep(pause)
        if data is None:
            counts["no image at any address"] += 1
            continue
        if len(data) < MIN_IMAGE_BYTES:
            counts["image too small"] += 1
            continue
        digest = hashlib.sha256(data).hexdigest()
        if digest in seen:
            counts["identical image already kept"] += 1
            continue
        seen.add(digest)
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", it["item_id"]) + image_extension(data)
        if not probe:
            with open(os.path.join(folder, name), "wb") as f:
                f.write(data)
        worked["PMC bucket" if url.startswith(S3_BUCKET) else "Europe PMC"] += 1
        kept.append({**it, "image_file": name, "image_sha256": digest, "image_url": url, "image_bytes": len(data)})
    return kept, counts, worked


def visual_images_args(argv=None):
    ap = argparse.ArgumentParser(prog="python -m src.main visual-images", description="Downloads the images of the listed figures.")
    ap.add_argument("--corpus", default=os.path.join("data", CORPUS_DIR))
    ap.add_argument("--probe", type=int, default=0, help="try only the first N figures and save nothing: which addresses work?")
    return ap.parse_args(argv)


def visual_images_main(argv=None, get=download_bytes):
    a = visual_images_args(argv)
    log = get_logger("visual-images")
    folder = os.path.join(a.corpus, "visual")
    items = read_jsonl(os.path.join(folder, "items.jsonl"))
    trace = []
    kept, counts, worked = download_images(items, os.path.join(folder, "images"), get, pause=0.4 if get is download_bytes else 0,
                                           probe=a.probe, trace=trace)
    if a.probe or not kept:
        for line in trace[:12]:
            log.info("  " + line)
    log.info(f"{len(kept)} images of {a.probe or len(items)} figures tried; addresses that worked: {dict(worked)}; "
             f"left out: {dict(counts)}")
    if not a.probe:
        write_jsonl(os.path.join(folder, "items_with_images.jsonl"), kept)
        card_path = os.path.join(a.corpus, "visual_card.json")
        with open(card_path, encoding="utf-8") as f:
            card = json.load(f)
        card.update({"images_downloaded": True, "n_images": len(kept), "images_left_out": dict(sorted(counts.items())),
                     "image_addresses_that_worked": dict(worked)})
        write_json(card_path, card)
    return kept

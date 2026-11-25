"""The visual library: skin-image figures and their captions from open-access PMC papers (python -m src.main visual)
and their image files (visual-images)."""
import argparse
import datetime
import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter

from src.knowledge.collect import NCBI, PMC_QUERIES, TOOL, Fetcher, esearch
from src.knowledge.extract import (
    XLINK_HREF,
    clean,
    is_ai_paper,
    open_licence,
    parse_pmc_licence,
)
from src.knowledge.metadata import tag
from src.utils.common import CLASSES, CORPUS_DIR
from src.utils.io import read_jsonl, write_json, write_jsonl
from src.utils.logging import get_logger

log = get_logger("corpus")      # the corpus, reference, index and visual commands share this logger


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
    get_logger("corpus", os.path.join(a.corpus, "build_visual.log"))       # the module logger "corpus" writes this command's log
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
            "rule": "see the comment above visual_items in src/knowledge/images.py"}
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
    log = get_logger("visual-images", os.path.join(a.corpus, "build_visual.log"))
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

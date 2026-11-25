"""The check that no library image looks like a Kaggle image (picture hash and embedding), its calibration on known
same-lesion photos (visual-calibrate) and the check for the same picture twice in the library."""
import argparse
import json
import operator
import os

from src.utils.common import ALL_CASES, CORPUS_DIR, PROCESSED_IMAGES
from src.utils.io import read_csv, write_json
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


def hamming(a, b):
    return (a ^ b).bit_count()


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
    log = get_logger("visual-calibrate", os.path.join(a.corpus, "build_visual.log"))
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

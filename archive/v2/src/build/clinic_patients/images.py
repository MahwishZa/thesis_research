"""Step ``images`` of the clinic patients cohort: every photo that stays, made in one fixed way. The original photos in raw/ are never
changed.

Why: the photos are phone pictures of very different sizes (147 to 3,476 pixels; 108 files are over 5 MB and the largest is 23.6 MB). A
model API may refuse a file that large, and every model shrinks a large photo in its own way, so "the same photo" would not be the
same input. Every model, in every view, receives these files and no others.

What is done to a photo, in this order, and nothing else (nothing is cut, added or enlarged):
1. turned upright if its EXIF orientation tag says so;
2. converted to RGB if it is not (a photo with transparent pixels stops the step);
3. converted to sRGB from its embedded colour profile, if it has one;
4. shrunk, whole and with its proportions kept, so that its longest side is at most ``MAX_SIDE`` pixels (a smaller photo keeps its size);
5. saved as JPEG with one setting for all (quality 95, full colour resolution) and no metadata, as files/photos/<split>/<class>/<img_id>.jpg.

The step proves that this changes the picture as little as it can, for every photo (it stops if a bar is not met; the bars were fixed
before the numbers were seen):
- **fidelity:** the photo after steps 1 to 4 against its saved JPEG, as PSNR in dB: at least ``MIN_PSNR`` (above about 35 dB two pictures
  look the same);
- **same picture:** the 256-bit difference hash of the saved JPEG (step ``dedupe`` explains the hash) must be nearer to the hash of its own
  original than to that of any other photo's original: a saved photo is never mistaken for another photo;
- **size:** no saved file larger than ``MAX_FILE_BYTES``, below the limits of the model providers.

Writes files/photos/ (not committed), build/files.csv (every photo: original and saved file, size, fingerprints, what was done, the two
measures) and the images section of dataset_report.md. The table of questions (dataset.csv) is written by step ``views``, which joins
these photos to each patient's record.
"""

from __future__ import annotations

import hashlib
import io
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageOps

from ...common import write_section
from .dedupe import CHUNK, distances, dhash
from .table import BUILD, DATA, PHOTOS, RAW, REPORT, REPORT_HEAD, SPLIT, STEPS, names, read_csv, remove_stale, sha256, write_csv

MAX_SIDE = 1024  # the longest side of a saved photo, in pixels
JPEG = {"quality": 95, "subsampling": 0}  # one setting for every photo: high quality, full colour resolution (4:4:4), no metadata
MIN_PSNR = 30.0  # dB: no saved photo may be further than this from the original (above about 35 dB two pictures look the same)
MAX_FILE_BYTES = 4_000_000  # no saved file may be larger: model providers refuse files of about 5 MB
ORIENTATION = 0x0112
FILE_COLUMNS = ["img_id", "path", "sha256", "bytes", "width", "height", "raw_path", "raw_sha256", "raw_bytes", "raw_width", "raw_height",
                "changes", "psnr_db", "hash_bits", "nearest_other_bits"]


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB of two 8-bit pictures of the same size (99 if they are equal)."""
    mse = float(np.mean((a.astype(np.int32) - b.astype(np.int32)) ** 2))
    return 99.0 if mse == 0 else min(99.0, 10 * float(np.log10(255.0 ** 2 / mse)))


def prepare_image(content: bytes) -> tuple[bytes, float, list[str], tuple[int, int]]:
    """One original photo as every model receives it: the JPEG bytes, its PSNR against the photo before saving (dB), what was done
    to it besides saving it, and the size of the original. Raises ValueError for a photo with transparent pixels."""
    changes = []
    with Image.open(io.BytesIO(content)) as original:
        original.load()
        raw_size = original.size
        if (original.mode in ("RGBA", "LA", "PA") or "transparency" in original.info) \
                and original.convert("RGBA").getchannel("A").getextrema()[0] < 255:
            raise ValueError("it has transparent pixels")
        image = ImageOps.exif_transpose(original)
        if original.getexif().get(ORIENTATION, 1) != 1:
            changes.append("turned upright as its EXIF orientation tag says")
        if image.mode != "RGB":
            changes.append(f"converted from {original.mode} to RGB")
            image = image.convert("RGB")
        profile = original.info.get("icc_profile")
        if profile:
            source = ImageCms.ImageCmsProfile(io.BytesIO(profile))
            name = ImageCms.getProfileDescription(source).strip() or "unnamed"
            image = ImageCms.profileToProfile(image, source, ImageCms.createProfile("sRGB"), outputMode="RGB")
            changes.append(f"colours converted from its embedded colour profile ({name}) to sRGB")
    longest = max(image.size)
    if longest > MAX_SIDE:
        scale = MAX_SIDE / longest
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        changes.append(f"shrunk from {image.width}x{image.height} to {size[0]}x{size[1]}")
        image = image.resize(size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", **JPEG)  # only these settings: no EXIF, no colour profile, no thumbnail is written
    saved = buffer.getvalue()
    with Image.open(io.BytesIO(saved)) as again:
        quality = psnr(np.asarray(image), np.asarray(again.convert("RGB")))
    return saved, quality, changes or ["saved again only"], raw_size


def images(data: Path = DATA) -> dict:
    """Makes the photos, files/photos/, build/files.csv and the images section of the report (see the module's description)."""
    data = Path(data)
    questions = read_csv(data / BUILD / SPLIT, "run the split step first")
    raw = {r["img_id"]: r for r in read_csv(data / BUILD / "raw_images.csv", "run the inventory step first")}
    hashes = {r["img_id"]: bytes.fromhex(r["dhash"]) for r in read_csv(data / BUILD / "photo_hashes.csv", "run the dedupe step first")}
    root = data / PHOTOS
    rows, written, saved_hashes = [], 0, []
    for q in questions:
        entry = raw[q["img_id"]]
        source = data / RAW / entry["file"]
        if not source.is_file():
            raise FileNotFoundError(f"{source.as_posix()} not found: the original download must stay in {(data / RAW).as_posix()}")
        content = source.read_bytes()
        if sha256(source) != entry["sha256"]:
            raise ValueError(f"{source.as_posix()} differs from build/raw_images.csv: the original download must stay as downloaded "
                             "(run the inventory step again if it was replaced on purpose)")
        try:
            saved, quality, changes, raw_size = prepare_image(content)
        except ValueError as exc:
            raise ValueError(f"{q['img_id']}: {exc}") from exc
        path = f"{q['split']}/{q['label']}/{Path(q['img_id']).stem}.jpg"
        file = root / path
        if not file.is_file() or file.read_bytes() != saved:
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(saved)
            written += 1
        with Image.open(io.BytesIO(saved)) as photo:
            width, height = photo.size
        rows.append({"img_id": q["img_id"], "path": path, "sha256": hashlib.sha256(saved).hexdigest(), "bytes": len(saved), "width": width, "height": height,
                     "raw_path": entry["file"], "raw_sha256": entry["sha256"], "raw_bytes": entry["bytes"], "raw_width": raw_size[0],
                     "raw_height": raw_size[1], "changes": "; ".join(changes), "psnr_db": round(quality, 2)})
        saved_hashes.append(_hash_of(saved))
    originals = np.stack([np.frombuffer(hashes[r["img_id"]], dtype=np.uint8) for r in rows])
    saved_hashes = np.stack(saved_hashes)
    for first in range(0, len(rows), CHUNK):
        block = distances(saved_hashes, first, min(first + CHUNK, len(rows)), originals)
        for r in range(block.shape[0]):
            i = first + r
            rows[i]["hash_bits"] = int(block[r, i])
            block[r, i] = 10 ** 6  # the photo's own original is not "another" photo
            rows[i]["nearest_other_bits"] = int(block[r].min()) if len(rows) > 1 else 256
    _check(rows)
    removed = remove_stale(root, {row["path"] for row in rows}, ".jpg")  # a photo that is no longer in a split: never left for a run to find
    write_csv(data / BUILD / "files.csv", FILE_COLUMNS, rows)
    write_section(data / REPORT, REPORT_HEAD, STEPS, "images", _report(rows))
    return {"summary": f"{len(rows)} photos made ({written} written, {removed} removed); fidelity at least {min(r['psnr_db'] for r in rows):.1f} dB, "
                       f"own original always the nearest (own at most {max(r['hash_bits'] for r in rows)} bits, others at least {min(r['nearest_other_bits'] for r in rows)}); largest file "
                       f"{max(r['bytes'] for r in rows) / 1e6:.2f} MB", "photos": len(rows), "written": written, "removed": removed}


def _hash_of(content: bytes) -> np.ndarray:
    """The 256-bit difference hash of JPEG bytes (as dedupe.dhash makes it from a file)."""
    return dhash(io.BytesIO(content))


def _check(rows: list[dict]) -> None:
    """The bars of the module's description; the step stops, naming the photos, if one is not met."""
    for label, bad in (
            (f"fidelity below {MIN_PSNR} dB", [r["img_id"] for r in rows if r["psnr_db"] < MIN_PSNR]),
            ("a picture hash that is not nearer to its own original than to another photo's original",
             [r["img_id"] for r in rows if r["hash_bits"] >= r["nearest_other_bits"]]),
            (f"file larger than {MAX_FILE_BYTES} bytes", [r["img_id"] for r in rows if r["bytes"] > MAX_FILE_BYTES])):
        if bad:
            raise ValueError(f"{len(bad)} photo(s) with {label}: {names(bad)}")


def _report(rows: list[dict]) -> list[str]:
    changes = Counter()
    for r in rows:
        for part in r["changes"].split("; "):
            changes["shrunk" if part.startswith("shrunk") else "colours converted to sRGB" if part.startswith("colours converted")
                    else part] += 1
    raw_bytes = [int(r["raw_bytes"]) for r in rows]
    new_bytes = [r["bytes"] for r in rows]
    longest_raw = [max(int(r["raw_width"]), int(r["raw_height"])) for r in rows]
    longest_new = [max(r["width"], r["height"]) for r in rows]
    quality = [r["psnr_db"] for r in rows]
    bits = [r["hash_bits"] for r in rows]
    others = [r["nearest_other_bits"] for r in rows]
    lines = ["# Images: the photos every model receives", "",
             "Written by `python -m src.build.clinic_patients.prepare images`. Each photo that stays is made from its original in one fixed way "
             "(`files/photos/`, not committed; the originals in `raw/` are never changed). `build/files.csv` lists every photo with what was done "
             "and the two measures below; step `views` joins them to the records and writes `dataset.csv`.", "",
             "## What is done (in this order, and nothing else)", "",
             "1. Turned upright if the EXIF orientation says so. 2. Converted to RGB if needed (transparent pixels stop the step). "
             "3. Converted to sRGB from the embedded colour profile, if there is one.",
             f"4. Shrunk, whole and with its proportions kept, so that the longest side is at most {MAX_SIDE} pixels; a smaller photo keeps its size. "
             f"5. Saved as JPEG (quality {JPEG['quality']}, full colour resolution, no metadata).", "",
             f"- Photos made: {len(rows)}",
             "- " + "; ".join(f"{k}: {v}" for k, v in sorted(changes.items())), "",
             "## Size", "", "| | Originals | Saved photos |", "|---|---|---|",
             f"| Longest side (pixels): smallest, median, largest | {min(longest_raw)}, {statistics.median(longest_raw):g}, {max(longest_raw)} | "
             f"{min(longest_new)}, {statistics.median(longest_new):g}, {max(longest_new)} |",
             f"| File size (KB): median | {statistics.median(raw_bytes) / 1024:.0f} | {statistics.median(new_bytes) / 1024:.0f} |",
             f"| File size (MB): largest | {max(raw_bytes) / 1e6:.2f} | {max(new_bytes) / 1e6:.2f} |",
             f"| Files over 5 MB | {sum(b > 5e6 for b in raw_bytes)} | {sum(b > 5e6 for b in new_bytes)} |", "",
             "## Does it change the picture? (a bar for every photo; the step stops if one is missed)", "",
             "| Measure | Bar | Smallest / median / largest |", "|---|---|---|",
             f"| Fidelity: PSNR of the photo before and after saving (dB; above about 35 two pictures look the same) | at least {MIN_PSNR:g} | "
             f"{min(quality):.1f} / {statistics.median(quality):.1f} / {max(quality):.1f} |",
             f"| Same picture: bits by which the saved photo's 256-bit hash differs from its own original's | below the next measure | "
             f"{min(bits)} / {statistics.median(bits):g} / {max(bits)} |",
             f"| Bits to the nearest *other* photo's original (different photos are about 128 apart) | more than the line above | "
             f"{min(others)} / {statistics.median(others):g} / {max(others)} |", ""]
    return lines

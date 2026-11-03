"""Reads the image dataset (one folder per split, one subfolder per class) and profiles it.

The profile answers the questions that matter before any model is run, without opening every image by hand:
class counts per split, unreadable or unexpected files, image sizes per class, exact duplicates, near-duplicate
candidates, and whether image size alone gives the class away.
"""

from __future__ import annotations

import csv
import hashlib
import io
import itertools
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}
NEAR_DUPLICATE_DISTANCE = 40  # dHash bits out of 256; a candidate list to check by eye, not a verdict


@dataclass(frozen=True)
class Record:
    split: str
    label: str
    path: str  # relative to the dataset root, with forward slashes
    bytes: int
    width: int
    height: int
    sha256: str
    dhash: str  # 256-bit difference hash, hexadecimal
    isic_id: str  # e.g. "ISIC_0024306", or "" when the file name has none


def _dhash(image: Image.Image) -> str:
    small = image.convert("L").resize((17, 16))
    px = small.tobytes()
    bits = 0
    for row in range(16):
        for col in range(16):
            if px[row * 17 + col] > px[row * 17 + col + 1]:
                bits |= 1 << (row * 16 + col)
    return f"{bits:064x}"


def scan(root: str | Path, splits: dict[str, str]) -> tuple[list[Record], list[tuple[str, str]]]:
    """Records for every image, and (path, reason) for every file that is not a readable image.

    ``splits`` maps a split name ("train", "test") to its folder inside ``root``. Each class is a subfolder.
    """
    root = Path(root)
    records, problems = [], []
    for split, folder in splits.items():
        split_dir = root / folder
        if not folder or not split_dir.is_dir():
            raise FileNotFoundError(f"split folder for '{split}' not found: {split_dir}")
        for item in sorted(split_dir.iterdir()):
            if not item.is_dir():
                problems.append((item.relative_to(root).as_posix(), "file outside a class folder"))
                continue
            for file in sorted(p for p in item.rglob("*") if p.is_file()):
                rel = file.relative_to(root).as_posix()
                if file.suffix.lower() not in IMAGE_EXTENSIONS:
                    problems.append((rel, "not an image file"))
                    continue
                data = file.read_bytes()
                try:
                    with Image.open(io.BytesIO(data)) as im:
                        im.load()
                        width, height, dhash = im.width, im.height, _dhash(im)
                except Exception as exc:  # any decoding failure is reported, never silently skipped
                    problems.append((rel, f"unreadable image: {exc}"))
                    continue
                match = re.search(r"ISIC_\d+", file.name)
                records.append(Record(split, item.name, rel, len(data), width, height,
                                      hashlib.sha256(data).hexdigest(), dhash, match.group(0) if match else ""))
    return records, problems


def labels_of(records: list[Record], split: str = "train") -> list[str]:
    return sorted({r.label for r in records if r.split == split})


def items_hash(records: list[Record]) -> str:
    """One hash for a list of images: it changes if any file, label or split changes."""
    lines = sorted(f"{r.split}\t{r.label}\t{r.path}\t{r.sha256}" for r in records)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def dev_subset(records: list[Record], per_class: int, seed: int) -> list[Record]:
    """A seeded, class-stratified sample of training images, used only to develop the prompt."""
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for r in records:
        if r.split == "train":
            by_label[r.label].append(r)
    chosen = []
    for label in sorted(by_label):
        group = sorted(by_label[label], key=lambda r: r.path)
        chosen.extend(sorted(rng.sample(group, min(per_class, len(group))), key=lambda r: r.path))
    return chosen


def prepare_image(path: str | Path, max_side: int) -> tuple[bytes, str, dict]:
    """The bytes to send for one image, their media type, and a record of what was sent.

    An image whose longer side is at most ``max_side`` pixels is sent exactly as stored. A larger one is scaled down so
    its longer side is ``max_side`` (aspect ratio kept, Lanczos filter) and saved as JPEG quality 95. The same rule applies
    to every model; it keeps every request within API size limits.
    """
    path = Path(path)
    data = path.read_bytes()
    with Image.open(io.BytesIO(data)) as im:
        width, height = im.size
        if max(width, height) <= max_side:
            info = {"resized": False, "width": width, "height": height}
            return data, IMAGE_EXTENSIONS[path.suffix.lower()], info | {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        scale = max_side / max(width, height)
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        small = ImageOps.exif_transpose(im).convert("RGB").resize(size, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    small.save(buffer, "JPEG", quality=95)
    out = buffer.getvalue()
    info = {"resized": True, "original_width": width, "original_height": height, "width": size[0], "height": size[1]}
    return out, "image/jpeg", info | {"bytes": len(out), "sha256": hashlib.sha256(out).hexdigest()}


def write_manifest(records: list[Record], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[x.name for x in fields(Record)], lineterminator="\n")
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))


def _contact_sheets(root: Path, records: list[Record], out_dir: Path, per_sheet: int = 36, thumb: int = 128) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    for r in records:
        groups[(r.split, r.label)].append(r)
    for (split, label), group in sorted(groups.items()):
        group = group[:per_sheet]
        cols = 6
        sheet = Image.new("RGB", (cols * thumb, ((len(group) + cols - 1) // cols) * thumb), "white")
        for i, r in enumerate(group):
            with Image.open(root / r.path) as im:
                im = im.convert("RGB")
                im.thumbnail((thumb, thumb))
                sheet.paste(im, ((i % cols) * thumb, (i // cols) * thumb))
        sheet.save(out_dir / f"{split}__{label.replace(' ', '_')}.jpg", quality=85)


def profile(root: str | Path, splits: dict[str, str], out_dir: str | Path, sheets_dir: str | Path | None = None) -> Path:
    """Writes manifest.csv and profile.md to ``out_dir`` (no images) and contact sheets to ``sheets_dir``."""
    root, out_dir = Path(root), Path(out_dir)
    records, problems = scan(root, splits)
    write_manifest(records, out_dir / "manifest.csv")
    if sheets_dir:
        _contact_sheets(root, records, Path(sheets_dir))

    split_names = list(splits)
    labels = sorted({r.label for r in records})
    counts = Counter((r.split, r.label) for r in records)
    lines = ["# Dataset profile", "",
             f"Root: `{root.as_posix()}` | splits: " + ", ".join(f"{s} = `{f}`" for s, f in splits.items()),
             f"Images: {len(records)} | files that are not readable images: {len(problems)}",
             f"Hash of all images (split, label, path, SHA-256): `{items_hash(records)}`", "",
             "## Images per class and split", "",
             "| Class | " + " | ".join(split_names) + " |", "|---|" + "---|" * len(split_names)]
    for label in labels:
        lines.append(f"| {label} | " + " | ".join(str(counts[(s, label)]) for s in split_names) + " |")
    lines.append("| **Total** | " + " | ".join(str(sum(counts[(s, x)] for x in labels)) for s in split_names) + " |")
    test_only = sorted({r.label for r in records if r.split != "train"} - set(labels_of(records)))
    if test_only:
        lines += ["", f"**Classes without training images:** {', '.join(test_only)}"]

    lines += ["", "## Image sizes per class (width x height: count)", "", "| Split | Class | Sizes |", "|---|---|---|"]
    for (split, label) in sorted(counts):
        sizes = Counter(f"{r.width}x{r.height}" for r in records if r.split == split and r.label == label)
        lines.append(f"| {split} | {label} | " + ", ".join(f"{s}: {n}" for s, n in sizes.most_common()) + " |")

    by_size = defaultdict(Counter)
    for r in records:
        by_size[(r.width, r.height)][r.label] += 1
    size_rule = sum(c.most_common(1)[0][1] for c in by_size.values())
    majority = Counter(r.label for r in records).most_common(1)[0][1] if records else 0
    lines += ["", "## Shortcut check: does image size alone give the class away?", "",
              f"Guessing the most common class for each image size is right for {size_rule} of {len(records)} images; "
              f"always guessing the largest class is right for {majority}. A large gap means size is a shortcut."]

    by_hash = defaultdict(list)
    for r in records:
        by_hash[r.sha256].append(r)
    dupes = [g for g in by_hash.values() if len(g) > 1]
    lines += ["", f"## Exact duplicate files: {len(dupes)} groups", ""]
    for g in dupes:
        cross = " (ACROSS SPLITS)" if len({r.split for r in g}) > 1 else ""
        lines.append("- " + ", ".join(f"{r.split}/{r.label}/{Path(r.path).name}" for r in g) + cross)

    hashes = [(r, int(r.dhash, 16)) for r in records]
    near = [((a[1] ^ b[1]).bit_count(), a[0], b[0]) for a, b in itertools.combinations(hashes, 2)
            if (a[1] ^ b[1]).bit_count() <= NEAR_DUPLICATE_DISTANCE and a[0].sha256 != b[0].sha256]
    near.sort(key=lambda x: (x[1].split == x[2].split, x[0], x[1].path, x[2].path))  # pairs across splits first
    lines += ["", f"## Near-duplicate candidates (difference hash distance <= {NEAR_DUPLICATE_DISTANCE} of 256): {len(near)} pairs", "",
              "Candidates only: plain, low-contrast images can look similar to this check. Confirm by eye.", ""]
    for dist, a, b in near[:50]:
        lines.append(f"- {dist}: {a.split}/{a.label}/{Path(a.path).name} ~ {b.split}/{b.label}/{Path(b.path).name}")

    lines += ["", f"## Files that are not readable images: {len(problems)}", ""]
    lines += [f"- `{p}`: {why}" for p, why in problems[:200]]
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "profile.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return report

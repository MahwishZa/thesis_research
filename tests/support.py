"""A tiny synthetic project for tests: coloured squares as images, a manifest, corrections, a prompt and a config (no real data)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from src import dataset

COLOURS = {"melanoma": (120, 40, 30), "nevus": (200, 150, 120)}

CONFIG = """name = "fake"
[data]
root = "{root}"
train = "Train"
test = "Test"
corrections = "{corrections}"
manifest = "{manifest}"
[model]
provider = "fake"
id = "fake-1"
[prompt]
file = "{prompt}"
[run]
requests_per_minute = 600000
dev_per_class = 2
dev_seed = 0
order_seed = 0
max_image_side = 50
"""


def make_dataset(root: Path, per_class: dict[str, dict[str, int]]) -> None:
    """``per_class[split][label]`` images of one colour per label, each with a small varying mark."""
    number = 0
    for split, labels in per_class.items():
        for label, n in labels.items():
            folder = root / split / label
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                image = Image.new("RGB", (60, 45), COLOURS[label])
                image.putpixel((i % 60, (i * 7) % 45), (0, 0, 0))
                image.paste((255, 255, 255), (0, 0, 5 + i, 5 + i))
                if split == "Test":
                    image.paste((10, 200, 10), (50, 35, 59, 44))  # a test image is never a copy of a training image
                number += 1
                image.save(folder / f"ISIC_{number:07d}.jpg", quality=95)


def write_corrections(path: Path, *rows: tuple[str, str, str]) -> None:
    """The corrections file: (path, folder label, corrected label) per row."""
    lines = ["path,isic_id,folder_label,corrected_label,isic_fields"]
    lines += [f"{p},{Path(p).stem},{folder},{corrected},diagnosis={corrected}" for p, folder, corrected in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_project(base: Path) -> SimpleNamespace:
    """A dataset (3+3 training, 2+2 testing images), its manifest, an empty corrections file, a prompt and a config."""
    make_dataset(base / "data", {"Train": {"melanoma": 3, "nevus": 3}, "Test": {"melanoma": 2, "nevus": 2}})
    records, _ = dataset.scan(base / "data", {"train": "Train", "test": "Test"})
    manifest = base / "manifest.csv"
    dataset.write_manifest(records, manifest)
    prompt = base / "prompt.txt"
    prompt.write_text("Pick one:\n{labels}\nAnswer with the name only.", encoding="utf-8")
    corrections = base / "label_corrections.csv"
    write_corrections(corrections)
    config = base / "fake.toml"
    config.write_text(CONFIG.format(root=(base / "data").as_posix(), prompt=prompt.as_posix(), manifest=manifest.as_posix(),
                                    corrections=corrections.as_posix()), encoding="utf-8")
    return SimpleNamespace(base=base, data=base / "data", manifest=manifest, prompt=prompt, corrections=corrections,
                           config=config, results=base / "results")

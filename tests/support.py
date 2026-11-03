"""A tiny synthetic image dataset for tests (coloured squares; no real data)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

COLOURS = {"melanoma": (120, 40, 30), "nevus": (200, 150, 120)}


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
                number += 1
                image.save(folder / f"ISIC_{number:07d}.jpg", quality=95)

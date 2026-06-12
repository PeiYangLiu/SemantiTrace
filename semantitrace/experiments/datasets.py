from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from semantitrace.utils.image import SUPPORTED_IMAGE_EXTS, list_images


@dataclass
class ImageRecord:
    image_id: str
    image_path: str
    metadata: dict[str, Any]


@dataclass
class ImageCorpus:
    name: str
    records: list[ImageRecord]

    @property
    def image_paths(self) -> list[str]:
        return [record.image_path for record in self.records]

    @property
    def image_ids(self) -> list[str]:
        return [record.image_id for record in self.records]


def load_corpus(name: str, image_dir: str | None, manifest: str | None = None, max_images: int | None = None) -> ImageCorpus:
    if not image_dir:
        raise FileNotFoundError(
            f"Dataset {name} has no image_dir configured. Set datasets.{name}.image_dir "
            "in configs/main_experiment.yaml or pass --dry_run_sample."
        )
    root = Path(image_dir).expanduser()
    if manifest:
        records = _load_manifest(name, root, Path(manifest).expanduser())
    else:
        records = [
            ImageRecord(path.stem, str(path), {"source": "directory"})
            for path in list_images(root)
        ]
    if max_images:
        records = records[:max_images]
    if not records:
        raise FileNotFoundError(f"No images found for dataset {name} under {root}")
    return ImageCorpus(name=name, records=records)


def _load_manifest(name: str, image_root: Path, manifest_path: Path) -> list[ImageRecord]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found for dataset {name}: {manifest_path}")
    suffix = manifest_path.suffix.lower()
    if suffix == ".jsonl":
        items = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif suffix == ".json":
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else data.get("data", data.get("items", []))
    elif suffix == ".csv":
        with manifest_path.open("r", encoding="utf-8", newline="") as fh:
            items = list(csv.DictReader(fh))
    else:
        raise ValueError(f"Unsupported manifest format: {manifest_path}")

    records: list[ImageRecord] = []
    for idx, item in enumerate(items):
        rel = item.get("image_path") or item.get("path") or item.get("image") or item.get("file_name")
        if not rel:
            continue
        path = Path(rel)
        if not path.is_absolute():
            candidates = [
                manifest_path.parent / path,
                image_root / path,
                image_root / path.name,
            ]
            path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
        if path.suffix.lower() not in SUPPORTED_IMAGE_EXTS or not path.is_file():
            continue
        image_id = str(item.get("id") or item.get("image_id") or path.stem or idx)
        records.append(ImageRecord(image_id=image_id, image_path=str(path), metadata=dict(item)))
    return records


def create_synthetic_corpus(output_dir: str | Path, name: str = "DryRun", count: int = 12, seed: int = 42) -> ImageCorpus:
    root = Path(output_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    records: list[ImageRecord] = []
    for idx in range(count):
        color = tuple(rng.randint(40, 230) for _ in range(3))
        image = Image.new("RGB", (192, 192), color)
        draw = ImageDraw.Draw(image)
        draw.rectangle((42, 42, 150, 150), outline=(255, 255, 255), width=3)
        draw.line((20, 165, 172, 28), fill=(0, 0, 0), width=2)
        path = root / f"{name.lower()}_{idx:03d}.png"
        image.save(path)
        records.append(ImageRecord(image_id=path.stem, image_path=str(path), metadata={"synthetic": True}))
    return ImageCorpus(name=name, records=records)

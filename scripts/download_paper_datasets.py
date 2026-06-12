#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantitrace.utils.image import SUPPORTED_IMAGE_EXTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Download/export MMQA and WebQA image corpora for SemantiTrace experiments")
    parser.add_argument("--output_root", default="data", help="Output root containing mmqa/ and webqa/")
    parser.add_argument("--mmqa_limit", type=int, default=500, help="0 means extract all MMQA images")
    parser.add_argument("--webqa_limit", type=int, default=500, help="0 means export all WebQA images")
    parser.add_argument("--skip_mmqa", action="store_true")
    parser.add_argument("--skip_webqa", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite existing images/manifests")
    return parser.parse_args()


def export_mmqa(output_root: Path, limit: int, force: bool) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub first") from exc

    out = output_root / "mmqa"
    image_dir = out / "images"
    manifest_path = out / "manifest.json"
    image_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists() and not force:
        print(f"[MMQA] existing manifest found, skipping: {manifest_path}")
        return manifest_path

    zip_path = hf_hub_download("BiXie/multimodalqa", "MMQA.zip", repo_type="dataset")
    metadata_by_path: dict[str, dict[str, Any]] = {}
    manifest: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith("MMQA_images.jsonl"):
                with zf.open(name) as fh:
                    for raw in fh:
                        item = json.loads(raw.decode("utf-8"))
                        metadata_by_path[str(item.get("path", ""))] = item
        names = [
            name for name in zf.namelist()
            if _is_real_image_member(name)
        ]
        if names:
            manifest = _export_zip_images(zf, names, image_dir, out, limit, force, metadata_by_path)
        else:
            nested = next((name for name in zf.namelist() if name.endswith(".zip")), None)
            if nested is None:
                raise RuntimeError("No image files or nested image zip found in MMQA.zip")
            cache_dir = out / "_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            nested_path = cache_dir / Path(nested).name
            if force or not nested_path.exists():
                print(f"[MMQA] extracting nested image archive: {nested}")
                with zf.open(nested) as src, nested_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024 * 16)
            with zipfile.ZipFile(nested_path) as inner:
                inner_names = [
                    name for name in inner.namelist()
                    if _is_real_image_member(name)
                ]
                manifest = _export_zip_images(inner, inner_names, image_dir, out, limit, force, metadata_by_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[MMQA] wrote {len(manifest)} images -> {image_dir}")
    return manifest_path


def _export_zip_images(
    zf: zipfile.ZipFile,
    names: list[str],
    image_dir: Path,
    dataset_root: Path,
    limit: int,
    force: bool,
    metadata_by_path: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    names.sort()
    if limit:
        names = names[:limit]
    manifest: list[dict[str, Any]] = []
    for idx, name in enumerate(names):
        suffix = Path(name).suffix.lower() or ".jpg"
        image_id = Path(name).stem
        out_path = image_dir / f"{image_id}{suffix}"
        if force or not out_path.exists():
            try:
                with zf.open(name) as src:
                    image = Image.open(src).convert("RGB")
                    image.save(out_path)
            except Exception as exc:
                print(f"[MMQA] skip unreadable image {name}: {exc}")
                continue
        meta = metadata_by_path.get(Path(name).name, {})
        manifest.append(
            {
                "id": str(meta.get("id") or image_id),
                "image_path": str(out_path.relative_to(dataset_root)),
                "archive_path": name,
                "title": meta.get("title", ""),
                "url": meta.get("url", ""),
                "source_dataset": "BiXie/multimodalqa",
            }
        )
        if (idx + 1) % 100 == 0:
            print(f"[MMQA] exported {idx + 1}/{len(names)}")
    return manifest


def _is_real_image_member(name: str) -> bool:
    path = Path(name)
    if name.endswith("/") or "__MACOSX/" in name or path.name.startswith("._"):
        return False
    return path.suffix.lower() in SUPPORTED_IMAGE_EXTS


def export_webqa(output_root: Path, limit: int, force: bool) -> Path:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install datasets first: pip install datasets") from exc

    out = output_root / "webqa"
    image_dir = out / "images"
    manifest_path = out / "manifest.json"
    image_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists() and not force:
        print(f"[WebQA] existing manifest found, skipping: {manifest_path}")
        return manifest_path

    ds = load_dataset("MrZilinXiao/MMEB-eval-WebQA-beir-v2", "corpus", split="test")
    manifest: list[dict[str, Any]] = []
    max_items = len(ds) if not limit else min(limit, len(ds))
    for idx, row in enumerate(ds):
        if limit and idx >= limit:
            break
        image = row.get("img")
        if image is None:
            continue
        image_id = str(row.get("did") or row.get("corpus-id") or idx)
        safe_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in image_id)
        out_path = image_dir / f"{safe_id}.png"
        if force or not out_path.exists():
            image.convert("RGB").save(out_path)
        manifest.append(
            {
                "id": image_id,
                "image_path": str(out_path.relative_to(out)),
                "txt": row.get("txt", ""),
                "corpus_id": row.get("corpus-id", ""),
                "img_path": row.get("img_path", ""),
                "source_dataset": "MrZilinXiao/MMEB-eval-WebQA-beir-v2",
            }
        )
        if (idx + 1) % 100 == 0:
            print(f"[WebQA] exported {idx + 1}/{max_items}")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[WebQA] wrote {len(manifest)} images -> {image_dir}")
    return manifest_path


def main() -> None:
    args = parse_args()
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    if not args.skip_mmqa:
        export_mmqa(root, args.mmqa_limit, args.force)
    if not args.skip_webqa:
        export_webqa(root, args.webqa_limit, args.force)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Export unique images from cached HuggingFace datasets for retrieval-scale stress tests.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_column", default="image")
    parser.add_argument("--id_column", default="image_id")
    parser.add_argument("--limit_unique", type=int, default=0)
    parser.add_argument("--require_ocr", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--max_side", type=int, default=512)
    parser.add_argument("--quality", type=int, default=85)
    return parser.parse_args()


def safe_name(value: Any, fallback: int) -> str:
    text = str(value if value is not None else fallback)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("._")
    return text or str(fallback)


def main() -> None:
    from datasets import load_dataset

    args = parse_args()
    out = Path(args.output_dir)
    image_dir = out / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(args.dataset, split=args.split, trust_remote_code=args.trust_remote_code, streaming=args.streaming)

    manifest_path = out / "manifest.json"
    manifest: list[dict[str, Any]] = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else []
    seen: set[str] = {str(item["id"]) for item in manifest}
    for idx, row in enumerate(ds):
        row = dict(row)
        image_id = safe_name(row.get(args.id_column), idx)
        if image_id in seen:
            continue
        if args.require_ocr and not row.get("ocr_tokens"):
            continue
        image = row.get(args.image_column)
        if image is None or not hasattr(image, "convert"):
            continue
        seen.add(image_id)
        path = image_dir / f"{image_id}.jpg"
        if not path.exists():
            image = image.convert("RGB")
            if args.max_side > 0:
                image.thumbnail((args.max_side, args.max_side))
            image.save(path, quality=args.quality, optimize=True)
        meta = {
            "id": image_id,
            "source_index": idx,
            "image_path": str(path.relative_to(out)),
            "dataset": args.dataset,
            "ocr_token_count": len(row.get("ocr_tokens") or []),
        }
        manifest.append(meta)
        if len(manifest) % 1000 == 0:
            print(f"exported {len(manifest)} unique images", flush=True)
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        if args.limit_unique and len(manifest) >= args.limit_unique:
            break

    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"dataset": args.dataset, "unique_images": len(manifest), "output_dir": str(out)}, indent=2))


if __name__ == "__main__":
    main()

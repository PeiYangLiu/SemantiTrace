#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from semantitrace.records import load_records_with_resolved_paths, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Materialize defended natural Mode-B records.")
    parser.add_argument("--records", required=True)
    parser.add_argument("--record_root", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--defense", choices=["jpeg_resize_q50", "oracle_object_blur", "oracle_object_remove"], required=True)
    parser.add_argument("--max_records", type=int, default=500)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    return resolve_repo_path(path)


def expand_bbox(bbox: list[int], image: Image.Image, pad: int = 6) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return max(0, x1 - pad), max(0, y1 - pad), min(image.width, x2 + pad), min(image.height, y2 + pad)


def bbox_for_record(record: dict[str, Any], image: Image.Image) -> list[int]:
    plan = record.get("nontext_plan") if isinstance(record.get("nontext_plan"), dict) else {}
    bbox = plan.get("bbox")
    if bbox and len(bbox) == 4:
        return [int(v) for v in bbox]
    metrics = record.get("injection_metrics") if isinstance(record.get("injection_metrics"), dict) else {}
    bbox = metrics.get("effective_mask_bbox")
    if bbox and len(bbox) == 4:
        return [int(v) for v in bbox]
    canvas = record.get("selected_canvas") if isinstance(record.get("selected_canvas"), dict) else {}
    bbox = canvas.get("bbox") if canvas else None
    if bbox and len(bbox) == 4:
        return [int(v) for v in bbox]
    return [0, 0, image.width, image.height]


def apply_jpeg_resize(image: Image.Image) -> tuple[Image.Image, dict[str, Any]]:
    original = image.convert("RGB")
    small = original.resize((max(1, original.width // 2), max(1, original.height // 2)), Image.Resampling.LANCZOS)
    restored = small.resize(original.size, Image.Resampling.LANCZOS)
    buf = BytesIO()
    restored.save(buf, format="JPEG", quality=50)
    buf.seek(0)
    return Image.open(buf).convert("RGB"), {"defended_area_ratio": 1.0}


def fill_region_with_median(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    out = image.convert("RGB").copy()
    x1, y1, x2, y2 = box
    region = np.asarray(out)[y1:y2, x1:x2]
    if region.size == 0:
        return out
    color = tuple(int(v) for v in np.median(region.reshape(-1, 3), axis=0))
    draw = ImageDraw.Draw(out)
    draw.rectangle((x1, y1, x2, y2), fill=color)
    return out


def apply_oracle_defense(image: Image.Image, bbox: list[int], defense: str) -> tuple[Image.Image, dict[str, Any]]:
    out = image.convert("RGB").copy()
    box = expand_bbox(bbox, out, pad=6)
    x1, y1, x2, y2 = box
    if defense == "oracle_object_blur":
        crop = out.crop(box).filter(ImageFilter.GaussianBlur(radius=12))
        out.paste(crop, box)
    elif defense == "oracle_object_remove":
        out = fill_region_with_median(out, box)
    else:
        raise ValueError(defense)
    return out, {"defended_area_ratio": ((x2 - x1) * (y2 - y1)) / max(1, image.width * image.height)}


def quality_delta(a: Image.Image, b: Image.Image) -> float:
    arr_a = np.asarray(a.convert("RGB"), dtype=np.float32)
    arr_b = np.asarray(b.convert("RGB").resize(a.size), dtype=np.float32)
    return float(np.mean(np.abs(arr_a - arr_b)) / 255.0)


def main() -> None:
    args = parse_args()
    out = resolve(args.output_dir)
    records_path = out / "canary_records.json"
    if records_path.exists():
        print(f"[resume] defended records already exist: {records_path}", flush=True)
        return
    records = load_records_with_resolved_paths(args.records, args.max_records, record_root=args.record_root)
    clean_dir = out / "clean"
    wm_dir = out / "watermarked"
    clean_dir.mkdir(parents=True, exist_ok=True)
    wm_dir.mkdir(parents=True, exist_ok=True)
    defended_records: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        wm_src = Path(record["_resolved_watermarked_image_path"])
        clean_src = Path(record["_resolved_anchor_image_path"])
        wm = Image.open(wm_src).convert("RGB")
        clean = Image.open(clean_src).convert("RGB")
        bbox = bbox_for_record(record, wm)
        if args.defense == "jpeg_resize_q50":
            defended, stats = apply_jpeg_resize(wm)
        else:
            defended, stats = apply_oracle_defense(wm, bbox, args.defense)
        clean_path = clean_dir / f"{args.defense}-{idx:04d}_{clean_src.name}"
        wm_path = wm_dir / f"{args.defense}-{idx:04d}_{wm_src.name}"
        if not clean_path.exists():
            shutil.copy2(clean_src, clean_path)
        defended.save(wm_path)
        rec = json.loads(json.dumps(record))
        rec["id"] = f"{args.defense}-{idx:04d}"
        rec["source_modeb_id"] = record.get("id")
        rec["defense"] = args.defense
        rec["anchor_image_path"] = str(clean_path)
        rec["watermarked_image_path"] = str(wm_path)
        rec.setdefault("defense_metrics", {})
        rec["defense_metrics"].update(
            {
                **stats,
                "bbox": bbox,
                "quality_delta_vs_watermarked": quality_delta(wm, defended),
                "quality_delta_vs_clean": quality_delta(clean, defended),
            }
        )
        defended_records.append(rec)
        if (idx + 1) % 50 == 0:
            print(f"[{args.defense}] materialized {idx + 1}/{len(records)}", flush=True)
    records_path.write_text(json.dumps(defended_records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"defense": args.defense, "records": len(defended_records), "output": str(records_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()

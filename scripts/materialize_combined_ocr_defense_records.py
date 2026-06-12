#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from semantitrace.records import load_records_with_resolved_paths, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Materialize OCR blur/fill defended combined A+B records.")
    parser.add_argument("--records", required=True)
    parser.add_argument("--record_root", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--defense", choices=["ocr_blur", "ocr_fill"], required=True)
    parser.add_argument("--max_records", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    return resolve_repo_path(path)


def expand_bbox(bbox: list[int], image: Image.Image, pad: int = 4) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return max(0, x1 - pad), max(0, y1 - pad), min(image.width, x2 + pad), min(image.height, y2 + pad)


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


def ocr_boxes(image_path: Path, reader: Any) -> list[dict[str, Any]]:
    result = reader.readtext(str(image_path), detail=1, paragraph=False)
    boxes: list[dict[str, Any]] = []
    for item in result:
        if len(item) < 2:
            continue
        points = item[0]
        text = str(item[1])
        conf = float(item[2]) if len(item) > 2 else 1.0
        xs = [int(p[0]) for p in points]
        ys = [int(p[1]) for p in points]
        boxes.append({"bbox": [min(xs), min(ys), max(xs), max(ys)], "text": text, "conf": conf})
    return boxes


def apply_ocr_defense(image: Image.Image, boxes: list[dict[str, Any]], defense: str) -> tuple[Image.Image, dict[str, Any]]:
    out = image.convert("RGB").copy()
    total_area = 0
    for item in boxes:
        box = expand_bbox(item["bbox"], out)
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            continue
        total_area += (x2 - x1) * (y2 - y1)
        if defense == "ocr_blur":
            crop = out.crop(box).filter(ImageFilter.GaussianBlur(radius=8))
            out.paste(crop, box)
        elif defense == "ocr_fill":
            out = fill_region_with_median(out, box)
        else:
            raise ValueError(defense)
    return out, {"ocr_boxes": len(boxes), "defended_area_ratio": total_area / max(1, image.width * image.height)}


def quality_delta(a: Image.Image, b: Image.Image) -> float:
    arr_a = np.asarray(a.convert("RGB"), dtype=np.float32)
    arr_b = np.asarray(b.convert("RGB").resize(a.size), dtype=np.float32)
    return float(np.mean(np.abs(arr_a - arr_b)) / 255.0)


def main() -> None:
    args = parse_args()
    out = resolve(args.output_dir)
    records_path = out / "canary_records.json"
    if records_path.exists():
        existing = json.loads(records_path.read_text(encoding="utf-8"))
        if len(existing) >= args.max_records:
            print(f"[resume] defended records already exist: {records_path}", flush=True)
            return
    else:
        existing = []
    import torch
    import easyocr

    reader = easyocr.Reader(["en"], gpu=(args.device != "cpu" and torch.cuda.is_available()))
    records = load_records_with_resolved_paths(args.records, args.max_records, record_root=args.record_root)
    clean_dir = out / "clean"
    wm_dir = out / "watermarked"
    clean_dir.mkdir(parents=True, exist_ok=True)
    wm_dir.mkdir(parents=True, exist_ok=True)
    defended_records: list[dict[str, Any]] = list(existing)
    start = len(defended_records)
    for idx, record in enumerate(records[start:], start=start):
        wm_src = Path(record["_resolved_watermarked_image_path"])
        clean_src = Path(record["_resolved_anchor_image_path"])
        wm = Image.open(wm_src).convert("RGB")
        clean = Image.open(clean_src).convert("RGB")
        boxes = ocr_boxes(wm_src, reader)
        defended, stats = apply_ocr_defense(wm, boxes, args.defense)
        clean_path = clean_dir / f"{args.defense}-{idx:04d}_{clean_src.name}"
        wm_path = wm_dir / f"{args.defense}-{idx:04d}_{wm_src.name}"
        if not clean_path.exists():
            shutil.copy2(clean_src, clean_path)
        defended.save(wm_path)
        rec = json.loads(json.dumps(record))
        rec["id"] = f"{args.defense}-{idx:04d}"
        rec["source_combined_id"] = record.get("id")
        rec["defense"] = args.defense
        rec["anchor_image_path"] = str(clean_path)
        rec["watermarked_image_path"] = str(wm_path)
        rec.setdefault("defense_metrics", {})
        rec["defense_metrics"].update(
            {
                **stats,
                "quality_delta_vs_watermarked": quality_delta(wm, defended),
                "quality_delta_vs_clean": quality_delta(clean, defended),
            }
        )
        defended_records.append(rec)
        if (idx + 1) % 25 == 0:
            records_path.write_text(json.dumps(defended_records, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[{args.defense}] materialized {idx + 1}/{len(records)}", flush=True)
    records_path.write_text(json.dumps(defended_records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"defense": args.defense, "records": len(defended_records), "output": str(records_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()

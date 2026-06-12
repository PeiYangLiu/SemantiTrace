#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from semantitrace.experiments.transforms import apply_transform
from semantitrace.records import load_records_with_resolved_paths, resolve_repo_path


OPS: dict[str, dict[str, Any]] = {
    "jpeg_q75": {"type": "jpeg", "quality": 75},
    "rescale_0_5": {"type": "rescale", "scale": 0.5},
    "gaussian_sigma5": {"type": "gaussian_noise", "sigma": 5.0, "seed": 42},
    "center_crop_10": {"type": "center_crop", "fraction": 0.1},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Materialize DIP-transformed combined A+B records.")
    parser.add_argument("--records", required=True)
    parser.add_argument("--record_root", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--operation", choices=sorted(OPS), required=True)
    parser.add_argument("--max_records", type=int, default=250)
    return parser.parse_args()


def quality_delta(a: Image.Image, b: Image.Image) -> float:
    arr_a = np.asarray(a.convert("RGB"), dtype=np.float32)
    arr_b = np.asarray(b.convert("RGB").resize(a.size), dtype=np.float32)
    return float(np.mean(np.abs(arr_a - arr_b)) / 255.0)


def main() -> None:
    args = parse_args()
    out = resolve_repo_path(args.output_dir)
    records_path = out / "canary_records.json"
    if records_path.exists():
        existing = json.loads(records_path.read_text(encoding="utf-8"))
        if len(existing) >= args.max_records:
            print(f"[resume] transformed records already exist: {records_path}", flush=True)
            return
    else:
        existing = []

    records = load_records_with_resolved_paths(args.records, args.max_records, record_root=args.record_root)
    clean_dir = out / "clean"
    wm_dir = out / "watermarked"
    clean_dir.mkdir(parents=True, exist_ok=True)
    wm_dir.mkdir(parents=True, exist_ok=True)
    transformed_records: list[dict[str, Any]] = list(existing)
    start = len(transformed_records)
    spec = OPS[args.operation]

    for idx, record in enumerate(records[start:], start=start):
        wm_src = Path(record["_resolved_watermarked_image_path"])
        clean_src = Path(record["_resolved_anchor_image_path"])
        wm = Image.open(wm_src).convert("RGB")
        clean = Image.open(clean_src).convert("RGB")
        transformed = apply_transform(wm, spec)
        clean_path = clean_dir / f"{args.operation}-{idx:04d}_{clean_src.name}"
        wm_path = wm_dir / f"{args.operation}-{idx:04d}_{wm_src.name}"
        if not clean_path.exists():
            shutil.copy2(clean_src, clean_path)
        transformed.save(wm_path)

        rec = json.loads(json.dumps(record))
        rec["id"] = f"{args.operation}-{idx:04d}"
        rec["source_combined_id"] = record.get("id")
        rec["defense"] = args.operation
        rec["dip_operation"] = spec
        rec["anchor_image_path"] = str(clean_path)
        rec["watermarked_image_path"] = str(wm_path)
        rec.setdefault("defense_metrics", {})
        rec["defense_metrics"].update(
            {
                "quality_delta_vs_watermarked": quality_delta(wm, transformed),
                "quality_delta_vs_clean": quality_delta(clean, transformed),
            }
        )
        transformed_records.append(rec)
        if (idx + 1) % 25 == 0:
            records_path.write_text(json.dumps(transformed_records, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[{args.operation}] materialized {idx + 1}/{len(records)}", flush=True)

    records_path.write_text(json.dumps(transformed_records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"operation": args.operation, "records": len(transformed_records), "output": str(records_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()

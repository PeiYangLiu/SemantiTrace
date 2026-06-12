#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantitrace.backends.real import QwenVLMClient
from semantitrace.baselines import AQUABaseline
from semantitrace.defenses import ANOMALY_FILTER_PROMPT


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run artifact-backed VLM rejection checks for Table XVII")
    parser.add_argument("--records", default="n_scaling_subsets_20260609/combined_a125_b125_n250.json")
    parser.add_argument("--image_root", default="amlt_combined_a500_b500_records")
    parser.add_argument("--output_dir", default="outputs/table_xvii_vlm_rejection_n250")
    parser.add_argument("--qwen_model", default="amlt_model_snapshots/qwen")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=["mode_a", "all"], default="mode_a")
    parser.add_argument("--max_records", type=int, default=125)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    return parser.parse_args()


def resolve(path: str | Path, roots: list[Path] | None = None) -> Path:
    p = Path(path)
    if p.is_absolute() and p.exists():
        return p
    candidates = [ROOT / p]
    for root in roots or []:
        candidates.append(root / p)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(str(path))


def is_mode_a(record: dict[str, Any]) -> bool:
    return "text" in str(record.get("parasitism_mode", "")).lower() or str(record.get("id", "")).startswith("modea")


def load_records(path: Path, image_root: Path, mode: str, max_records: int) -> list[dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if mode == "mode_a":
        records = [record for record in records if is_mode_a(record)]
    if max_records:
        records = records[:max_records]
    for record in records:
        record["_clean_path"] = str(resolve(record["anchor_image_path"], [image_root, path.parent]))
        record["_watermarked_path"] = str(resolve(record["watermarked_image_path"], [image_root, path.parent]))
    return records


def crop_bbox(record: dict[str, Any], image: Image.Image, pad_ratio: float = 0.5) -> tuple[int, int, int, int]:
    metrics = record.get("injection_metrics") if isinstance(record.get("injection_metrics"), dict) else {}
    bbox = metrics.get("effective_mask_bbox") or (record.get("selected_canvas") or {}).get("bbox")
    if not bbox or len(bbox) != 4:
        return (0, 0, image.width, image.height)
    x1, y1, x2, y2 = [int(v) for v in bbox]
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    pad = int(max(w, h) * pad_ratio)
    return (
        max(0, x1 - pad),
        max(0, y1 - pad),
        min(image.width, x2 + pad),
        min(image.height, y2 + pad),
    )


def center_crop(image: Image.Image, fraction: float = 0.65) -> Image.Image:
    w, h = image.size
    cw, ch = int(w * fraction), int(h * fraction)
    x1 = max(0, (w - cw) // 2)
    y1 = max(0, (h - ch) // 2)
    return image.crop((x1, y1, x1 + cw, y1 + ch))


def verdict(response: str) -> bool:
    text = response.upper()
    if "[VERDICT: ANOMALY]" in text:
        return True
    if "[VERDICT: SAFE]" in text:
        return False
    if re.search(r"\bANOMAL(Y|OUS|IES)\b", text) and not re.search(r"\bSAFE\b", text):
        return True
    return False


def inspect_prompt(view: str) -> str:
    return (
        ANOMALY_FILTER_PROMPT
        + "\n\n"
        + f"You are inspecting the {view} view. Do not reject ordinary natural scene text, logos, "
        + "or normal objects. Reject only if the image appears synthetic, globally anomalous, "
        + "watermarked, locally edited in an implausible way, or contains rare standalone acronyms "
        + "with no plausible scene context. A flat synthetic image with a centered rare acronym, "
        + "a bizarre object arrangement, or an obvious generated watermark should be rejected even "
        + "if the pixels are clean. A natural photo with plausible typography should be safe unless "
        + "the inspected crop exposes a clear edit artifact or context-free rare code. Answer with "
        + "exactly one verdict line first."
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = resolve(args.records)
    image_root = resolve(args.image_root)
    qwen_model = str(resolve(args.qwen_model))
    records = load_records(records_path, image_root, args.mode, args.max_records)
    details_path = out_dir / "table_xvii_vlm_rejection_details.jsonl"

    seen: set[tuple[str, str, str]] = set()
    details: list[dict[str, Any]] = []
    if details_path.exists():
        with details_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                details.append(item)
                seen.add((item["record_id"], item["method"], item["view"]))

    print(f"Loading Qwen model from {qwen_model}", flush=True)
    vlm = QwenVLMClient(model_name=qwen_model, device=args.device)

    with details_path.open("a", encoding="utf-8") as out:
        for idx, record in enumerate(records):
            clean = Image.open(record["_clean_path"]).convert("RGB")
            sem = Image.open(record["_watermarked_path"]).convert("RGB")
            bbox = crop_bbox(record, sem)
            signature = str(record.get("trap_signature") or f"CANARY{idx:04d}")
            aqua = AQUABaseline(seed=42 + idx).acronym(signature[:24]).image
            cases = [
                ("clean", "global", clean),
                ("clean", "local_crop", clean.crop(bbox)),
                ("semantitrace", "global", sem),
                ("semantitrace", "local_crop", sem.crop(bbox)),
                ("aqua_acronym", "global", aqua),
                ("aqua_acronym", "local_crop", center_crop(aqua)),
            ]
            for method, view, image in cases:
                key = (str(record["id"]), method, view)
                if key in seen:
                    continue
                response = vlm.generate(image, inspect_prompt(view), temperature=0.0, max_new_tokens=args.max_new_tokens)
                item = {
                    "record_id": str(record["id"]),
                    "source_semantitrace_id": record.get("source_semantitrace_id"),
                    "method": method,
                    "view": view,
                    "signature": signature,
                    "bbox": list(bbox),
                    "rejected": verdict(response),
                    "response": response,
                }
                out.write(json.dumps(item, ensure_ascii=False) + "\n")
                out.flush()
                details.append(item)
                seen.add(key)
                print(f"[{len(details):04d}] {record['id']} {method}/{view} rejected={item['rejected']}", flush=True)

    summary_rows: list[dict[str, Any]] = []
    for method in ["clean", "aqua_acronym", "semantitrace"]:
        for view in ["global", "local_crop"]:
            subset = [item for item in details if item["method"] == method and item["view"] == view]
            if not subset:
                continue
            summary_rows.append(
                {
                    "method": method,
                    "view": view,
                    "num_images": len(subset),
                    "vlm_rejection_rate": float(np.mean([bool(item["rejected"]) for item in subset])),
                }
            )
    (out_dir / "table_xvii_vlm_rejection_summary.json").write_text(
        json.dumps({"records": str(records_path), "mode": args.mode, "summary": summary_rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(out_dir / "table_xvii_vlm_rejection_summary.csv", summary_rows)
    print(json.dumps(summary_rows, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

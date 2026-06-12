#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_cached_100k_e2e_profile import encode_image_paths
from run_end_to_end_profiles import build_fallback_index, resolve_anchor_path
from run_large_clip_retrieval import encode_texts
from run_pipeline_generality import load_records, resolve
from semantitrace.metrics import compute_psnr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run n=500 same-anchor direct text overlay baseline on the 100k retrieval cache.")
    parser.add_argument("--records", default="outputs/semantitrace_n500_textvqa_merged/canary_records_first500_local.json")
    parser.add_argument("--retrieval_cache_dir", default="outputs/pipeline_generality_flux_n500_textvqa_100k_clip")
    parser.add_argument("--output_dir", default="outputs/n500_overlay_baseline_100k")
    parser.add_argument("--max_records", type=int, default=500)
    parser.add_argument("--num_queries", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument(
        "--anchor_fallback_dirs",
        nargs="*",
        default=[
            "data_scene_text/total_text/images",
            "data_scene_text/coco_text/images",
            "data_webqa_5000/webqa/images",
            "data_expanded/mmqa/images",
            "data_expanded/webqa/images",
            "data_textvqa_ocr_shards/shard_0",
            "data_textvqa_ocr_shards/shard_1",
            "data_textvqa_ocr_shards/shard_2",
            "data_textvqa_ocr_shards/shard_3",
            "data/mmqa/images",
            "data/webqa/images",
        ],
    )
    return parser.parse_args()


def font_for_box(box_h: int) -> ImageFont.ImageFont:
    size = max(14, int(box_h * 0.72))
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        p = Path(path)
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def bbox_for_record(record: dict[str, Any], image: Image.Image) -> list[int]:
    canvas = record.get("selected_canvas") if isinstance(record.get("selected_canvas"), dict) else {}
    bbox = canvas.get("bbox") if canvas else None
    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        return [max(0, x1), max(0, y1), min(image.width, x2), min(image.height, y2)]
    w, h = image.size
    return [int(0.68 * w), int(0.82 * h), int(0.96 * w), int(0.93 * h)]


def overlay_signature(image: Image.Image, bbox: list[int], signature: str) -> Image.Image:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    x1, y1, x2, y2 = bbox
    pad = max(2, (y2 - y1) // 10)
    x1, y1, x2, y2 = max(0, x1 - pad), max(0, y1 - pad), min(out.width, x2 + pad), min(out.height, y2 + pad)
    draw.rectangle((x1, y1, x2, y2), fill=(248, 248, 238), outline=(30, 30, 30), width=max(1, (y2 - y1) // 18))
    font = font_for_box(y2 - y1)
    text_bbox = draw.textbbox((0, 0), signature, font=font)
    tw, th = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    tx = x1 + max(2, (x2 - x1 - tw) // 2)
    ty = y1 + max(0, (y2 - y1 - th) // 2)
    draw.text((tx, ty), signature, fill=(0, 0, 0), font=font)
    return out


def local_delta(clean: Image.Image, edited: Image.Image, bbox: list[int]) -> float:
    x1, y1, x2, y2 = bbox
    a = np.asarray(clean.convert("RGB").crop((x1, y1, x2, y2)), dtype=np.float32)
    b = np.asarray(edited.convert("RGB").crop((x1, y1, x2, y2)), dtype=np.float32)
    return float(np.mean(np.abs(a - b)) / 255.0)


def materialize(records: list[dict[str, Any]], out_dir: Path, fallback: dict[str, Path]) -> tuple[list[dict[str, Any]], list[Path]]:
    records_path = out_dir / "overlay_records.json"
    if records_path.exists():
        rows = json.loads(records_path.read_text(encoding="utf-8"))
        return rows, [resolve(row["watermarked_image_path"]) for row in rows]
    img_dir = out_dir / "watermarked"
    clean_dir = out_dir / "clean"
    img_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    out_records: list[dict[str, Any]] = []
    paths: list[Path] = []
    for idx, record in enumerate(records):
        clean_path = resolve_anchor_path(record, fallback)
        clean = Image.open(clean_path).convert("RGB")
        bbox = bbox_for_record(record, clean)
        edited = overlay_signature(clean, bbox, str(record["trap_signature"]))
        wm_path = img_dir / f"overlay-{idx:04d}_{record['trap_signature']}.png"
        cp_path = clean_dir / f"overlay-{idx:04d}_{clean_path.name}"
        if not cp_path.exists():
            clean.save(cp_path)
        edited.save(wm_path)
        rec = {
            "id": f"overlay-{idx:04d}",
            "source_semantitrace_id": record["id"],
            "anchor_image_path": str(cp_path.relative_to(ROOT)),
            "watermarked_image_path": str(wm_path.relative_to(ROOT)),
            "trap_signature": record["trap_signature"],
            "probe_queries": record.get("probe_queries", [])[:3],
            "selected_canvas": {"bbox": bbox, "source": "same_anchor_overlay"},
            "quality": {
                "local_delta": local_delta(clean, edited, bbox),
                "psnr": compute_psnr(np.asarray(clean), np.asarray(edited)),
            },
        }
        out_records.append(rec)
        paths.append(wm_path)
    records_path.write_text(json.dumps(out_records, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_records, paths


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_records = load_records(resolve(args.records), args.max_records)
    fallback = build_fallback_index(args.anchor_fallback_dirs)
    overlay_records, overlay_paths = materialize(source_records, out_dir, fallback)
    overlay_emb = encode_image_paths(overlay_paths, args, out_dir / "overlay_image_embeddings.npy")
    cache = resolve(args.retrieval_cache_dir)
    cache_entries = json.loads((cache / "clip_image_entries.json").read_text(encoding="utf-8"))
    cache_emb = np.load(cache / "clip_image_embeddings.npy", mmap_mode="r")
    distractor_start = next(i for i, entry in enumerate(cache_entries) if entry["role"] == "distractor")
    distractor_entries = cache_entries[distractor_start:]
    distractor_emb = np.asarray(cache_emb[distractor_start:], dtype="float32")
    image_emb = np.vstack([overlay_emb.astype("float32"), distractor_emb])

    query_rows = []
    queries = []
    for idx, record in enumerate(overlay_records):
        for probe_index, query in enumerate(record.get("probe_queries", [])[: args.num_queries]):
            query_rows.append((idx, probe_index, str(query)))
            queries.append(str(query))
    text_emb = encode_texts(queries, args)
    details = []
    ranks = []
    for qi, (idx, probe_index, query) in enumerate(query_rows):
        scores = image_emb @ text_emb[qi]
        target_score = scores[idx]
        rank = int(np.count_nonzero(scores > target_score) + 1)
        ranks.append(rank)
        record = overlay_records[idx]
        details.append(
            {
                "variant": "same_anchor_direct_overlay_100k",
                "record_id": record["id"],
                "source_semantitrace_id": record["source_semantitrace_id"],
                "probe_index": probe_index,
                "signature": record["trap_signature"],
                "query": query,
                "rank": rank,
                "target_retained": True,
            }
        )
    ranks_np = np.asarray(ranks, dtype=float)
    local_delta = np.asarray([r["quality"]["local_delta"] for r in overlay_records], dtype=float)
    summary = {
        "variant": "same_anchor_direct_overlay_100k",
        "num_records": len(overlay_records),
        "num_queries": len(details),
        "index_size": len(overlay_records) + len(distractor_entries),
        "distractors": len(distractor_entries),
        "recall_at_1": float(np.mean(ranks_np <= 1)),
        "recall_at_3": float(np.mean(ranks_np <= 3)),
        "recall_at_10": float(np.mean(ranks_np <= 10)),
        "recall_at_50": float(np.mean(ranks_np <= 50)),
        "recall_at_100": float(np.mean(ranks_np <= 100)),
        "mrr": float(np.mean(1.0 / ranks_np)),
        "mean_rank": float(ranks_np.mean()),
        "median_rank": float(np.median(ranks_np)),
        "avg_local_delta": float(local_delta.mean()),
        "max_local_delta": float(local_delta.max()),
        "notes": "Same-anchor same-bbox direct text overlay baseline over 100k distractors; retrieval-only large-scale baseline.",
    }
    (out_dir / "overlay_retrieval_details.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in details) + "\n",
        encoding="utf-8",
    )
    (out_dir / "overlay_baseline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "overlay_baseline_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

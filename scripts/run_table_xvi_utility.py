#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantitrace.backends.real import HFSigLIPEncoder, OpenCLIPEncoder
from semantitrace.baselines import AQUABaseline


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run artifact-backed utility preservation checks for Table XVI")
    parser.add_argument("--records", default="n_scaling_subsets_20260609/combined_a125_b125_n250.json")
    parser.add_argument("--image_root", default="amlt_combined_a500_b500_records")
    parser.add_argument("--output_dir", default="outputs/table_xvi_utility_n250")
    parser.add_argument("--mmqa_manifest", default="data_expanded/mmqa/manifest.json")
    parser.add_argument("--webqa_manifest", default="data_expanded/webqa/manifest.json")
    parser.add_argument("--max_dataset_images", type=int, default=1000)
    parser.add_argument("--max_records", type=int, default=250)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encoders", default="clip,siglip", help="Comma-separated: clip,siglip")
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


def load_records(path: Path, image_root: Path, max_records: int) -> list[dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    if max_records:
        records = records[:max_records]
    for record in records:
        record["_clean_path"] = str(resolve(record["anchor_image_path"], [image_root, path.parent]))
        record["_watermarked_path"] = str(resolve(record["watermarked_image_path"], [image_root, path.parent]))
    return records


def load_manifest(path: Path, limit: int) -> tuple[list[dict[str, Any]], list[Path], list[str]]:
    rows = json.loads(path.read_text(encoding="utf-8"))[:limit]
    image_paths: list[Path] = []
    queries: list[str] = []
    for row in rows:
        image_paths.append(resolve(row["image_path"], [path.parent]))
        text = str(row.get("txt") or row.get("title") or row.get("id") or "")
        queries.append(text)
    return rows, image_paths, queries


def build_encoder(name: str, device: str):
    if name == "clip":
        return "CLIP-ViT-L-14", OpenCLIPEncoder(model_name="ViT-L-14", pretrained="openai", device=device, batch_size=96)
    if name == "siglip":
        return "SigLIP-SO400M", HFSigLIPEncoder(
            model_name="google/siglip-so400m-patch14-384",
            device=device,
            batch_size=24,
        )
    raise ValueError(f"Unknown encoder {name}")


def paired_bbox(record: dict[str, Any], image: Image.Image) -> tuple[int, int, int, int]:
    metrics = record.get("injection_metrics") if isinstance(record.get("injection_metrics"), dict) else {}
    bbox = metrics.get("effective_mask_bbox") or (record.get("selected_canvas") or {}).get("bbox")
    if not bbox or len(bbox) != 4:
        return (0, 0, image.width, image.height)
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.width, x2), min(image.height, y2)
    if x2 <= x1 or y2 <= y1:
        return (0, 0, image.width, image.height)
    return (x1, y1, x2, y2)


def mean_abs_delta(clean: Image.Image, edited: Image.Image, bbox: tuple[int, int, int, int] | None = None) -> float:
    clean = clean.convert("RGB")
    edited = edited.convert("RGB").resize(clean.size)
    if bbox is not None:
        clean = clean.crop(bbox)
        edited = edited.crop(bbox)
    a = np.asarray(clean, dtype=np.float32)
    b = np.asarray(edited, dtype=np.float32)
    return float(np.mean(np.abs(a - b)) / 255.0)


def off_bbox_delta(clean: Image.Image, edited: Image.Image, bbox: tuple[int, int, int, int]) -> float:
    clean = clean.convert("RGB")
    edited = edited.convert("RGB").resize(clean.size)
    a = np.asarray(clean, dtype=np.float32)
    b = np.asarray(edited, dtype=np.float32)
    mask = np.ones(a.shape[:2], dtype=bool)
    x1, y1, x2, y2 = bbox
    mask[y1:y2, x1:x2] = False
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs(a[mask] - b[mask])) / 255.0)


def retrieval_rows(
    dataset: str,
    encoder_name: str,
    dataset_image_emb: np.ndarray,
    query_emb: np.ndarray,
    contaminant_embs: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = dataset_image_emb.shape[0]
    for status in ["clean", "aqua_spatial", "semantitrace"]:
        if status == "clean":
            image_emb = dataset_image_emb
            contam_count = 0
        else:
            image_emb = np.vstack([dataset_image_emb, contaminant_embs[status]])
            contam_count = contaminant_embs[status].shape[0]
        sims = query_emb @ image_emb.T
        order = np.argsort(-sims, axis=1)
        ranks = np.empty(n, dtype=np.int32)
        top1_contaminant = 0
        for idx in range(n):
            rank = int(np.where(order[idx] == idx)[0][0]) + 1
            ranks[idx] = rank
            if int(order[idx, 0]) >= n:
                top1_contaminant += 1
        rows.append(
            {
                "dataset": dataset,
                "encoder": encoder_name,
                "corpus_status": status,
                "num_queries": n,
                "num_contaminants": contam_count,
                "retrieval_r_at_1": float(np.mean(ranks <= 1)),
                "retrieval_r_at_5": float(np.mean(ranks <= 5)),
                "retrieval_r_at_10": float(np.mean(ranks <= 10)),
                "mrr": float(np.mean(1.0 / ranks)),
                "mean_target_rank": float(np.mean(ranks)),
                "median_target_rank": float(np.median(ranks)),
                "top1_contaminant_rate": top1_contaminant / n,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
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
    records = load_records(records_path, image_root, args.max_records)

    aqua_dir = out_dir / "aqua_spatial"
    aqua_dir.mkdir(parents=True, exist_ok=True)
    aqua_paths: list[Path] = []
    for idx, record in enumerate(records):
        signature = str(record.get("trap_signature") or f"CANARY{idx:04d}")
        path = aqua_dir / f"aqua_spatial-{idx:04d}.png"
        if not path.exists():
            AQUABaseline(seed=42 + idx).spatial(signature[:24]).image.save(path)
        aqua_paths.append(path)

    sem_paths = [Path(r["_watermarked_path"]) for r in records]
    clean_anchor_paths = [Path(r["_clean_path"]) for r in records]

    dataset_specs = [
        ("MMQA", resolve(args.mmqa_manifest)),
        ("WebQA", resolve(args.webqa_manifest)),
    ]
    encoder_keys = [item.strip() for item in args.encoders.split(",") if item.strip()]
    retrieval_summary: list[dict[str, Any]] = []
    preservation_summary: list[dict[str, Any]] = []

    clean_anchor_images = [Image.open(p).convert("RGB") for p in clean_anchor_paths]
    sem_images = [Image.open(p).convert("RGB") for p in sem_paths]
    aqua_images = [Image.open(p).convert("RGB") for p in aqua_paths]

    pixel_rows: list[dict[str, Any]] = []
    for status, images in [("semantitrace", sem_images), ("aqua_spatial", aqua_images)]:
        global_deltas = []
        local_deltas = []
        off_deltas = []
        for record, clean, edited in zip(records, clean_anchor_images, images, strict=True):
            bbox = paired_bbox(record, clean)
            global_deltas.append(mean_abs_delta(clean, edited))
            local_deltas.append(mean_abs_delta(clean, edited, bbox))
            off_deltas.append(off_bbox_delta(clean, edited, bbox))
        pixel_rows.append(
            {
                "corpus_status": status,
                "num_images": len(records),
                "global_pixel_delta": float(np.mean(global_deltas)),
                "local_pixel_delta": float(np.mean(local_deltas)),
                "off_canvas_pixel_delta": float(np.mean(off_deltas)),
            }
        )

    for encoder_key in encoder_keys:
        encoder_name, encoder = build_encoder(encoder_key, args.device)
        print(f"[{encoder_name}] encoding contaminant/local images", flush=True)
        sem_emb = encoder.encode_images(sem_images)
        aqua_emb = encoder.encode_images(aqua_images)
        clean_anchor_emb = encoder.encode_images(clean_anchor_images)
        for status, emb in [("semantitrace", sem_emb), ("aqua_spatial", aqua_emb)]:
            paired = np.sum(clean_anchor_emb * emb, axis=1)
            row = {
                "encoder": encoder_name,
                "corpus_status": status,
                "num_images": len(records),
                "paired_image_cosine": float(np.mean(paired)),
                "paired_image_cosine_median": float(np.median(paired)),
            }
            row.update(next(p for p in pixel_rows if p["corpus_status"] == status))
            preservation_summary.append(row)

        for dataset_name, manifest_path in dataset_specs:
            _, dataset_paths, queries = load_manifest(manifest_path, args.max_dataset_images)
            print(f"[{encoder_name}] {dataset_name}: encoding {len(dataset_paths)} dataset images/text queries", flush=True)
            dataset_images = [Image.open(p).convert("RGB") for p in dataset_paths]
            dataset_image_emb = encoder.encode_images(dataset_images)
            query_emb = encoder.encode_texts(queries)
            retrieval_summary.extend(
                retrieval_rows(
                    dataset_name,
                    encoder_name,
                    dataset_image_emb,
                    query_emb,
                    {"semantitrace": sem_emb, "aqua_spatial": aqua_emb},
                )
            )

        try:
            import torch

            del encoder
            torch.cuda.empty_cache()
        except Exception:
            pass

    summary = {
        "protocol": {
            "records": str(records_path),
            "num_canaries": len(records),
            "utility_definition": (
                "Global utility is text-to-image retrieval of normal MMQA/WebQA manifest entries after "
                "adding SemantiTrace or AQUA spatial canaries as extra distractors. Local preservation is "
                "paired clean-vs-edited image similarity and pixel change over the n=250 canary hosts."
            ),
        },
        "retrieval_summary": retrieval_summary,
        "preservation_summary": preservation_summary,
    }
    (out_dir / "table_xvi_utility_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(out_dir / "table_xvi_global_retrieval_utility.csv", retrieval_summary)
    write_csv(out_dir / "table_xvi_local_preservation.csv", preservation_summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

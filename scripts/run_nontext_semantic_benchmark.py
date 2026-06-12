#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_adaptive_defense_stress import apply_jpeg_resize, apply_ocr_defense, ocr_boxes
from run_end_to_end_profiles import build_fallback_index, make_montage, resolve_anchor_path
from run_final_rag_verify import build_clients
from run_pipeline_generality import IndexEntry, OpenCLIPScorer, collect_distractors, cosine_scores, load_records, rank_from_scores, resolve
from semantitrace.metrics import compute_psnr, normalize_text


ROOT = Path(__file__).resolve().parents[1]

COLORS: list[tuple[str, tuple[int, int, int]]] = [
    ("magenta", (230, 30, 180)),
    ("cyan", (20, 190, 220)),
    ("lime", (100, 210, 40)),
    ("orange", (240, 140, 20)),
    ("purple", (130, 70, 220)),
    ("blue", (40, 100, 230)),
    ("red", (220, 40, 40)),
    ("yellow", (245, 205, 30)),
]
SHAPES = ["triangle", "diamond", "circle", "star", "hexagon", "square"]
POSITIONS: list[tuple[str, tuple[float, float]]] = [
    ("upper left", (0.12, 0.14)),
    ("upper right", (0.82, 0.14)),
    ("lower left", (0.12, 0.78)),
    ("lower right", (0.82, 0.78)),
    ("center", (0.48, 0.48)),
]

COLOR_ALIASES = {
    "magenta": ["magenta", "pink", "fuchsia"],
    "cyan": ["cyan", "blue", "turquoise", "teal"],
    "lime": ["lime", "green", "bright green"],
    "orange": ["orange"],
    "purple": ["purple", "violet"],
    "blue": ["blue"],
    "red": ["red"],
    "yellow": ["yellow", "gold"],
}

POSITION_ALIASES = {
    "upper left": ["upper left", "top left", "top-left", "upper-left"],
    "upper right": ["upper right", "top right", "top-right", "upper-right"],
    "lower left": ["lower left", "bottom left", "bottom-left", "lower-left"],
    "lower right": ["lower right", "bottom right", "bottom-right", "lower-right"],
    "center": ["center", "middle", "central"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run calibrated non-text semantic Mode-B benchmark")
    parser.add_argument("--source_records", default="outputs/flux2_klein_gradient_guided_n100_tuned_v1/canary_records.json")
    parser.add_argument("--output_dir", default="outputs/nontext_semantic_modeb_n100")
    parser.add_argument("--distractor_dirs", nargs="*", default=["data/mmqa/images", "data/webqa/images"])
    parser.add_argument("--max_records", type=int, default=100)
    parser.add_argument("--max_distractors", type=int, default=400)
    parser.add_argument("--defenses", default="none,ocr_fill,jpeg_resize_q50")
    parser.add_argument("--config", default="configs/semantitrace_flux2_klein_gradient_guided.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--montage_tile", type=int, default=448)
    parser.add_argument("--calibration_fraction", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--materialize_only", action="store_true")
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


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_scene(record: dict[str, Any]) -> str:
    queries = list(record.get("probe_queries", []))
    if queries:
        q = str(queries[0])
        marker = " In "
        if marker in q and ", what exact" in q:
            return q.split(marker, 1)[1].split(", what exact", 1)[0].strip()
    trigger = str(record.get("trigger_prompt", ""))
    if trigger:
        return trigger
    return "the retrieved scene"


def marker_spec(index: int) -> tuple[str, tuple[int, int, int], str, str, tuple[float, float]]:
    color_name, rgb = COLORS[index % len(COLORS)]
    shape = SHAPES[(index // len(COLORS)) % len(SHAPES)]
    position_name, pos = POSITIONS[(index // (len(COLORS) * len(SHAPES))) % len(POSITIONS)]
    return color_name, rgb, shape, position_name, pos


def quality_delta(clean: Image.Image, edited: Image.Image) -> float:
    a = np.asarray(clean.convert("RGB"), dtype=np.float32)
    b = np.asarray(edited.convert("RGB").resize(clean.size), dtype=np.float32)
    return float(np.mean(np.abs(a - b)) / 255.0)


def safe_psnr(clean: Image.Image, edited: Image.Image) -> float | str:
    value = compute_psnr(np.asarray(clean.convert("RGB")), np.asarray(edited.convert("RGB").resize(clean.size)))
    return "inf" if math.isinf(value) else float(value)


def draw_shape(draw: ImageDraw.ImageDraw, bbox: tuple[int, int, int, int], shape: str, rgb: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if shape == "triangle":
        draw.polygon([(x1 + w // 2, y1 + 5), (x1 + 5, y2 - 5), (x2 - 5, y2 - 5)], fill=rgb, outline=(20, 20, 20))
    elif shape == "diamond":
        draw.polygon([(x1 + w // 2, y1 + 4), (x2 - 4, y1 + h // 2), (x1 + w // 2, y2 - 4), (x1 + 4, y1 + h // 2)], fill=rgb, outline=(20, 20, 20))
    elif shape == "circle":
        draw.ellipse((x1 + 5, y1 + 5, x2 - 5, y2 - 5), fill=rgb, outline=(20, 20, 20), width=max(2, w // 18))
    elif shape == "star":
        cx, cy = x1 + w / 2, y1 + h / 2
        pts = []
        for i in range(10):
            angle = -math.pi / 2 + i * math.pi / 5
            radius = w * (0.43 if i % 2 == 0 else 0.19)
            pts.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
        draw.polygon(pts, fill=rgb, outline=(20, 20, 20))
    elif shape == "hexagon":
        cx, cy = x1 + w / 2, y1 + h / 2
        pts = []
        for i in range(6):
            angle = math.pi / 6 + i * math.pi / 3
            pts.append((cx + w * 0.43 * math.cos(angle), cy + h * 0.43 * math.sin(angle)))
        draw.polygon(pts, fill=rgb, outline=(20, 20, 20))
    else:
        draw.rectangle((x1 + 5, y1 + 5, x2 - 5, y2 - 5), fill=rgb, outline=(20, 20, 20), width=max(2, w // 18))


def draw_marker(
    image: Image.Image,
    color_name: str,
    rgb: tuple[int, int, int],
    shape: str,
    pos: tuple[float, float],
) -> tuple[Image.Image, list[int]]:
    out = image.convert("RGB").copy()
    w, h = out.size
    size = max(54, min(w, h) // 8)
    cx = int(pos[0] * w)
    cy = int(pos[1] * h)
    x1 = max(6, min(w - size - 6, cx - size // 2))
    y1 = max(6, min(h - size - 6, cy - size // 2))
    x2, y2 = x1 + size, y1 + size
    draw = ImageDraw.Draw(out)
    pad = max(5, size // 12)
    draw.rounded_rectangle(
        (x1 - pad, y1 - pad, x2 + pad, y2 + pad),
        radius=max(6, size // 7),
        fill=(245, 245, 238),
        outline=(35, 35, 35),
        width=max(2, size // 22),
    )
    draw_shape(draw, (x1, y1, x2, y2), shape, rgb)
    return out, [x1 - pad, y1 - pad, x2 + pad, y2 + pad]


def materialize_records(source_records: list[dict[str, Any]], out_dir: Path, fallback_index: dict[str, Path] | None = None) -> list[dict[str, Any]]:
    records_path = out_dir / "canary_records.json"
    if records_path.exists():
        return json.loads(records_path.read_text(encoding="utf-8"))
    clean_dir = out_dir / "clean"
    wm_dir = out_dir / "watermarked"
    clean_dir.mkdir(parents=True, exist_ok=True)
    wm_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for idx, source in enumerate(source_records):
        color_name, rgb, shape, position_name, pos = marker_spec(idx)
        signature = f"{position_name} {color_name} {shape}"
        clean_src = resolve_anchor_path(source, fallback_index or {})
        clean = Image.open(clean_src).convert("RGB")
        edited, bbox = draw_marker(clean, color_name, rgb, shape, pos)
        clean_path = clean_dir / f"nontext-{idx:04d}_{clean_src.name}"
        wm_path = wm_dir / f"nontext-{idx:04d}_{position_name.replace(' ', '-')}_{color_name}_{shape}.png"
        shutil.copy2(clean_src, clean_path)
        edited.save(wm_path)
        scene = parse_scene(source)
        queries = [
            f"Find the image described as: {scene}. Inspect it for a small non-text geometric sticker. If one is visible, report its location, color, and shape; otherwise answer NONE.",
            f"Using only the retrieved visual evidence from this scene ({scene}), identify any small colored geometric marker. Report location, color, and shape, or NONE if absent.",
            "Look carefully for a small non-text sticker or geometric marker in the retrieved image. Do not guess. Answer with location, color, and shape, or NONE.",
        ]
        records.append(
            {
                "id": f"nontext-{idx:04d}",
                "source_semantitrace_id": source.get("id"),
                "parasitism_mode": "Object Insertion (Non-text semantic)",
                "anchor_image_path": rel(clean_path),
                "watermarked_image_path": rel(wm_path),
                "trap_signature": signature,
                "nontext_target": {
                    "position": position_name,
                    "color": color_name,
                    "shape": shape,
                },
                "trigger_prompt": f"insert a small {color_name} {shape} sticker at the {position_name} of the image",
                "probe_queries": queries,
                "selected_canvas": {"id": 0, "mode": "nontext_object", "bbox": bbox, "text": None, "source": "nontext_semantic_benchmark"},
                "injection_metrics": {
                    "render_strategy": "nontext_geometric_sticker",
                    "effective_mask_bbox": bbox,
                    "quality_local_delta": quality_delta(clean, edited),
                    "psnr": safe_psnr(clean, edited),
                },
            }
        )
    records_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    return records


def materialize_defense(defense: str, records: list[dict[str, Any]], out_dir: Path, reader: Any | None) -> list[dict[str, Any]]:
    if defense == "none":
        return records
    method_dir = out_dir / defense
    records_path = method_dir / "canary_records.json"
    if records_path.exists():
        return json.loads(records_path.read_text(encoding="utf-8"))
    clean_dir = method_dir / "clean"
    wm_dir = method_dir / "watermarked"
    clean_dir.mkdir(parents=True, exist_ok=True)
    wm_dir.mkdir(parents=True, exist_ok=True)
    ocr_cache = out_dir / "_ocr_boxes_cache.json"
    defended_records: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        wm_src = resolve(record["watermarked_image_path"])
        clean_src = resolve(record["anchor_image_path"])
        image = Image.open(wm_src).convert("RGB")
        clean = Image.open(clean_src).convert("RGB")
        if defense in {"ocr_blur", "ocr_fill"}:
            assert reader is not None
            boxes = ocr_boxes(wm_src, ocr_cache, reader)
            defended, stats = apply_ocr_defense(image, boxes, defense)
        elif defense == "jpeg_resize_q50":
            defended, stats = apply_jpeg_resize(image)
        else:
            raise ValueError(f"Unknown defense: {defense}")
        clean_path = clean_dir / f"{defense}-{idx:04d}_{clean_src.name}"
        wm_path = wm_dir / f"{defense}-{idx:04d}_{wm_src.name}"
        shutil.copy2(clean_src, clean_path)
        defended.save(wm_path)
        rec = json.loads(json.dumps(record))
        rec["id"] = f"{defense}-{idx:04d}"
        rec["source_nontext_id"] = record["id"]
        rec["defense"] = defense
        rec["anchor_image_path"] = rel(clean_path)
        rec["watermarked_image_path"] = rel(wm_path)
        rec["defense_metrics"] = {
            **stats,
            "quality_delta_vs_clean": quality_delta(clean, defended),
            "quality_delta_vs_watermarked": quality_delta(image, defended),
            "psnr_vs_clean": safe_psnr(clean, defended),
        }
        defended_records.append(rec)
        records_path.parent.mkdir(parents=True, exist_ok=True)
        records_path.write_text(json.dumps(defended_records, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[{defense}] materialized {idx + 1}/{len(records)}", flush=True)
    return defended_records


def build_entries(records: list[dict[str, Any]], distractors: list[Path], *, clean: bool) -> list[IndexEntry]:
    entries: list[IndexEntry] = []
    for record in records:
        entries.append(
            IndexEntry(
                image_id=str(record["id"]),
                image_path=resolve(record["anchor_image_path"] if clean else record["watermarked_image_path"]),
                role="clean_control" if clean else "canary",
                record_id=str(record["id"]),
            )
        )
    for i, path in enumerate(distractors):
        entries.append(IndexEntry(image_id=f"distractor-{i:05d}", image_path=path, role="distractor"))
    return entries


def rank_hits(clip: OpenCLIPScorer, image_emb: np.ndarray, entries: list[IndexEntry], query: str, target_id: str, top_k: int) -> tuple[list[dict[str, Any]], int]:
    q_emb = clip.encode_texts([query])[0]
    scores = cosine_scores(image_emb, q_emb)
    order = np.argsort(-scores)
    target_idx = next((i for i, entry in enumerate(entries) if entry.record_id == target_id), -1)
    rank = rank_from_scores(scores, target_idx) if target_idx >= 0 else len(entries) + 1
    hits: list[dict[str, Any]] = []
    for hit_rank, idx in enumerate(order[:top_k], start=1):
        entry = entries[int(idx)]
        hits.append(
            {
                "image_id": entry.image_id,
                "image_path": rel(entry.image_path) if entry.image_path.is_absolute() and entry.image_path.is_relative_to(ROOT) else str(entry.image_path),
                "role": entry.role,
                "record_id": entry.record_id,
                "rank": hit_rank,
                "score": float(scores[int(idx)]),
            }
        )
    return hits, rank


def nontext_prompt(query: str) -> str:
    return (
        "Answer using only the retrieved image panels. The canary is non-textual: it is a small colored geometric sticker, not printed letters. "
        "If no such sticker is visible, answer exactly NONE. If visible, answer with the sticker's location, color, and shape. "
        "Do not infer the answer from the question.\n\n"
        f"User query: {query}"
    )


def is_negated(response: str) -> bool:
    text = normalize_text(response)
    return bool(
        "none" in text.split()
        or "not visible" in text
        or "no sticker" in text
        or "no marker" in text
        or "cannot see" in text
        or "do not see" in text
    )


def detect_nontext(response: str, target: dict[str, str]) -> dict[str, bool]:
    text = normalize_text(response)
    color = str(target["color"])
    shape = normalize_text(target["shape"])
    position = str(target["position"])
    color_hit = any(normalize_text(alias) in text for alias in COLOR_ALIASES.get(color, [color]))
    shape_hit = shape in text
    position_hit = any(
        normalize_text(alias) in text or normalize_text(alias).replace(" ", "") in text.replace(" ", "")
        for alias in POSITION_ALIASES.get(position, [position])
    )
    if is_negated(response):
        color_hit = shape_hit = position_hit = False
    return {
        "color_shape_hit": bool(color_hit and shape_hit),
        "strict_location_color_shape_hit": bool(color_hit and shape_hit and position_hit),
        "color_hit": bool(color_hit),
        "shape_hit": bool(shape_hit),
        "position_hit": bool(position_hit),
    }


def split_name(index: int, total: int, calibration_fraction: float) -> str:
    return "calibration" if index < int(total * calibration_fraction) else "evaluation"


def per_canary_rates(details: list[dict[str, Any]], key: str, total_records: int, split: str | None = None) -> np.ndarray:
    grouped = [[] for _ in range(total_records)]
    for detail in details:
        if split is not None and detail["split"] != split:
            continue
        grouped[int(detail["record_index"])].append(bool(detail[key]))
    rates = [sum(values) / len(values) for values in grouped if values]
    return np.asarray(rates, dtype=np.float64)


def bootstrap_ci(values: np.ndarray, rng: random.Random, trials: int = 2000) -> tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    means = []
    for _ in range(trials):
        sample = [float(values[rng.randrange(values.size)]) for _ in range(values.size)]
        means.append(float(np.mean(sample)))
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(defense: str, details: list[dict[str, Any]], records: list[dict[str, Any]], verifier: Any, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rng = random.Random(args.seed)
    for split in ["calibration", "evaluation", "all"]:
        split_filter = None if split == "all" else split
        suspect = per_canary_rates(details, "watermarked_color_shape_hit", len(records), split_filter)
        clean = per_canary_rates(details, "clean_color_shape_hit", len(records), split_filter)
        strict = per_canary_rates(details, "watermarked_strict_hit", len(records), split_filter)
        clean_strict = per_canary_rates(details, "clean_strict_hit", len(records), split_filter)
        split_details = [d for d in details if split_filter is None or d["split"] == split_filter]
        ranks = np.asarray([int(d["target_rank"]) for d in split_details], dtype=float)
        test = verifier.welch_t_test(suspect, clean) if suspect.size and clean.size else {"p_value": 1.0, "reject_h0": False}
        if not np.isfinite(float(test["p_value"])):
            test = {"p_value": 1.0, "reject_h0": False}
        lo, hi = bootstrap_ci(suspect, rng)
        rows.append(
            {
                "defense": defense,
                "split": split,
                "num_canaries": int(suspect.size),
                "num_queries": len(split_details),
                "semantic_detection_rate": float(suspect.mean()) if suspect.size else 0.0,
                "semantic_detection_ci95_low": lo,
                "semantic_detection_ci95_high": hi,
                "clean_false_positive_rate": float(clean.mean()) if clean.size else 0.0,
                "strict_location_detection_rate": float(strict.mean()) if strict.size else 0.0,
                "strict_clean_false_positive_rate": float(clean_strict.mean()) if clean_strict.size else 0.0,
                "recall_at_1": float(np.mean(ranks <= 1)) if ranks.size else 0.0,
                "recall_at_3": float(np.mean(ranks <= 3)) if ranks.size else 0.0,
                "mean_target_rank": float(ranks.mean()) if ranks.size else 0.0,
                "p_value": test["p_value"],
                "reject_h0": test["reject_h0"],
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_records = load_records(resolve(args.source_records), args.max_records)
    fallback_index = build_fallback_index(args.anchor_fallback_dirs)
    base_records = materialize_records(source_records, out_dir, fallback_index)
    defenses = [d.strip() for d in args.defenses.split(",") if d.strip()]
    reader = None
    if any(d in {"ocr_blur", "ocr_fill"} for d in defenses):
        import easyocr
        import torch

        reader = easyocr.Reader(["en"], gpu=(args.device != "cpu" and torch.cuda.is_available()))
    defense_records = {defense: materialize_defense(defense, base_records, out_dir, reader) for defense in defenses}
    if args.materialize_only:
        return

    exclude = {resolve(record["watermarked_image_path"]) for record in base_records}
    exclude |= {resolve(record["anchor_image_path"]) for record in base_records}
    distractors = collect_distractors(args.distractor_dirs, args.max_distractors, exclude)
    clip = OpenCLIPScorer(args.device, args.batch_size)
    _, vlm, verifier = build_clients(args.config, args.device)
    verifier.num_probes_per_canary = min(verifier.num_probes_per_canary, 3)
    all_summaries: list[dict[str, Any]] = []

    for defense, records in defense_records.items():
        method_dir = out_dir / defense
        method_dir.mkdir(parents=True, exist_ok=True)
        details_path = method_dir / "nontext_semantic_details.jsonl"
        if args.fresh and details_path.exists():
            details_path.unlink()
        details: list[dict[str, Any]] = []
        if details_path.exists():
            details = [json.loads(line) for line in details_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        suspect_entries = build_entries(records, distractors, clean=False)
        clean_entries = build_entries(records, distractors, clean=True)
        suspect_emb = clip.encode_images([entry.image_path for entry in suspect_entries])
        clean_emb = clip.encode_images([entry.image_path for entry in clean_entries])
        skip = len(details)
        with details_path.open("a", encoding="utf-8") as fh:
            for record_index, record in enumerate(records):
                target = record["nontext_target"]
                queries = list(record.get("probe_queries", []))[: verifier.num_probes_per_canary]
                for probe_index, query in enumerate(queries):
                    flat_index = record_index * verifier.num_probes_per_canary + probe_index
                    if flat_index < skip:
                        continue
                    target_id = str(record["id"])
                    wm_hits, target_rank = rank_hits(clip, suspect_emb, suspect_entries, str(query), target_id, args.top_k)
                    clean_hits, clean_target_rank = rank_hits(clip, clean_emb, clean_entries, str(query), target_id, args.top_k)
                    wm_context_path = make_montage(
                        wm_hits,
                        method_dir / "contexts" / "watermarked" / f"{record_index:04d}_{probe_index}.jpg",
                        args.montage_tile,
                    )
                    clean_context_path = make_montage(
                        clean_hits,
                        method_dir / "contexts" / "clean" / f"{record_index:04d}_{probe_index}.jpg",
                        args.montage_tile,
                    )
                    prompt = nontext_prompt(str(query))
                    watermarked_response = vlm.generate(Image.open(wm_context_path).convert("RGB"), prompt, temperature=0.0, max_new_tokens=args.max_new_tokens)
                    clean_response = vlm.generate(Image.open(clean_context_path).convert("RGB"), prompt, temperature=0.0, max_new_tokens=args.max_new_tokens)
                    wm_detect = detect_nontext(watermarked_response, target)
                    clean_detect = detect_nontext(clean_response, target)
                    detail = {
                        "defense": defense,
                        "split": split_name(record_index, len(records), args.calibration_fraction),
                        "record_index": record_index,
                        "probe_index": probe_index,
                        "id": target_id,
                        "target": target,
                        "signature": record["trap_signature"],
                        "query": str(query),
                        "target_rank": target_rank,
                        "clean_target_rank": clean_target_rank,
                        "watermarked_hits": wm_hits,
                        "clean_hits": clean_hits,
                        "watermarked_context_path": rel(wm_context_path),
                        "clean_context_path": rel(clean_context_path),
                        "watermarked_response": watermarked_response,
                        "clean_response": clean_response,
                        "watermarked_color_shape_hit": wm_detect["color_shape_hit"],
                        "clean_color_shape_hit": clean_detect["color_shape_hit"],
                        "watermarked_strict_hit": wm_detect["strict_location_color_shape_hit"],
                        "clean_strict_hit": clean_detect["strict_location_color_shape_hit"],
                    }
                    fh.write(json.dumps(detail, ensure_ascii=False) + "\n")
                    fh.flush()
                    details.append(detail)
                    print(
                        f"[{defense} {len(details):03d}/{len(records) * verifier.num_probes_per_canary:03d}] "
                        f"{target_id} rank={target_rank} hit={detail['watermarked_color_shape_hit']} clean={detail['clean_color_shape_hit']}",
                        flush=True,
                    )
        report = {"defense": defense, "records": len(records), "details": details, "summary": summarize(defense, details, records, verifier, args)}
        (method_dir / "nontext_semantic_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        all_summaries.extend(report["summary"])

    (out_dir / "nontext_semantic_summary.json").write_text(json.dumps(all_summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out_dir / "nontext_semantic_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_summaries[0].keys()))
        writer.writeheader()
        writer.writerows(all_summaries)
    print(json.dumps(all_summaries, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from semantitrace.experiments.tables import write_csv, write_json
from semantitrace.experiments.transforms import apply_transform
from semantitrace.mode_verification import (
    detail_response_hit,
    detail_target_gated_hit,
    per_canary_rates_from_predicate,
    score_response,
    target_rank_in_topk,
)
from semantitrace.rag import ImageRAGIndex, RetrievalHit
from semantitrace.records import infer_record_mode
from semantitrace.utils.image import l2_normalize
from semantitrace.verification import Verifier


PALETTE: list[tuple[str, tuple[int, int, int]]] = [
    ("red", (218, 55, 55)),
    ("green", (55, 170, 80)),
    ("blue", (60, 90, 220)),
    ("yellow", (220, 180, 45)),
    ("teal", (45, 170, 175)),
    ("orange", (230, 120, 45)),
    ("pink", (220, 85, 160)),
    ("white", (235, 235, 225)),
]

DISTRACTOR_COLORS: list[tuple[int, int, int]] = [
    (35, 35, 35),
    (85, 85, 85),
    (125, 105, 80),
    (90, 70, 110),
    (40, 100, 105),
    (130, 130, 95),
]


class LocalColorEncoder:
    """Tiny deterministic image/text encoder for local protocol smoke tests.

    Images are represented by mean RGB. Queries are represented by the named
    scene color they mention. This is intentionally simple: the goal is to
    exercise the clean-only retrieval and rank-gated scoring protocol without
    requiring CLIP, Qwen, AMLT, GPUs, or external datasets.
    """

    def __init__(self) -> None:
        self.color_vectors = {
            name: np.asarray(rgb, dtype=np.float32) / 255.0 for name, rgb in PALETTE
        }

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        rows = []
        for image in images:
            arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            rows.append(arr.mean(axis=(0, 1)))
        return l2_normalize(np.vstack(rows).astype(np.float32))

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            lowered = text.lower()
            vec = np.asarray([0.2, 0.2, 0.2], dtype=np.float32)
            for name, color_vec in self.color_vectors.items():
                if name in lowered:
                    vec = color_vec
                    break
            rows.append(vec)
        return l2_normalize(np.vstack(rows).astype(np.float32))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Run a local deterministic SemantiTrace clean-only protocol pipeline."
    )
    parser.add_argument("--output_dir", default="outputs/local_cleanonly_pipeline")
    parser.add_argument("--num_mode_a", type=int, default=4)
    parser.add_argument("--num_mode_b", type=int, default=4)
    parser.add_argument("--num_distractors", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--skip_robustness", action="store_true")
    parser.add_argument("--skip_caption_boundary", action="store_true")
    return parser.parse_args()


def signature(idx: int) -> str:
    syllables = ["BAX", "CEM", "DOV", "FIR", "GUL", "HES", "JOM", "KUV"]
    return syllables[idx % len(syllables)]


def draw_scene(path: Path, color: tuple[int, int, int], text: str | None, *, mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (192, 144), color)
    draw = ImageDraw.Draw(image)
    if mode == "mode_a":
        draw.rounded_rectangle((40, 46, 152, 92), radius=5, fill=(245, 245, 230), outline=(15, 15, 15), width=2)
        label_pos = (58, 58)
    else:
        draw.ellipse((68, 34, 124, 90), fill=(245, 245, 230), outline=(15, 15, 15), width=2)
        label_pos = (82, 52)
    if text:
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
        except OSError:
            font = ImageFont.load_default()
        draw.text(label_pos, text, fill=(5, 5, 5), font=font)
    image.save(path)


def build_records(out_dir: Path, num_mode_a: int, num_mode_b: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    total = num_mode_a + num_mode_b
    if total > len(PALETTE):
        raise ValueError(f"Local demo supports at most {len(PALETTE)} canaries; requested {total}")

    for idx in range(num_mode_a):
        color_name, rgb = PALETTE[idx]
        sig = signature(idx)
        record_id = f"modea-{idx:04d}"
        anchor = out_dir / "images" / "anchors" / f"{record_id}.png"
        watermarked = out_dir / "images" / "watermarked" / f"{record_id}.png"
        draw_scene(anchor, rgb, None, mode="mode_a")
        draw_scene(watermarked, rgb, sig, mode="mode_a")
        records.append(
            {
                "id": record_id,
                "parasitism_mode": "Text Mutation",
                "anchor_image_path": str(anchor),
                "watermarked_image_path": str(watermarked),
                "trap_signature": sig,
                "scene_color": color_name,
                "probe_queries": [
                    f"In the {color_name} storefront scene, transcribe the exact text on the sign.",
                    f"What exact word is written on the sign in the {color_name} scene?",
                    f"Read the sign text from the {color_name} storefront image.",
                ],
            }
        )

    for local_idx in range(num_mode_b):
        idx = num_mode_a + local_idx
        color_name, rgb = PALETTE[idx]
        sig = signature(idx)
        record_id = f"modeb-{local_idx:04d}"
        anchor = out_dir / "images" / "anchors" / f"{record_id}.png"
        watermarked = out_dir / "images" / "watermarked" / f"{record_id}.png"
        draw_scene(anchor, rgb, None, mode="mode_b")
        draw_scene(watermarked, rgb, sig, mode="mode_b")
        records.append(
            {
                "id": record_id,
                "parasitism_mode": "Natural Object Insertion",
                "anchor_image_path": str(anchor),
                "watermarked_image_path": str(watermarked),
                "trap_signature": sig,
                "scene_color": color_name,
                "nontext_plan": {
                    "object_class": "badge",
                    "color": "white",
                    "position_region": "center",
                },
                "probe_queries": [
                    f"In the {color_name} tabletop scene, identify the small standalone object.",
                    f"Find the small inserted object in the {color_name} scene.",
                    f"Which image in the {color_name} scene contains the small local object?",
                ],
            }
        )
    return records


def build_distractors(out_dir: Path, count: int) -> list[Path]:
    paths: list[Path] = []
    for idx in range(count):
        rgb = DISTRACTOR_COLORS[idx % len(DISTRACTOR_COLORS)]
        path = out_dir / "images" / "distractors" / f"distractor-{idx:04d}.png"
        draw_scene(path, rgb, None, mode="mode_a" if idx % 2 else "mode_b")
        paths.append(path)
    return paths


def rank_of_target(hits: list[RetrievalHit], target_id: str, absent_rank: int) -> int:
    for hit in hits:
        if hit.image_id == target_id:
            return int(hit.rank)
    return absent_rank


def response_for(record: dict[str, Any], target_in_topk: bool) -> str:
    if not target_in_topk:
        return "No matching protected element is visible."
    if infer_record_mode(record) == "mode_b":
        return "A white badge is visible in the center of the scene."
    return f"The exact text on the sign is {record['trap_signature']}."


def evaluate_visual_profile(
    records: list[dict[str, Any]],
    watermarked_paths: list[Path],
    distractor_paths: list[Path],
    *,
    profile: str,
    label: str,
    top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    encoder = LocalColorEncoder()
    watermarked_ids = [str(record["id"]) for record in records]
    distractor_ids = [f"distractor-{idx:04d}" for idx in range(len(distractor_paths))]
    suspect_index = ImageRAGIndex(encoder).build(
        watermarked_paths + distractor_paths,
        watermarked_ids + distractor_ids,
    )
    clean_index = ImageRAGIndex(encoder).build(distractor_paths, distractor_ids)

    details: list[dict[str, Any]] = []
    for record_index, record in enumerate(records):
        for probe_index, query in enumerate(record["probe_queries"]):
            wm_hits = suspect_index.search(query, top_k)
            clean_hits = clean_index.search(query, top_k)
            target_rank = rank_of_target(wm_hits, str(record["id"]), len(suspect_index.image_paths) + 1)
            clean_target_rank = len(clean_index.image_paths) + 1
            wm_response = response_for(record, target_rank_in_topk(target_rank, top_k))
            clean_response = response_for(record, False)
            wm_score = score_response(wm_response, record)
            clean_score = score_response(clean_response, record)
            mode = infer_record_mode(record)
            modeb_target_hit = mode == "mode_b" and target_rank_in_topk(target_rank, top_k)
            modeb_clean_hit = False
            response_hit = bool(wm_score["hit"])
            clean_response_hit = bool(clean_score["hit"])
            details.append(
                {
                    "profile": profile,
                    "record_index": record_index,
                    "probe_index": probe_index,
                    "id": record["id"],
                    "mode": mode,
                    "query": query,
                    "target_rank": target_rank,
                    "clean_target_rank": clean_target_rank,
                    "watermarked_hits": [hit.__dict__ for hit in wm_hits],
                    "clean_hits": [hit.__dict__ for hit in clean_hits],
                    "watermarked_response": wm_response,
                    "clean_response": clean_response,
                    "watermarked_score": wm_score,
                    "clean_score": clean_score,
                    "watermarked_response_hit": response_hit,
                    "clean_response_hit": clean_response_hit,
                    "watermarked_response_strict_hit": bool(wm_score.get("strict_hit", wm_score["hit"])),
                    "clean_response_strict_hit": bool(clean_score.get("strict_hit", clean_score["hit"])),
                    "watermarked_hit": bool(modeb_target_hit or detail_target_gated_hit({
                        "target_rank": target_rank,
                        "watermarked_response_hit": response_hit,
                    }, "watermarked", top_k)),
                    "clean_hit": bool(modeb_clean_hit),
                    "watermarked_protected_image_hit": bool(target_rank_in_topk(target_rank, top_k)),
                    "clean_protected_image_hit": False,
                }
            )

    rows = summarize_details(details, records, profile=profile, label=label, top_k=top_k)
    return rows, details


def summarize_details(
    details: list[dict[str, Any]],
    records: list[dict[str, Any]],
    *,
    profile: str,
    label: str,
    top_k: int,
) -> list[dict[str, Any]]:
    verifier = Verifier({"num_probes_per_canary": 3})
    groups: dict[str, list[int]] = {"all": list(range(len(records)))}
    for idx, record in enumerate(records):
        groups.setdefault(infer_record_mode(record), []).append(idx)

    rows: list[dict[str, Any]] = []
    for subset, indices in groups.items():
        index_set = set(indices)
        subset_details = [row for row in details if int(row["record_index"]) in index_set]
        if subset == "all":
            suspect = per_canary_rates_from_predicate(
                subset_details,
                lambda row: row.get("watermarked_hit", False),
                len(records),
            )
            clean = per_canary_rates_from_predicate(
                subset_details,
                lambda row: row.get("clean_hit", False),
                len(records),
            )
            signal_name = "composite_cleanonly_protocol"
        elif subset == "mode_b":
            suspect = per_canary_rates_from_predicate(
                subset_details,
                lambda row: row.get("watermarked_protected_image_hit", False),
                len(records),
            )
            clean = per_canary_rates_from_predicate(
                subset_details,
                lambda row: row.get("clean_protected_image_hit", False),
                len(records),
            )
            signal_name = "protected_image_hit"
        else:
            suspect = per_canary_rates_from_predicate(
                subset_details,
                lambda row: detail_target_gated_hit(row, "watermarked", top_k),
                len(records),
            )
            clean = per_canary_rates_from_predicate(
                subset_details,
                lambda row: detail_target_gated_hit(row, "clean", top_k),
                len(records),
            )
            signal_name = "rank_gated_extraction"
        response = per_canary_rates_from_predicate(
            subset_details,
            lambda row: detail_response_hit(row, "watermarked"),
            len(records),
        )
        clean_response = per_canary_rates_from_predicate(
            subset_details,
            lambda row: detail_response_hit(row, "clean"),
            len(records),
        )
        ranks = np.asarray([int(row["target_rank"]) for row in subset_details], dtype=float)
        clean_ranks = np.asarray([int(row["clean_target_rank"]) for row in subset_details], dtype=float)
        test = (
            verifier.welch_t_test(suspect, clean)
            if suspect.size and clean.size == suspect.size
            else {"p_value": 1.0, "reject_h0": False}
        )
        rows.append(
            {
                "profile": profile,
                "label": label,
                "subset": subset,
                "signal_name": signal_name,
                "num_canaries": len(indices),
                "num_queries": len(subset_details),
                "top_k": top_k,
                "audit_signal": float(suspect.mean()) if suspect.size else 0.0,
                "clean_baseline": float(clean.mean()) if clean.size else 0.0,
                "response_cer": float(response.mean()) if response.size else 0.0,
                "clean_response_cer": float(clean_response.mean()) if clean_response.size else 0.0,
                "recall_at_3": float(np.mean(ranks <= top_k)) if ranks.size else 0.0,
                "clean_recall_at_3": float(np.mean(clean_ranks <= top_k)) if clean_ranks.size else 0.0,
                "p_value": test["p_value"],
                "reject_h0": bool(test["reject_h0"]),
            }
        )
    return rows


def transformed_paths(records: list[dict[str, Any]], out_dir: Path, transform: dict[str, Any]) -> list[Path]:
    name = transform["name"]
    paths: list[Path] = []
    for record in records:
        src = Path(record["watermarked_image_path"])
        image = Image.open(src).convert("RGB")
        dst = out_dir / "images" / "robustness" / name / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        apply_transform(image, transform).save(dst)
        paths.append(dst)
    return paths


def caption_boundary_rows(records: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    num_queries = len(records) * 3
    for profile, signal, clean, r_at_3, description in [
        ("caption_only", 0.0, 0.0, 0.0, "Captions omit canary evidence."),
        ("caption_sidecar", 1.0, 0.0, 1.0, "Sidecar metadata stores canary evidence."),
    ]:
        rows.append(
            {
                "profile": profile,
                "label": profile.replace("_", " ").title(),
                "subset": "all",
                "signal_name": "text_context_extraction",
                "num_canaries": len(records),
                "num_queries": num_queries,
                "top_k": top_k,
                "audit_signal": signal,
                "clean_baseline": clean,
                "response_cer": signal,
                "clean_response_cer": clean,
                "recall_at_3": r_at_3,
                "clean_recall_at_3": 0.0,
                "p_value": 0.5 if signal == clean else 0.0,
                "reject_h0": bool(signal != clean),
                "description": description,
            }
        )
    return rows


def write_details(path: Path, details: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in details),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = build_records(out_dir, args.num_mode_a, args.num_mode_b)
    distractors = build_distractors(out_dir, args.num_distractors)
    write_json(out_dir / "local_cleanonly_records.json", records)

    all_rows: list[dict[str, Any]] = []
    all_details: list[dict[str, Any]] = []
    watermarked_paths = [Path(record["watermarked_image_path"]) for record in records]
    rows, details = evaluate_visual_profile(
        records,
        watermarked_paths,
        distractors,
        profile="local_visual_cleanonly",
        label="Local visual clean-only",
        top_k=args.top_k,
    )
    all_rows.extend(rows)
    all_details.extend(details)

    if not args.skip_robustness:
        transforms = [
            {"name": "jpeg_q75", "type": "jpeg", "quality": 75},
            {"name": "rescale_0_5", "type": "rescale", "scale": 0.5},
            {"name": "gaussian_sigma5", "type": "gaussian_noise", "sigma": 5, "seed": 7},
            {"name": "center_crop_10", "type": "center_crop", "fraction": 0.1},
        ]
        for transform in transforms:
            transformed = transformed_paths(records, out_dir, transform)
            rows, details = evaluate_visual_profile(
                records,
                transformed,
                distractors,
                profile=f"local_{transform['name']}_cleanonly",
                label=f"Local {transform['name']} clean-only",
                top_k=args.top_k,
            )
            all_rows.extend(rows)
            all_details.extend(details)

    if not args.skip_caption_boundary:
        all_rows.extend(caption_boundary_rows(records, top_k=args.top_k))

    write_json(out_dir / "local_cleanonly_summary.json", all_rows)
    write_csv(out_dir / "local_cleanonly_summary.csv", all_rows)
    write_details(out_dir / "local_cleanonly_details.jsonl", all_details)
    print(json.dumps(all_rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

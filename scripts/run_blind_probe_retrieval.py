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


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_pipeline_generality import OpenCLIPScorer, build_entries, collect_distractors, cosine_scores, load_records, resolve
from semantitrace.metrics import normalize_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate signature-blind/natural retrieval probes.")
    parser.add_argument("--records", default="outputs/semantitrace_n500_textvqa_merged/canary_records_first500_local.json")
    parser.add_argument("--output_dir", default="outputs/blind_probe_retrieval_n500")
    parser.add_argument(
        "--distractor_dirs",
        nargs="*",
        default=[
            "data_scene_text/coco_text/images",
            "data_scene_text/total_text/images",
            "data_webqa_5000/webqa/images",
            "data_expanded/mmqa/images",
            "data_expanded/webqa/images",
            "data/mmqa/images",
            "data/webqa/images",
        ],
    )
    parser.add_argument("--max_records", type=int, default=500)
    parser.add_argument("--max_distractors", type=int, default=6000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top_ks", default="1,3,10,50")
    parser.add_argument("--ocr_texts", default="outputs/pipeline_generality_flux_n500_textvqa/_ocr_cache/ocr_texts.json")
    parser.add_argument("--fresh", action="store_true", help="Ignore existing blind_probe_details.jsonl")
    return parser.parse_args()


def clean_scene_description(record: dict[str, Any]) -> str:
    query = str(record.get("probe_queries", [""])[0])
    match = re.search(r"\bIn (.*?), what exact text", query)
    scene = match.group(1) if match else query
    scene = re.sub(r"\s*with '[^']+' highlighted as the target for mutation\.?", "", scene)
    scene = re.sub(r"\s*The text '[^']+' is highlighted[^.]*\.", "", scene)
    scene = re.sub(r"\s*indicating the target for mutation\.?", "", scene)
    scene = scene.replace(str(record.get("trap_signature", "")), "")
    scene = re.sub(r"\s+", " ", scene).strip(" ,.")
    return scene or "the protected source image"


def blind_queries(record: dict[str, Any]) -> list[str]:
    scene = clean_scene_description(record)
    canvas = ((record.get("selected_canvas") or {}).get("text") or "local sign/text region")
    return [
        f"Find the image matching this scene: {scene}.",
        f"Find an image of {scene}. Inspect the local sign or text region and transcribe any short mark visible there.",
        f"Which image contains the described visual evidence: {scene}? The relevant local region originally contained {canvas}.",
    ]


def rank_target(scores: np.ndarray, target_idx: int) -> int:
    order = np.argsort(-scores)
    match = np.where(order == target_idx)[0]
    return int(match[0] + 1) if match.size else int(len(scores) + 1)


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(resolve(args.records), args.max_records)
    details_path = out_dir / "blind_probe_details.jsonl"
    detail_rows: list[dict[str, Any]]
    if details_path.exists() and not args.fresh:
        detail_rows = [json.loads(line) for line in details_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        exclude = {resolve(record["watermarked_image_path"]) for record in records}
        distractors = collect_distractors(args.distractor_dirs, args.max_distractors, exclude)
        entries = build_entries(records, distractors)
        target_index = {entry.record_id: idx for idx, entry in enumerate(entries) if entry.record_id}
        scorer = OpenCLIPScorer(args.device, args.batch_size)
        image_emb = scorer.encode_images([entry.image_path for entry in entries])

        detail_rows = []
        for record in records:
            rid = str(record["id"])
            target_idx = target_index[rid]
            for probe_index, query in enumerate(blind_queries(record)):
                q_emb = scorer.encode_texts([query])[0]
                rank = rank_target(cosine_scores(image_emb, q_emb), target_idx)
                detail_rows.append(
                    {
                        "record_id": rid,
                        "probe_index": probe_index,
                        "query": query,
                        "rank": rank,
                        "query_type": "signature_blind_scene",
                    }
                )
        details_path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in detail_rows) + "\n",
            encoding="utf-8",
        )

    top_ks = [int(item) for item in args.top_ks.split(",") if item.strip()]
    ranks = np.asarray([row["rank"] for row in detail_rows], dtype=float)
    record_by_id = {str(record["id"]): record for record in records}
    readable_by_id: dict[str, bool] = {}
    ocr_path = resolve(args.ocr_texts)
    if ocr_path.exists():
        ocr_texts = json.loads(ocr_path.read_text(encoding="utf-8"))
        for record in records:
            wm_path = str(resolve(record["watermarked_image_path"]).resolve())
            readable_by_id[str(record["id"])] = normalize_text(str(record["trap_signature"])) in normalize_text(ocr_texts.get(wm_path, ""))
    summary = {
        "query_type": "signature_blind_scene",
        "num_records": len(records),
        "num_queries": len(detail_rows),
        "index_size": args.max_records + args.max_distractors,
        "contains_trap_signature": False,
        "mean_rank": float(ranks.mean()) if ranks.size else 0.0,
        "median_rank": float(np.median(ranks)) if ranks.size else 0.0,
        "ocr_readable_rate": (sum(readable_by_id.values()) / len(readable_by_id)) if readable_by_id else None,
    }
    for k in top_ks:
        summary[f"recall_at_{k}"] = float(np.mean(ranks <= k)) if ranks.size else 0.0
        if readable_by_id:
            summary[f"extractive_context_at_{k}"] = float(
                np.mean([(row["rank"] <= k) and readable_by_id.get(str(row["record_id"]), False) for row in detail_rows])
            )
    (out_dir / "blind_probe_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out_dir / "blind_probe_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

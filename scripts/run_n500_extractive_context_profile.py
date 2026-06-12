#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from semantitrace.metrics import normalize_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Compute n=500 retrieval+OCR extractive context profiles from saved rank and OCR artifacts."
    )
    parser.add_argument("--records", default="outputs/semantitrace_n500_textvqa_merged/canary_records_first500_local.json")
    parser.add_argument(
        "--ocr_texts",
        default="outputs/pipeline_generality_flux_n500_textvqa/_ocr_cache/ocr_texts.json",
    )
    parser.add_argument(
        "--detail_sets",
        nargs="+",
        default=[
            "n500_900=outputs/pipeline_generality_flux_n500_textvqa/pipeline_generality_details.jsonl",
            "n500_6500=outputs/pipeline_generality_flux_n500_textvqa_large_distractors/pipeline_generality_details.jsonl",
        ],
    )
    parser.add_argument("--output_dir", default="outputs/n500_extractive_context_profile")
    parser.add_argument("--top_ks", default="3,10,50")
    parser.add_argument("--bootstraps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20270605)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def parse_labeled_paths(items: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in items:
        label, raw_path = item.split("=", 1)
        out[label] = resolve(raw_path)
    return out


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * pct
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def bootstrap_ci(values: list[float], *, bootstraps: int, rng: random.Random) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    n = len(values)
    means: list[float] = []
    for _ in range(bootstraps):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    return percentile(means, 0.025), percentile(means, 0.975)


def load_details(path: Path) -> dict[str, dict[str, list[int]]]:
    grouped: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("target_retained", True):
                grouped[str(row["variant"])][str(row["record_id"])].append(int(row["rank"]))
    return {variant: dict(by_record) for variant, by_record in grouped.items()}


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = json.loads(resolve(args.records).read_text(encoding="utf-8"))
    ocr_texts = json.loads(resolve(args.ocr_texts).read_text(encoding="utf-8"))
    top_ks = [int(item) for item in args.top_ks.split(",") if item.strip()]
    rng = random.Random(args.seed)

    record_by_id = {str(record["id"]): record for record in records}
    ocr_hit_by_id: dict[str, bool] = {}
    normalized_ocr_by_id: dict[str, str] = {}
    for record in records:
        rid = str(record["id"])
        wm_path = str(resolve(record["watermarked_image_path"]).resolve())
        norm_text = normalize_text(ocr_texts.get(wm_path, ""))
        normalized_ocr_by_id[rid] = norm_text
        ocr_hit_by_id[rid] = normalize_text(str(record["trap_signature"])) in norm_text

    cross_collisions = 0
    for rid, record in record_by_id.items():
        sig = normalize_text(str(record["trap_signature"]))
        if any(sig and sig in text for other, text in normalized_ocr_by_id.items() if other != rid):
            cross_collisions += 1

    summary_rows: list[dict[str, Any]] = []
    for label, detail_path in parse_labeled_paths(args.detail_sets).items():
        grouped = load_details(detail_path)
        for variant, ranks_by_record in sorted(grouped.items()):
            record_ids = sorted(ranks_by_record)
            readability = [1.0 if ocr_hit_by_id.get(rid, False) else 0.0 for rid in record_ids]
            readability_ci = bootstrap_ci(readability, bootstraps=args.bootstraps, rng=rng)
            for top_k in top_ks:
                per_record: list[float] = []
                for rid in record_ids:
                    ranks = ranks_by_record[rid]
                    readable = ocr_hit_by_id.get(rid, False)
                    per_query = [1.0 if readable and rank <= top_k else 0.0 for rank in ranks]
                    per_record.append(sum(per_query) / len(per_query))
                ci_low, ci_high = bootstrap_ci(per_record, bootstraps=args.bootstraps, rng=rng)
                summary_rows.append(
                    {
                        "index_label": label,
                        "variant": variant,
                        "top_k": top_k,
                        "num_canaries": len(record_ids),
                        "num_queries": sum(len(ranks_by_record[rid]) for rid in record_ids),
                        "ocr_readable_rate": sum(readability) / len(readability) if readability else 0.0,
                        "ocr_readable_ci95_low": readability_ci[0],
                        "ocr_readable_ci95_high": readability_ci[1],
                        "extractive_context_success": sum(per_record) / len(per_record) if per_record else 0.0,
                        "extractive_context_ci95_low": ci_low,
                        "extractive_context_ci95_high": ci_high,
                        "cross_image_signature_collision_rate": cross_collisions / len(records) if records else 0.0,
                        "interpretation": "retrieval plus OCR-readable exact signature; not a VLM generation benchmark",
                    }
                )

    (out_dir / "extractive_context_summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    with (out_dir / "extractive_context_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(json.dumps({"rows": len(summary_rows), "output_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()

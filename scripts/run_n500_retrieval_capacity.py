#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Compute n=500 retrieval-stage capacity bounds from per-query rank details."
    )
    parser.add_argument(
        "--detail_sets",
        nargs="+",
        default=[
            "n500_900=outputs/pipeline_generality_flux_n500_textvqa/pipeline_generality_details.jsonl",
            "n500_6500=outputs/pipeline_generality_flux_n500_textvqa_large_distractors/pipeline_generality_details.jsonl",
        ],
        help="One or more LABEL=DETAIL_JSONL inputs.",
    )
    parser.add_argument(
        "--summary_sets",
        nargs="*",
        default=[
            "n500_900=outputs/pipeline_generality_flux_n500_textvqa/pipeline_generality_summary_with_r50.json",
            "n500_6500=outputs/pipeline_generality_flux_n500_textvqa_large_distractors/pipeline_generality_summary_with_r50.json",
        ],
        help="Optional LABEL=SUMMARY_JSON files used to recover index size and target retention.",
    )
    parser.add_argument("--output_dir", default="outputs/n500_retrieval_capacity")
    parser.add_argument("--top_ks", default="3,10,50")
    parser.add_argument("--bootstraps", type=int, default=5000)
    parser.add_argument("--overlap_trials", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20270604)
    parser.add_argument("--overlap_sizes", default="1,5,10,25,50,100,250,500")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def parse_labeled_paths(items: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected LABEL=PATH, got {item!r}")
        label, raw_path = item.split("=", 1)
        if not label:
            raise ValueError(f"Empty label in {item!r}")
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
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def load_summary_metadata(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = [rows]
    return {str(row["variant"]): row for row in rows}


def load_rank_details(path: Path) -> dict[str, dict[str, list[int]]]:
    grouped: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("target_retained", True):
                continue
            grouped[str(row["variant"])][str(row["record_id"])].append(int(row["rank"]))
    return {variant: dict(by_record) for variant, by_record in grouped.items()}


def bootstrap_mean_ci(
    per_record_values: list[float],
    *,
    bootstraps: int,
    rng: random.Random,
) -> tuple[float, float]:
    if not per_record_values:
        return 0.0, 0.0
    n = len(per_record_values)
    means: list[float] = []
    for _ in range(bootstraps):
        sample_sum = 0.0
        for _ in range(n):
            sample_sum += per_record_values[rng.randrange(n)]
        means.append(sample_sum / n)
    return percentile(means, 0.025), percentile(means, 0.975)


def summarize_variant(
    *,
    label: str,
    variant: str,
    ranks_by_record: dict[str, list[int]],
    top_k: int,
    metadata: dict[str, Any],
    bootstraps: int,
    rng: random.Random,
) -> dict[str, Any]:
    record_ids = sorted(ranks_by_record)
    per_record_means: list[float] = []
    per_record_any: list[float] = []
    per_record_all: list[float] = []
    all_ranks: list[int] = []
    for rid in record_ids:
        ranks = ranks_by_record[rid]
        hits = [1.0 if rank <= top_k else 0.0 for rank in ranks]
        per_record_means.append(sum(hits) / len(hits))
        per_record_any.append(1.0 if any(hits) else 0.0)
        per_record_all.append(1.0 if all(hits) else 0.0)
        all_ranks.extend(ranks)
    ci_low, ci_high = bootstrap_mean_ci(per_record_means, bootstraps=bootstraps, rng=rng)
    return {
        "index_label": label,
        "index_size": int(metadata.get("index_size", 0) or 0),
        "variant": variant,
        "top_k": top_k,
        "num_canaries": len(record_ids),
        "num_queries": len(all_ranks),
        "probe_hit_rate": sum(per_record_means) / len(per_record_means) if per_record_means else 0.0,
        "probe_hit_ci95_low": ci_low,
        "probe_hit_ci95_high": ci_high,
        "canary_any_probe_rate": sum(per_record_any) / len(per_record_any) if per_record_any else 0.0,
        "canary_all_probe_rate": sum(per_record_all) / len(per_record_all) if per_record_all else 0.0,
        "mean_rank": sum(all_ranks) / len(all_ranks) if all_ranks else 0.0,
        "median_rank": float(median(all_ranks)) if all_ranks else 0.0,
        "target_retention": float(metadata.get("target_retention", 1.0)),
        "interpretation": "retrieval-stage upper bound; generation can only reduce this rate",
    }


def overlap_capacity(
    *,
    label: str,
    variant: str,
    ranks_by_record: dict[str, list[int]],
    top_k: int,
    subset_size: int,
    trials: int,
    rng: random.Random,
) -> dict[str, Any]:
    record_ids = sorted(ranks_by_record)
    subset_size = min(subset_size, len(record_ids))
    retrievable = {
        rid: any(rank <= top_k for rank in ranks_by_record[rid])
        for rid in record_ids
    }
    counts: list[int] = []
    for _ in range(trials):
        sample = rng.sample(record_ids, subset_size)
        counts.append(sum(1 for rid in sample if retrievable[rid]))
    return {
        "index_label": label,
        "variant": variant,
        "top_k": top_k,
        "indexed_canaries": subset_size,
        "trials": trials,
        "mean_retrievable_canaries": sum(counts) / len(counts) if counts else 0.0,
        "p_at_least_1_retrievable": sum(c >= 1 for c in counts) / len(counts) if counts else 0.0,
        "p_at_least_5_retrievable": sum(c >= 5 for c in counts) / len(counts) if counts else 0.0,
        "p_at_least_10_retrievable": sum(c >= 10 for c in counts) / len(counts) if counts else 0.0,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    top_ks = [int(item) for item in args.top_ks.split(",") if item.strip()]
    overlap_sizes = [int(item) for item in args.overlap_sizes.split(",") if item.strip()]

    detail_paths = parse_labeled_paths(args.detail_sets)
    summary_paths = parse_labeled_paths(args.summary_sets)

    summary_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []
    for label, detail_path in detail_paths.items():
        grouped = load_rank_details(detail_path)
        metadata_by_variant = load_summary_metadata(summary_paths.get(label, Path("")))
        for variant, ranks_by_record in sorted(grouped.items()):
            metadata = metadata_by_variant.get(variant, {})
            for top_k in top_ks:
                summary_rows.append(
                    summarize_variant(
                        label=label,
                        variant=variant,
                        ranks_by_record=ranks_by_record,
                        top_k=top_k,
                        metadata=metadata,
                        bootstraps=args.bootstraps,
                        rng=rng,
                    )
                )
                for subset_size in overlap_sizes:
                    overlap_rows.append(
                        overlap_capacity(
                            label=label,
                            variant=variant,
                            ranks_by_record=ranks_by_record,
                            top_k=top_k,
                            subset_size=subset_size,
                            trials=args.overlap_trials,
                            rng=rng,
                        )
                    )

    (out_dir / "retrieval_capacity_summary.json").write_text(
        json.dumps(summary_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "overlap_capacity_summary.json").write_text(
        json.dumps(overlap_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(out_dir / "retrieval_capacity_summary.csv", summary_rows)
    write_csv(out_dir / "overlap_capacity_summary.csv", overlap_rows)
    print(json.dumps({"summary_rows": len(summary_rows), "overlap_rows": len(overlap_rows)}, indent=2))


if __name__ == "__main__":
    main()

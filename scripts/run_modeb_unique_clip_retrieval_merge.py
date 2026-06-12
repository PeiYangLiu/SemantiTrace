#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Merge Mode-B unique retrieval shard count outputs into global R@k.")
    p.add_argument("--shards_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--variant", default="modeb_natural_clip_1m_unique")
    p.add_argument("--top_ks", default="1,3,5,10,20,50,100,1000")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    shards_dir = Path(args.shards_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    shard_dirs = sorted(p for p in shards_dir.glob("shard_*") if p.is_dir())
    if not shard_dirs:
        raise FileNotFoundError(f"No shard_* dirs under {shards_dir}")
    counts = None
    canary_competitor_counts = None
    query_entries = None
    summaries = []
    total_distractors = 0
    for shard in shard_dirs:
        c = np.load(shard / "distractor_rank_counts.npy").astype("int64")
        counts = c if counts is None else counts + c
        comp = np.load(shard / "canary_competitor_counts.npy").astype("int64")
        if canary_competitor_counts is None:
            canary_competitor_counts = comp
        elif not np.array_equal(canary_competitor_counts, comp):
            raise ValueError(f"Canary competitor counts differ in {shard}")
        qe = json.loads((shard / "query_entries.json").read_text(encoding="utf-8"))
        if query_entries is None:
            query_entries = qe
        elif query_entries != qe:
            raise ValueError(f"Query entries differ in {shard}")
        summary = json.loads((shard / "shard_summary.json").read_text(encoding="utf-8"))
        summaries.append(summary)
        total_distractors += int(summary["distractors_processed"])
    assert counts is not None and canary_competitor_counts is not None and query_entries is not None
    ranks = counts + canary_competitor_counts + 1
    ranks_np = ranks.astype("float64")
    top_ks = [int(k) for k in args.top_ks.split(",") if k.strip()]
    details = []
    for row, rank in zip(query_entries, ranks.tolist()):
        detail = dict(row)
        detail.update({"variant": args.variant, "rank": int(rank), "target_retained": True})
        details.append(detail)

    total_records = len({row["record_id"] for row in query_entries})
    global_index_size = int(total_records + total_distractors)

    def make_summary(subset: str, indices: np.ndarray) -> dict:
        subset_ranks = ranks_np[indices]
        subset_entries = [query_entries[int(i)] for i in indices.tolist()]
        summary = {
            "variant": args.variant,
            "subset": subset,
            "num_records": len({row["record_id"] for row in subset_entries}),
            "num_queries": len(subset_entries),
            "index_size": global_index_size,
            "distractors": int(total_distractors),
            "target_retention": 1.0,
            "mean_rank": float(subset_ranks.mean()),
            "median_rank": float(np.median(subset_ranks)),
            "mrr": float(np.mean(1.0 / subset_ranks)),
            "shard_summaries": summaries if subset == "all" else [],
        }
        for k in top_ks:
            summary[f"recall_at_{k}"] = float(np.mean(subset_ranks <= k))
        return summary

    all_indices = np.arange(len(query_entries), dtype=np.int64)
    summary_rows = [make_summary("all", all_indices)]
    modes = sorted({str(row.get("mode", "unknown")) for row in query_entries})
    for mode in modes:
        mode_indices = np.asarray(
            [idx for idx, row in enumerate(query_entries) if str(row.get("mode", "unknown")) == mode],
            dtype=np.int64,
        )
        if len(mode_indices):
            summary_rows.append(make_summary(mode, mode_indices))
    (out / "pipeline_generality_details.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in details) + "\n",
        encoding="utf-8",
    )
    (out / "pipeline_generality_summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    with (out / "pipeline_generality_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        fieldnames = list(summary_rows[0].keys())
        for row in summary_rows[1:]:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(json.dumps(summary_rows, indent=2), flush=True)


if __name__ == "__main__":
    main()

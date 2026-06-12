#!/usr/bin/env python
"""Merge per-shard SemantiTrace Mode B natural-object canary records into one suite.

Reads `canary_records.json` from each shard subdirectory under the run root and
writes a deduplicated, re-indexed merged record list plus a summary.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Merge per-shard Mode B natural canary records.")
    parser.add_argument("--shards_dir", required=True,
                        help="Directory containing shard subdirectories (each with canary_records.json).")
    parser.add_argument("--shard_glob", default="shard_*",
                        help="Glob pattern for shard subdirectories.")
    parser.add_argument("--output_dir", required=True,
                        help="Output dir for the merged canary_records.json and summary.")
    parser.add_argument("--target_count", type=int, default=500,
                        help="Total number of canaries the merged suite should contain.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shards_dir = Path(args.shards_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_dirs = sorted(shards_dir.glob(args.shard_glob))
    if not shard_dirs:
        print(f"[error] no shard subdirectories matched {args.shard_glob} under {shards_dir}", flush=True)
        sys.exit(1)

    seen_signatures: set[str] = set()
    merged: list[dict[str, Any]] = []
    shard_summaries: list[dict[str, Any]] = []

    for sd in shard_dirs:
        rec_path = sd / "canary_records.json"
        if not rec_path.exists():
            shard_summaries.append({"shard": sd.name, "accepted": 0, "missing": True})
            continue
        records = json.loads(rec_path.read_text(encoding="utf-8"))
        shard_summaries.append({"shard": sd.name, "accepted": len(records)})
        for rec in records:
            sig = str(rec.get("trap_signature", "")).strip().lower()
            if not sig or sig in seen_signatures:
                continue
            seen_signatures.add(sig)
            new = json.loads(json.dumps(rec))
            new["source_shard"] = sd.name
            new["source_record_id"] = rec.get("id")
            new["id"] = f"nontextmodeb-{len(merged):04d}"
            merged.append(new)

    if len(merged) > args.target_count:
        merged = merged[: args.target_count]

    (out_dir / "canary_records.json").write_text(
        json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (out_dir / "canary_records_first500.json").write_text(
        json.dumps(merged[:500], indent=2, ensure_ascii=False), encoding="utf-8"
    )

    from collections import Counter
    color_dist = Counter(rec["nontext_plan"]["color"] for rec in merged)
    object_dist = Counter(rec["nontext_plan"]["object_class"] for rec in merged)
    region_dist = Counter(rec["nontext_plan"]["position_region"] for rec in merged)
    summary = {
        "num_merged": len(merged),
        "num_unique_signatures": len(seen_signatures),
        "target_count": args.target_count,
        "shards_dir": str(shards_dir),
        "shard_summaries": shard_summaries,
        "color_distribution": dict(color_dist),
        "object_distribution": dict(object_dist),
        "region_distribution": dict(region_dist),
    }
    (out_dir / "merge_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Merge sharded Mode-B anchor mining outputs.

Each anchor-mining shard writes `anchor_records.json` containing accepted
anchors. This script deduplicates by source image path, reindexes records, and
writes a merged `anchor_records.json` usable by
`run_nontext_modeb_natural_generation.py`.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Merge sharded Mode-B anchor records.")
    parser.add_argument("--shards_dir", required=True)
    parser.add_argument("--shard_glob", default="shard_*")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_count", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shards_dir = Path(args.shards_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    shard_summaries: list[dict[str, Any]] = []
    for shard in sorted(shards_dir.glob(args.shard_glob)):
        path = shard / "anchor_records.json"
        if not path.exists() or path.stat().st_size == 0:
            shard_summaries.append({"shard": shard.name, "accepted": 0, "missing": True})
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        shard_summaries.append({"shard": shard.name, "accepted": len(rows)})
        for row in rows:
            source = str(row.get("source_image_path") or row.get("anchor_image_path") or "")
            if not source or source in seen:
                continue
            seen.add(source)
            rec = json.loads(json.dumps(row))
            rec["source_shard"] = shard.name
            rec["source_record_id"] = row.get("id")
            rec["id"] = f"modeb-anchor-{len(merged):04d}"
            merged.append(rec)
            if len(merged) >= args.target_count:
                break
        if len(merged) >= args.target_count:
            break
    (out_dir / "anchor_records.json").write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    score_bins = Counter(int(float(row.get("modeb_anchor_score", 0)) // 10 * 10) for row in merged)
    scene_counter = Counter(str(row.get("modeb_scene_type", ""))[:80] for row in merged)
    summary = {
        "num_merged": len(merged),
        "target_count": args.target_count,
        "num_unique_sources": len(seen),
        "shards_dir": str(shards_dir),
        "shard_summaries": shard_summaries,
        "score_bins": dict(sorted(score_bins.items())),
        "top_scene_types": scene_counter.most_common(30),
    }
    (out_dir / "anchor_merge_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

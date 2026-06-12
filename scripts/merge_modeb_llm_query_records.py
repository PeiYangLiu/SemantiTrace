#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from semantitrace.records import resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Merge sharded Mode-B LLM-query record files.")
    parser.add_argument("--shards_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected_records", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shards_dir = resolve_repo_path(args.shards_dir)
    rows = []
    for path in sorted(shards_dir.glob("shard_*.json")):
        shard_rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(shard_rows, list):
            raise ValueError(f"{path} is not a JSON list")
        rows.extend(shard_rows)
    if not rows:
        raise FileNotFoundError(f"No shard_*.json files found under {shards_dir}")

    seen: set[int] = set()
    for row in rows:
        if "query_materialization_record_index" not in row:
            raise ValueError(f"Missing query_materialization_record_index in {row.get('id')}")
        idx = int(row["query_materialization_record_index"])
        if idx in seen:
            raise ValueError(f"Duplicate record index {idx}")
        seen.add(idx)
    rows.sort(key=lambda item: int(item["query_materialization_record_index"]))
    if args.expected_records and len(rows) != int(args.expected_records):
        raise ValueError(f"Expected {args.expected_records} records, found {len(rows)}")
    expected = set(range(len(rows)))
    if seen != expected:
        missing = sorted(expected - seen)[:20]
        extra = sorted(seen - expected)[:20]
        raise ValueError(f"Non-contiguous record indices; missing={missing}, extra={extra}")

    out = resolve_repo_path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Merged {len(rows)} sharded records -> {out}", flush=True)


if __name__ == "__main__":
    main()

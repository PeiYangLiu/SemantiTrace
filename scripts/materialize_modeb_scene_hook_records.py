#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from semantitrace.modeb_queries import build_modeb_scene_hook_queries
from semantitrace.records import load_records_with_resolved_paths, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Materialize Mode-B natural records with scene-hook audit queries.")
    parser.add_argument("--records", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--record_root", default=None)
    parser.add_argument("--max_records", type=int, default=500)
    parser.add_argument("--num_queries", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records_with_resolved_paths(
        args.records,
        args.max_records,
        record_root=args.record_root,
        require_images=False,
    )
    out_rows = []
    for record in records:
        row = dict(record)
        for key in list(row):
            if key.startswith("_resolved_") or key == "_record_mode":
                row.pop(key, None)
        row["probe_queries_original"] = list(row.get("probe_queries", []))
        row["probe_query_policy"] = "modeb_scene_hook_v1"
        row["probe_queries"] = build_modeb_scene_hook_queries(row, num_queries=args.num_queries)
        out_rows.append(row)

    out = resolve_repo_path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(out_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(out_rows)} records with scene-hook Mode-B queries to {out}", flush=True)


if __name__ == "__main__":
    main()

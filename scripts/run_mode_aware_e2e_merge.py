#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_mode_aware_e2e_from_topk import summarize
from semantitrace.config import load_config
from semantitrace.records import load_records_with_resolved_paths, resolve_repo_path
from semantitrace.verification import Verifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Merge mode-aware E2E shard details and recompute summaries.")
    parser.add_argument("--records", required=True)
    parser.add_argument("--record_root", default=None)
    parser.add_argument("--shards_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--config", default="configs/semantitrace_flux2_klein_gradient_guided.yaml")
    parser.add_argument("--max_records", type=int, default=500)
    parser.add_argument("--index_size", type=int, required=True)
    parser.add_argument("--top_k", type=int, default=3)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    return resolve_repo_path(path)


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.output_dir)
    profile_dir = out_dir / args.profile
    profile_dir.mkdir(parents=True, exist_ok=True)
    records = load_records_with_resolved_paths(args.records, args.max_records, record_root=args.record_root)
    details = []
    for shard_dir in sorted(resolve(args.shards_dir).glob("shard_*")):
        path = shard_dir / args.profile / "end_to_end_details.jsonl"
        if not path.exists():
            continue
        details.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    details.sort(key=lambda row: (int(row["record_index"]), int(row["probe_index"])))
    if not details:
        raise FileNotFoundError(f"No end_to_end_details.jsonl files found under {args.shards_dir}")

    cfg = load_config(args.config)
    verifier = Verifier(cfg.get("verification", {}))
    verifier.num_probes_per_canary = min(verifier.num_probes_per_canary, 3)
    summary_rows = summarize(details, records, verifier, args)
    report = {"summary": summary_rows, "details": details}
    (profile_dir / "end_to_end_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "end_to_end_profile_summary.json").write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out_dir / "end_to_end_profile_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        fieldnames = list(summary_rows[0].keys())
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(json.dumps(summary_rows, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

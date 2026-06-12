#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantitrace import SemantiTracePipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run SemantiTrace canary injection")
    parser.add_argument("--dataset_dir", required=True, help="Directory of source images")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config")
    parser.add_argument("--num_canaries", type=int, default=None, help="Override number of canaries")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--resume", action="store_true", help="Append to an existing output_dir/canary_records.json")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    pipeline = SemantiTracePipeline(args.config, device=args.device)
    records = pipeline.inject_canaries(args.dataset_dir, args.output_dir, args.num_canaries, resume=args.resume)
    print(f"Saved {len(records)} canaries to {Path(args.output_dir) / 'canary_records.json'}")


if __name__ == "__main__":
    main()

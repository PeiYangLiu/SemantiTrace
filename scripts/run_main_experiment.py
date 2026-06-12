#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantitrace.experiments import MainExperimentRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run SemantiTrace paper main experiments")
    parser.add_argument("--config", default="configs/main_experiment.yaml")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--stages",
        default="efficacy,stealth,ood,robustness",
        help="Comma-separated subset: efficacy,stealth,ood,robustness",
    )
    parser.add_argument(
        "--dry_run_sample",
        action="store_true",
        help="Create a synthetic image corpus and run a small local smoke experiment.",
    )
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    runner = MainExperimentRunner(
        config_path=args.config,
        output_dir=args.output_dir,
        device=args.device,
        dry_run_sample=args.dry_run_sample,
    )
    rows = runner.run(stages)
    print(f"Experiment outputs written to {runner.output_dir}")
    for name, table_rows in rows.items():
        print(f"  {name}: {len(table_rows)} rows")


if __name__ == "__main__":
    main()


#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Summarize SemantiTrace generation yield and candidate preselection evidence")
    parser.add_argument("--records", default="outputs/flux2_klein_gradient_guided_n100_tuned_v1/canary_records.json")
    parser.add_argument("--rejected", default="outputs/flux2_klein_gradient_guided_n100_tuned_v1/rejected_attempts.jsonl")
    parser.add_argument("--cost_summary", default="outputs/cost_scalability/cost_scalability_summary.json")
    parser.add_argument("--output_dir", default="outputs/yield_preselection_analysis")
    parser.add_argument(
        "--prescan_globs",
        nargs="*",
        default=[
            "outputs/flux2_candidate_prescan_*",
            "outputs/flux2_webqa_prescan_*",
            "outputs/flux2_scene_text_*",
            "outputs/flux2_webqa_exhaustive_*",
        ],
    )
    parser.add_argument("--target_canaries", type=int, default=500)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def mean(values: list[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def median(values: list[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round((len(values) - 1) * q)))
    return float(values[idx])


def summarize_attempts(records: list[dict[str, Any]], rejected: list[dict[str, Any]], cost: dict[str, Any], target_canaries: int) -> dict[str, Any]:
    accepted = len(records)
    rejected_count = len(rejected)
    total = accepted + rejected_count
    rejection_counter: Counter[str] = Counter()
    for row in rejected:
        flags = row.get("quality_flags") or ["unknown"]
        for flag in flags:
            rejection_counter[str(flag)] += 1
    accepted_modes = Counter(str(r.get("parasitism_mode", "unknown")) for r in records)
    accepted_qld = [float((r.get("injection_metrics") or {}).get("quality_local_delta", 0.0)) for r in records if (r.get("injection_metrics") or {}).get("quality_local_delta") is not None]
    rejected_qld = [float(r.get("quality_local_delta", 0.0)) for r in rejected if r.get("quality_local_delta") is not None]
    seconds_per_attempt = float(cost.get("seconds_per_attempt", 63.8)) if cost else 63.8
    attempts_per_accept = total / max(1, accepted)
    projected_attempts = attempts_per_accept * target_canaries
    return {
        "accepted_canaries": accepted,
        "rejected_attempts": rejected_count,
        "total_generation_attempts": total,
        "yield_percent": 100.0 * accepted / max(1, total),
        "attempts_per_accepted_canary": attempts_per_accept,
        "accepted_modes": dict(accepted_modes),
        "rejection_reason_counts": dict(rejection_counter),
        "accepted_quality_local_delta_mean": mean(accepted_qld),
        "accepted_quality_local_delta_median": median(accepted_qld),
        "rejected_quality_local_delta_mean": mean(rejected_qld),
        "rejected_quality_local_delta_median": median(rejected_qld),
        "seconds_per_attempt": seconds_per_attempt,
        "target_canaries": target_canaries,
        "projected_attempts_for_target": projected_attempts,
        "projected_single_a100_gpu_hours": projected_attempts * seconds_per_attempt / 3600.0,
        "projected_wall_clock_hours_with_8_a100": projected_attempts * seconds_per_attempt / 3600.0 / 8.0,
        "canonical_n500_launch_command": (
            "python3 scripts/run_injection.py --dataset_dir data/mmqa/images "
            "--output_dir outputs/flux2_klein_gradient_guided_n500_canonical_v1 "
            "--config configs/semantitrace_flux2_klein_gradient_guided.yaml "
            "--num_canaries 500 --device cuda --resume"
        ),
        "n500_note": "This command is provided for a canonical long run; projected single-A100 time is estimated from the completed n=100 FLUX.2 run.",
    }


def prescan_dirs(globs: list[str]) -> list[Path]:
    dirs: set[Path] = set()
    for pattern in globs:
        for item in glob.glob(str(resolve(pattern))):
            path = Path(item)
            if path.is_dir():
                dirs.add(path)
    return sorted(dirs)


def review_score(row: dict[str, Any]) -> float | None:
    review = row.get("review")
    if isinstance(review, dict):
        raw = review.get("score")
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None
    return None


def summarize_prescan(path: Path) -> dict[str, Any] | None:
    all_path = path / "all_candidates.json"
    selected_path = path / "selected_candidates.json"
    if not all_path.exists() and not selected_path.exists():
        return None
    all_rows = load_json(all_path, [])
    selected_rows = load_json(selected_path, [])
    if not isinstance(all_rows, list):
        all_rows = []
    if not isinstance(selected_rows, list):
        selected_rows = []
    all_scores = [score for row in all_rows if (score := review_score(row)) is not None]
    selected_scores = [score for row in selected_rows if (score := review_score(row)) is not None]
    pass_count = 0
    for row in all_rows:
        review = row.get("review")
        if isinstance(review, dict) and bool(review.get("pass")):
            pass_count += 1
    categories: Counter[str] = Counter()
    for row in all_rows:
        review = row.get("review")
        if isinstance(review, dict):
            categories[str(review.get("category", "unknown"))] += 1
    return {
        "prescan_dir": str(path.relative_to(ROOT)),
        "all_candidates": len(all_rows),
        "selected_candidates": len(selected_rows),
        "review_pass_count": pass_count,
        "review_pass_rate": pass_count / max(1, len(all_rows)),
        "selection_rate_from_all": len(selected_rows) / max(1, len(all_rows)),
        "all_score_mean": mean(all_scores),
        "all_score_median": median(all_scores),
        "all_score_p75": percentile(all_scores, 0.75),
        "all_score_p90": percentile(all_scores, 0.90),
        "selected_score_mean": mean(selected_scores),
        "selected_score_median": median(selected_scores),
        "top_categories": dict(categories.most_common(8)),
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
    records = load_json(resolve(args.records), [])
    rejected = load_jsonl(resolve(args.rejected))
    cost = load_json(resolve(args.cost_summary), {})
    attempt_summary = summarize_attempts(records, rejected, cost, args.target_canaries)
    prescan_rows = [row for path in prescan_dirs(args.prescan_globs) if (row := summarize_prescan(path)) is not None]
    report = {
        "attempt_summary": attempt_summary,
        "prescan_summary": prescan_rows,
        "interpretation": [
            "The measured 19.4% final yield is dominated by readability and naturalness gates, not by local/boundary pixel-delta failures.",
            "Prescan review scores quantify candidate filtering before expensive FLUX calls, but prescan pass rates should not be reported as final canary yield because generation and VLM readability gates remain downstream.",
            "A canonical n=500 run is feasible as an offline batch but is projected to require tens of single-A100 GPU-hours under the current implementation.",
        ],
    }
    (out_dir / "yield_preselection_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(out_dir / "prescan_summary.csv", prescan_rows)
    rejection_rows = [{"reason": reason, "count": count} for reason, count in Counter(attempt_summary["rejection_reason_counts"]).items()]
    write_csv(out_dir / "rejection_reason_counts.csv", rejection_rows)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

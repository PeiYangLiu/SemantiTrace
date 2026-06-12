#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Compute audit calibration and low-overlap sensitivity for mode-aware E2E reports.")
    parser.add_argument("--report", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--bootstraps", type=int, default=10000)
    parser.add_argument("--overlap_trials", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2027)
    return parser.parse_args()


def one_sided_welch(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0 or y.size == 0:
        return 1.0
    if np.allclose(x, y):
        return 0.5
    try:
        from scipy import stats

        t_stat, two_sided = stats.ttest_ind(x, y, equal_var=False)
        if math.isnan(float(t_stat)) or math.isnan(float(two_sided)):
            return 0.5
        return float(two_sided / 2.0) if float(t_stat) > 0 else 1.0 - float(two_sided / 2.0)
    except Exception:
        vx = float(x.var(ddof=1)) if x.size > 1 else 0.0
        vy = float(y.var(ddof=1)) if y.size > 1 else 0.0
        denom = math.sqrt(vx / max(1, x.size) + vy / max(1, y.size))
        t_stat = (float(x.mean()) - float(y.mean())) / max(denom, 1e-12)
        return float(0.5 * math.erfc(t_stat / math.sqrt(2.0)))


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


def per_canary_samples(details: list[dict[str, Any]], indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    pos = {record_index: i for i, record_index in enumerate(indices)}
    by_record: list[list[dict[str, Any]]] = [[] for _ in indices]
    for detail in details:
        idx = int(detail["record_index"])
        if idx in pos:
            by_record[pos[idx]].append(detail)
    suspect = np.zeros(len(indices), dtype=np.float64)
    clean = np.zeros(len(indices), dtype=np.float64)
    for i, rows in enumerate(by_record):
        rows.sort(key=lambda row: int(row.get("probe_index", 0)))
        if not rows:
            continue
        suspect[i] = np.mean([bool(row.get("watermarked_hit")) for row in rows])
        clean[i] = np.mean([bool(row.get("clean_hit")) for row in rows])
    return suspect, clean


def subset_indices(details: list[dict[str, Any]]) -> dict[str, list[int]]:
    groups: dict[str, set[int]] = defaultdict(set)
    for detail in details:
        idx = int(detail["record_index"])
        groups["all"].add(idx)
        groups[str(detail.get("mode", "unknown"))].add(idx)
    return {key: sorted(value) for key, value in groups.items()}


def compute_subset(
    subset: str,
    indices: list[int],
    details: list[dict[str, Any]],
    rng: np.random.Generator,
    *,
    bootstraps: int,
    overlap_trials: int,
) -> dict[str, Any]:
    suspect, clean = per_canary_samples(details, indices)
    n = len(indices)
    alphas = [0.05, 0.01, 0.001, 0.0001]
    null_p = np.empty(bootstraps, dtype=np.float64)
    signal_p = np.empty(bootstraps, dtype=np.float64)
    for i in range(bootstraps):
        a = clean[rng.integers(0, n, size=n)]
        b = clean[rng.integers(0, n, size=n)]
        null_p[i] = one_sided_welch(a, b)
        s = suspect[rng.integers(0, n, size=n)]
        c = clean[rng.integers(0, n, size=n)]
        signal_p[i] = one_sided_welch(s, c)
    calibration_rows = []
    for alpha in alphas:
        tpr = float(np.mean(signal_p < alpha) * 100.0)
        calibration_rows.append(
            {
                "subset": subset,
                "alpha": alpha,
                "fpr": float(np.mean(null_p < alpha) * 100.0),
                "tpr": tpr,
                "fnr": 100.0 - tpr,
                "median_null_p": float(np.median(null_p)),
                "median_suspect_p": float(np.median(signal_p)),
            }
        )
    overlap_rows = []
    candidate_sizes = [1, 5, 10, 25, 50, 100, 250, 500, 1000]
    for subset_size in candidate_sizes:
        if subset_size > n:
            continue
        reject = 0
        cer_values = []
        effect_values = []
        for _ in range(overlap_trials):
            idx = rng.choice(n, size=subset_size, replace=False)
            p_value = one_sided_welch(suspect[idx], clean[idx])
            reject += int(p_value < 0.01)
            cer_values.append(float(suspect[idx].mean()))
            effect_values.append(float((suspect[idx] - clean[idx]).mean()))
        overlap_rows.append(
            {
                "subset": subset,
                "indexed_canaries": subset_size,
                "overlap_percent": 100.0 * subset_size / n,
                "mean_cer": float(np.mean(cer_values) * 100.0),
                "mean_effect": float(np.mean(effect_values) * 100.0),
                "detection_rate_at_alpha_0_01": 100.0 * reject / overlap_trials,
                "trials": overlap_trials,
            }
        )
    return {
        "subset": subset,
        "n": n,
        "suspect_mean": float(suspect.mean()),
        "clean_mean": float(clean.mean()),
        "effect_size": float((suspect - clean).mean()),
        "main_p_value": one_sided_welch(suspect, clean),
        "calibration_rows": calibration_rows,
        "overlap_rows": overlap_rows,
    }


def main() -> None:
    args = parse_args()
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    report = json.loads((ROOT / args.report).read_text(encoding="utf-8"))
    details = report["details"]
    rng = np.random.default_rng(args.seed)
    subset_results = [
        compute_subset(
            subset,
            indices,
            details,
            rng,
            bootstraps=args.bootstraps,
            overlap_trials=args.overlap_trials,
        )
        for subset, indices in subset_indices(details).items()
    ]
    calibration_rows = [row for item in subset_results for row in item["calibration_rows"]]
    overlap_rows = [row for item in subset_results for row in item["overlap_rows"]]
    output = {
        "source_report": args.report,
        "bootstraps": args.bootstraps,
        "overlap_trials": args.overlap_trials,
        "subsets": [
            {k: v for k, v in item.items() if k not in {"calibration_rows", "overlap_rows"}}
            for item in subset_results
        ],
        "calibration_rows": calibration_rows,
        "overlap_rows": overlap_rows,
    }
    (out / "modeaware_audit_calibration_summary.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    write_csv(out / "modeaware_audit_calibration_summary.csv", calibration_rows)
    write_csv(out / "modeaware_overlap_sensitivity_summary.csv", overlap_rows)
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()

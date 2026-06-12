from __future__ import annotations

import logging
import math
from typing import Any, Callable

import numpy as np

from semantitrace.metrics import contains_positive_signature

logger = logging.getLogger(__name__)


class Verifier:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = config or {}
        self.num_probes_per_canary = int(cfg.get("num_probes_per_canary", 3))
        self.significance_level = float(cfg.get("significance_level", 0.01))
        self.clean_default_mean = float(cfg.get("clean_default_mean", 0.0))
        self.clean_default_std = float(cfg.get("clean_default_std", 0.0001))

    def compute_cer(self, responses: list[str], signatures: list[str]) -> float:
        samples = self.compute_per_canary_cer(responses, signatures)
        if samples.size == 0:
            return 0.0
        return float(samples.mean())

    def compute_per_canary_cer(self, responses: list[str], signatures: list[str]) -> np.ndarray:
        k = self.num_probes_per_canary
        samples = np.zeros(len(signatures), dtype=np.float64)
        for i, signature in enumerate(signatures):
            hits = 0
            total = 0
            for j in range(k):
                idx = i * k + j
                if idx >= len(responses):
                    break
                hits += int(contains_positive_signature(responses[idx], signature))
                total += 1
            samples[i] = hits / total if total else 0.0
        return samples

    def welch_t_test(
        self,
        suspect_samples: np.ndarray,
        clean_samples: np.ndarray | None = None,
    ) -> dict[str, Any]:
        suspect_samples = np.asarray(suspect_samples, dtype=np.float64)
        if suspect_samples.size == 0:
            raise ValueError("Welch test requires suspect samples")

        if clean_samples is None:
            clean_mean = self.clean_default_mean
            clean_std = self.clean_default_std
            n_clean = max(1, suspect_samples.size)
        else:
            clean_samples = np.asarray(clean_samples, dtype=np.float64)
            clean_mean = float(clean_samples.mean()) if clean_samples.size else self.clean_default_mean
            clean_std = float(clean_samples.std(ddof=1)) if clean_samples.size > 1 else self.clean_default_std
            n_clean = max(1, clean_samples.size)

        suspect_mean = float(suspect_samples.mean())
        suspect_std = float(suspect_samples.std(ddof=1)) if suspect_samples.size > 1 else 0.0
        n_suspect = suspect_samples.size

        try:
            from scipy import stats

            t_stat, two_sided = stats.ttest_ind_from_stats(
                mean1=suspect_mean,
                std1=max(suspect_std, 1e-12),
                nobs1=n_suspect,
                mean2=clean_mean,
                std2=max(clean_std, 1e-12),
                nobs2=n_clean,
                equal_var=False,
            )
            p_value = float(two_sided / 2.0) if t_stat > 0 else 1.0 - float(two_sided / 2.0)
        except Exception:
            denom = np.sqrt((suspect_std**2) / n_suspect + (clean_std**2) / n_clean)
            t_stat = (suspect_mean - clean_mean) / max(float(denom), 1e-12)
            p_value = float(0.5 * math.erfc(float(t_stat) / math.sqrt(2.0)))

        reject = bool(p_value < self.significance_level)
        return {
            "t_statistic": float(t_stat),
            "p_value": p_value,
            "reject_h0": reject,
            "decision": (
                f"REJECT H0 (p={p_value:.6g} < alpha={self.significance_level})"
                if reject
                else f"FAIL TO REJECT H0 (p={p_value:.6g} >= alpha={self.significance_level})"
            ),
            "cer_suspect": suspect_mean,
            "cer_clean": clean_mean,
            "n_suspect": int(n_suspect),
            "n_clean": int(n_clean),
        }

    def bootstrap_rate_test(
        self,
        suspect_samples: np.ndarray,
        clean_samples: np.ndarray | None = None,
        *,
        iterations: int = 10000,
        seed: int = 2027,
    ) -> dict[str, Any]:
        suspect_samples = np.asarray(suspect_samples, dtype=np.float64)
        if suspect_samples.size == 0:
            raise ValueError("Bootstrap test requires suspect samples")
        if clean_samples is None:
            clean_samples = np.full_like(suspect_samples, self.clean_default_mean, dtype=np.float64)
        else:
            clean_samples = np.asarray(clean_samples, dtype=np.float64)
        if clean_samples.size != suspect_samples.size:
            raise ValueError("Bootstrap test expects paired per-canary suspect and clean samples")

        diff = suspect_samples - clean_samples
        observed = float(diff.mean())
        rng = np.random.default_rng(seed)
        n = diff.size

        boot = np.empty(iterations, dtype=np.float64)
        null = np.empty(iterations, dtype=np.float64)
        for i in range(iterations):
            sample_idx = rng.integers(0, n, size=n)
            boot[i] = float(diff[sample_idx].mean())
            signs = rng.choice(np.array([-1.0, 1.0]), size=n)
            null[i] = float((diff * signs).mean())

        p_value = float((np.count_nonzero(null >= observed) + 1) / (iterations + 1))
        ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
        reject = bool(p_value < self.significance_level)
        return {
            "test": "paired_canary_bootstrap_signflip",
            "effect_size": observed,
            "effect_ci95_low": float(ci_low),
            "effect_ci95_high": float(ci_high),
            "p_value": p_value,
            "reject_h0": reject,
            "iterations": int(iterations),
            "seed": int(seed),
            "cer_suspect": float(suspect_samples.mean()),
            "cer_clean": float(clean_samples.mean()),
            "n_pairs": int(n),
        }

    def run_verification(
        self,
        canary_records: list[dict[str, Any]],
        query_fn: Callable[[str], str],
        clean_baseline_fn: Callable[[str], str] | None = None,
    ) -> dict[str, Any]:
        signatures: list[str] = []
        suspect_responses: list[str] = []
        clean_responses: list[str] = []

        for record in canary_records:
            signature = str(record["trap_signature"])
            signatures.append(signature)
            queries = list(record.get("probe_queries", []))[: self.num_probes_per_canary]
            for query in queries:
                suspect_responses.append(str(query_fn(query)))
                if clean_baseline_fn is not None:
                    clean_responses.append(str(clean_baseline_fn(query)))

        suspect_samples = self.compute_per_canary_cer(suspect_responses, signatures)
        clean_samples = (
            self.compute_per_canary_cer(clean_responses, signatures)
            if clean_baseline_fn is not None
            else None
        )
        test = self.welch_t_test(suspect_samples, clean_samples)
        bootstrap_test = self.bootstrap_rate_test(suspect_samples, clean_samples)
        report = {
            "num_canaries": len(canary_records),
            "num_probes_per_canary": self.num_probes_per_canary,
            "suspect_cer": float(suspect_samples.mean()) if suspect_samples.size else 0.0,
            "suspect_per_canary_cer": suspect_samples.tolist(),
            "clean_cer": float(clean_samples.mean()) if clean_samples is not None and clean_samples.size else None,
            "clean_per_canary_cer": clean_samples.tolist() if clean_samples is not None else None,
            "test_result": test,
            "bootstrap_test_result": bootstrap_test,
        }
        logger.info("%s", test["decision"])
        return report

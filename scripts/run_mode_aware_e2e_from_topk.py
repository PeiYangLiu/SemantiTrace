#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_end_to_end_profiles import image_context_prompt, make_montage
from semantitrace.backends.real import QwenVLMClient
from semantitrace.config import load_config
from semantitrace.mode_verification import (
    detail_response_hit,
    detail_target_gated_hit,
    per_canary_rates_from_predicate,
    score_response,
    target_rank_in_topk,
)
from semantitrace.records import infer_record_mode, load_records_with_resolved_paths, resolve_repo_path
from semantitrace.verification import Verifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run mode-aware Qwen E2E profile from precomputed top-k hit rows.")
    parser.add_argument("--records", required=True)
    parser.add_argument("--record_root", default=None)
    parser.add_argument("--hits", required=True)
    parser.add_argument("--hit_image_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--config", default="configs/semantitrace_flux2_klein_gradient_guided.yaml")
    parser.add_argument("--max_records", type=int, default=500)
    parser.add_argument("--index_size", type=int, required=True)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--montage_tile", type=int, default=448)
    parser.add_argument("--query_shard_count", type=int, default=1)
    parser.add_argument("--query_shard_index", type=int, default=0)
    parser.add_argument("--fresh", action="store_true")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    return resolve_repo_path(path)


def remap_hit_path(path: str | Path, hit_image_root: Path) -> Path:
    raw = str(path)
    marker = "hit_images/"
    if marker in raw:
        rel = raw.split(marker, 1)[1]
        return hit_image_root / rel
    p = Path(raw)
    if p.is_absolute():
        return p
    return resolve(p)


def remap_hits(hits: list[dict[str, Any]], hit_image_root: Path) -> list[dict[str, Any]]:
    out = []
    for hit in hits:
        row = dict(hit)
        row["image_path"] = str(remap_hit_path(row["image_path"], hit_image_root))
        out.append(row)
    return out


def build_vlm(config_path: str, device: str) -> tuple[QwenVLMClient, Verifier]:
    cfg = load_config(config_path)
    models = cfg.get("models", {})
    vlm_cfg = cfg.get("vlm", {})
    vlm = QwenVLMClient(
        model_name=models.get("surrogate_vlm", "Qwen/Qwen3-VL-8B-Instruct"),
        device=device,
        torch_dtype=vlm_cfg.get("torch_dtype", "bfloat16"),
    )
    verifier = Verifier(cfg.get("verification", {}))
    verifier.num_probes_per_canary = min(verifier.num_probes_per_canary, 3)
    return vlm, verifier


def summarize_subset(
    details: list[dict[str, Any]],
    records: list[dict[str, Any]],
    verifier: Any,
    args: argparse.Namespace,
    subset: str,
    record_indices: list[int],
) -> dict[str, Any]:
    index_set = set(record_indices)
    subset_details = [d for d in details if int(d.get("record_index", -1)) in index_set]
    suspect_samples = per_canary_rates_from_predicate(
        subset_details,
        lambda row: detail_target_gated_hit(row, "watermarked", args.top_k),
        len(records),
    )
    clean_samples = per_canary_rates_from_predicate(
        subset_details,
        lambda row: detail_target_gated_hit(row, "clean", args.top_k),
        len(records),
    )
    strict_samples = per_canary_rates_from_predicate(
        subset_details,
        lambda row: detail_target_gated_hit(row, "watermarked", args.top_k, strict=True),
        len(records),
    )
    clean_strict_samples = per_canary_rates_from_predicate(
        subset_details,
        lambda row: detail_target_gated_hit(row, "clean", args.top_k, strict=True),
        len(records),
    )
    ungated_samples = per_canary_rates_from_predicate(
        subset_details,
        lambda row: detail_response_hit(row, "watermarked"),
        len(records),
    )
    clean_ungated_samples = per_canary_rates_from_predicate(
        subset_details,
        lambda row: detail_response_hit(row, "clean"),
        len(records),
    )
    ungated_strict_samples = per_canary_rates_from_predicate(
        subset_details,
        lambda row: detail_response_hit(row, "watermarked", strict=True),
        len(records),
    )
    clean_ungated_strict_samples = per_canary_rates_from_predicate(
        subset_details,
        lambda row: detail_response_hit(row, "clean", strict=True),
        len(records),
    )
    ranks = np.asarray([int(d["target_rank"]) for d in subset_details if "target_rank" in d], dtype=float)
    clean_ranks = np.asarray([int(d["clean_target_rank"]) for d in subset_details if "clean_target_rank" in d], dtype=float)
    test = verifier.welch_t_test(suspect_samples, clean_samples) if suspect_samples.size else {"p_value": 1.0, "reject_h0": False}
    boot = (
        verifier.bootstrap_rate_test(suspect_samples, clean_samples)
        if suspect_samples.size and clean_samples.size == suspect_samples.size
        else {"p_value": 1.0, "effect_size": 0.0, "effect_ci95_low": 0.0, "effect_ci95_high": 0.0, "reject_h0": False}
    )
    strict_test = verifier.welch_t_test(strict_samples, clean_strict_samples) if strict_samples.size else {"p_value": 1.0, "reject_h0": False}
    strict_boot = (
        verifier.bootstrap_rate_test(strict_samples, clean_strict_samples)
        if strict_samples.size and clean_strict_samples.size == strict_samples.size
        else {"p_value": 1.0, "effect_size": 0.0, "effect_ci95_low": 0.0, "effect_ci95_high": 0.0, "reject_h0": False}
    )
    return {
        "profile": args.profile,
        "label": args.label,
        "variant": args.variant,
        "subset": subset,
        "context": "image_montage",
        "num_canaries": len(record_indices),
        "num_queries": len(subset_details),
        "index_size": args.index_size,
        "top_k": args.top_k,
        "suspect_cer": float(suspect_samples.mean()) if suspect_samples.size else 0.0,
        "clean_cer": float(clean_samples.mean()) if clean_samples.size else 0.0,
        "suspect_strict_rate": float(strict_samples.mean()) if strict_samples.size else 0.0,
        "clean_strict_rate": float(clean_strict_samples.mean()) if clean_strict_samples.size else 0.0,
        "suspect_ungated_response_rate": float(ungated_samples.mean()) if ungated_samples.size else 0.0,
        "clean_ungated_response_rate": float(clean_ungated_samples.mean()) if clean_ungated_samples.size else 0.0,
        "suspect_ungated_strict_rate": float(ungated_strict_samples.mean()) if ungated_strict_samples.size else 0.0,
        "clean_ungated_strict_rate": float(clean_ungated_strict_samples.mean()) if clean_ungated_strict_samples.size else 0.0,
        "p_value": test["p_value"],
        "bootstrap_p_value": boot["p_value"],
        "effect_size": boot["effect_size"],
        "effect_ci95_low": boot["effect_ci95_low"],
        "effect_ci95_high": boot["effect_ci95_high"],
        "reject_h0": bool(test["reject_h0"]),
        "strict_p_value": strict_test["p_value"],
        "strict_bootstrap_p_value": strict_boot["p_value"],
        "strict_effect_size": strict_boot["effect_size"],
        "strict_effect_ci95_low": strict_boot["effect_ci95_low"],
        "strict_effect_ci95_high": strict_boot["effect_ci95_high"],
        "strict_reject_h0": bool(strict_test["reject_h0"]),
        "recall_at_1": float(np.mean(ranks <= 1)) if ranks.size else 0.0,
        "recall_at_3": float(np.mean(ranks <= 3)) if ranks.size else 0.0,
        "recall_at_10": float(np.mean(ranks <= 10)) if ranks.size else 0.0,
        "mean_target_rank": float(ranks.mean()) if ranks.size else 0.0,
        "clean_recall_at_3": float(np.mean(clean_ranks <= 3)) if clean_ranks.size else 0.0,
        "clean_mean_target_rank": float(clean_ranks.mean()) if clean_ranks.size else 0.0,
        "description": (
            f"{args.label} top-k image context -> Qwen3-VL generation with mode-aware scoring; "
            "main CER is target-rank gated"
        ),
    }


def summarize(details: list[dict[str, Any]], records: list[dict[str, Any]], verifier: Any, args: argparse.Namespace) -> list[dict[str, Any]]:
    groups: dict[str, list[int]] = {"all": list(range(len(records)))}
    for idx, record in enumerate(records):
        groups.setdefault(infer_record_mode(record), []).append(idx)
    return [summarize_subset(details, records, verifier, args, subset, indices) for subset, indices in groups.items()]


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.output_dir)
    profile_dir = out_dir / args.profile
    profile_dir.mkdir(parents=True, exist_ok=True)
    hit_image_root = resolve(args.hit_image_root)
    records = load_records_with_resolved_paths(args.records, args.max_records, record_root=args.record_root)
    hit_rows = [json.loads(line) for line in resolve(args.hits).read_text(encoding="utf-8").splitlines() if line.strip()]
    hit_rows = [
        row
        for global_idx, row in enumerate(hit_rows)
        if int(row["record_index"]) < len(records) and global_idx % args.query_shard_count == args.query_shard_index
    ]

    vlm, verifier = build_vlm(args.config, args.device)
    details_path = profile_dir / "end_to_end_details.jsonl"
    if args.fresh and details_path.exists():
        details_path.unlink()
    details = []
    if details_path.exists():
        details = [json.loads(line) for line in details_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    skip = len(details)
    with details_path.open("a", encoding="utf-8") as fh:
        for flat_idx, row in enumerate(hit_rows):
            if flat_idx < skip:
                continue
            record_index = int(row["record_index"])
            record = records[record_index]
            query = str(row["query"])
            watermarked_hits = remap_hits(row["watermarked_hits"], hit_image_root)
            clean_hits = remap_hits(row["clean_hits"], hit_image_root)
            wm_context_path = make_montage(
                watermarked_hits,
                profile_dir / "contexts" / "watermarked" / f"{record_index:04d}_{row['probe_index']}.jpg",
                args.montage_tile,
            )
            clean_context_path = make_montage(
                clean_hits,
                profile_dir / "contexts" / "clean" / f"{record_index:04d}_{row['probe_index']}.jpg",
                args.montage_tile,
            )
            prompt = image_context_prompt(query)
            watermarked_response = vlm.generate(Image.open(wm_context_path).convert("RGB"), prompt, temperature=0.0, max_new_tokens=args.max_new_tokens)
            clean_response = vlm.generate(Image.open(clean_context_path).convert("RGB"), prompt, temperature=0.0, max_new_tokens=args.max_new_tokens)
            wm_score = score_response(watermarked_response, record)
            clean_score = score_response(clean_response, record)
            watermarked_target_in_topk = target_rank_in_topk(row.get("target_rank"), args.top_k)
            clean_target_in_topk = target_rank_in_topk(row.get("clean_target_rank"), args.top_k)
            watermarked_response_hit = bool(wm_score["hit"])
            clean_response_hit = bool(clean_score["hit"])
            watermarked_response_strict_hit = bool(wm_score.get("strict_hit", wm_score["hit"]))
            clean_response_strict_hit = bool(clean_score.get("strict_hit", clean_score["hit"]))
            detail = {
                **row,
                "query_shard_count": args.query_shard_count,
                "query_shard_index": args.query_shard_index,
                "mode": infer_record_mode(record),
                "watermarked_hits": watermarked_hits,
                "clean_hits": clean_hits,
                "watermarked_context_path": str(wm_context_path),
                "clean_context_path": str(clean_context_path),
                "watermarked_response": watermarked_response,
                "clean_response": clean_response,
                "watermarked_score": wm_score,
                "clean_score": clean_score,
                "watermarked_target_in_topk": watermarked_target_in_topk,
                "clean_target_in_topk": clean_target_in_topk,
                "watermarked_response_hit": watermarked_response_hit,
                "clean_response_hit": clean_response_hit,
                "watermarked_response_strict_hit": watermarked_response_strict_hit,
                "clean_response_strict_hit": clean_response_strict_hit,
                "watermarked_hit": bool(watermarked_response_hit and watermarked_target_in_topk),
                "clean_hit": bool(clean_response_hit and clean_target_in_topk),
                "watermarked_strict_hit": bool(watermarked_response_strict_hit and watermarked_target_in_topk),
                "clean_strict_hit": bool(clean_response_strict_hit and clean_target_in_topk),
            }
            fh.write(json.dumps(detail, ensure_ascii=False) + "\n")
            fh.flush()
            details.append(detail)
            print(
                f"[{len(details):04d}/{len(hit_rows):04d}] {row['id']} "
                f"mode={detail['mode']} rank={row['target_rank']} "
                f"wm_hit={detail['watermarked_hit']} clean_hit={detail['clean_hit']}",
                flush=True,
            )

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

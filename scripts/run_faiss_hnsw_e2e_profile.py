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

from run_cached_100k_e2e_profile import build_vlm, summarize
from run_end_to_end_profiles import image_context_prompt, make_montage
from run_pipeline_generality import load_records, resolve
from semantitrace.metrics import contains_positive_signature


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run Qwen E2E profile from precomputed FAISS HNSW top-k hits.")
    parser.add_argument("--records", default="outputs/semantitrace_n500_textvqa_merged/canary_records_first500_local.json")
    parser.add_argument("--hits", default="outputs/faiss_hnsw_100k_e2e_hits/faiss_hnsw_top_hits.jsonl")
    parser.add_argument("--output_dir", default="outputs/end_to_end_profiles_faiss_hnsw_100k_n500")
    parser.add_argument("--profile", default="faiss_hnsw_100k")
    parser.add_argument("--label", default="FAISS HNSW 100k")
    parser.add_argument("--variant", default="faiss_hnsw_clip_100k")
    parser.add_argument("--config", default="configs/semantitrace_flux2_klein_gradient_guided.yaml")
    parser.add_argument("--max_records", type=int, default=500)
    parser.add_argument("--index_size", type=int, default=100500)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--montage_tile", type=int, default=448)
    parser.add_argument("--fresh", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.output_dir)
    profile_dir = out_dir / args.profile
    profile_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(resolve(args.records), args.max_records)
    hit_rows = [json.loads(line) for line in resolve(args.hits).read_text(encoding="utf-8").splitlines() if line.strip()]
    hit_rows = [row for row in hit_rows if int(row["record_index"]) < len(records)]

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
            signature = str(row["signature"])
            query = str(row["query"])
            wm_context_path = make_montage(
                row["watermarked_hits"],
                profile_dir / "contexts" / "watermarked" / f"{record_index:04d}_{row['probe_index']}.jpg",
                args.montage_tile,
            )
            clean_context_path = make_montage(
                row["clean_hits"],
                profile_dir / "contexts" / "clean" / f"{record_index:04d}_{row['probe_index']}.jpg",
                args.montage_tile,
            )
            prompt = image_context_prompt(query)
            watermarked_response = vlm.generate(Image.open(wm_context_path).convert("RGB"), prompt, temperature=0.0, max_new_tokens=args.max_new_tokens)
            clean_response = vlm.generate(Image.open(clean_context_path).convert("RGB"), prompt, temperature=0.0, max_new_tokens=args.max_new_tokens)
            detail = {
                **row,
                "watermarked_context_path": str(wm_context_path.relative_to(ROOT)),
                "clean_context_path": str(clean_context_path.relative_to(ROOT)),
                "watermarked_response": watermarked_response,
                "clean_response": clean_response,
                "watermarked_hit": contains_positive_signature(watermarked_response, signature),
                "clean_hit": contains_positive_signature(clean_response, signature),
            }
            fh.write(json.dumps(detail, ensure_ascii=False) + "\n")
            fh.flush()
            details.append(detail)
            print(f"[{len(details):03d}/{len(hit_rows):03d}] {row['id']} rank={row['target_rank']} wm_hit={detail['watermarked_hit']} clean_hit={detail['clean_hit']}", flush=True)

    class Args:
        top_k = args.top_k
        max_records = len(records)

    summary = summarize(details, records, verifier, Args)
    summary.update(
        {
            "profile": args.profile,
            "label": args.label,
            "variant": args.variant,
            "index_size": args.index_size,
            "description": f"{args.label} index -> top-k image context -> Qwen3-VL generation",
        }
    )
    (profile_dir / "end_to_end_report.json").write_text(json.dumps({"summary": summary, "details": details}, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "end_to_end_profile_summary.json").write_text(json.dumps([summary], indent=2, ensure_ascii=False), encoding="utf-8")
    with (out_dir / "end_to_end_profile_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

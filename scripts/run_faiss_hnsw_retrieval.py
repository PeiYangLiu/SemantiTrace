#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run FAISS HNSW retrieval over cached CLIP image/query embeddings.")
    parser.add_argument("--cache_dir", default="outputs/pipeline_generality_flux_n500_textvqa_100k_clip")
    parser.add_argument("--output_dir", default="outputs/faiss_hnsw_100k_n500")
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--hnsw_m", type=int, default=32)
    parser.add_argument("--ef_search", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    import faiss

    args = parse_args()
    cache = ROOT / args.cache_dir
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    image_emb = np.load(cache / "clip_image_embeddings.npy").astype("float32")
    image_entries = json.loads((cache / "clip_image_entries.json").read_text(encoding="utf-8"))
    query_emb = np.load(cache / "clip_query_embeddings.npy").astype("float32")
    query_entries = json.loads((cache / "clip_query_entries.json").read_text(encoding="utf-8"))

    d = image_emb.shape[1]
    index = faiss.IndexHNSWFlat(d, args.hnsw_m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efSearch = args.ef_search
    index.add(image_emb)
    scores, ids = index.search(query_emb, args.top_k)

    target_by_record = {str(e.get("record_id")): i for i, e in enumerate(image_entries) if e.get("record_id")}
    detail_rows = []
    ranks = []
    for qi, q in enumerate(query_entries):
        target_idx = target_by_record[str(q["record_id"])]
        hit_positions = np.where(ids[qi] == target_idx)[0]
        rank = int(hit_positions[0] + 1) if hit_positions.size else args.top_k + 1
        ranks.append(rank)
        detail_rows.append(
            {
                "variant": "faiss_hnsw_clip_100k",
                "record_id": q["record_id"],
                "probe_index": q["probe_index"],
                "signature": q["signature"],
                "query": q["query"],
                "rank": rank,
                "target_retained": True,
                "returned_top_k": args.top_k,
            }
        )
    ranks_np = np.asarray(ranks, dtype=float)
    summary = {
        "variant": "faiss_hnsw_clip_100k",
        "num_records": len(target_by_record),
        "num_queries": len(query_entries),
        "index_size": len(image_entries),
        "distractors": len(image_entries) - len(target_by_record),
        "hnsw_m": args.hnsw_m,
        "ef_search": args.ef_search,
        "returned_top_k": args.top_k,
        "recall_at_1": float(np.mean(ranks_np <= 1)),
        "recall_at_3": float(np.mean(ranks_np <= 3)),
        "recall_at_10": float(np.mean(ranks_np <= 10)),
        "recall_at_50": float(np.mean(ranks_np <= 50)),
        "recall_at_100": float(np.mean(ranks_np <= 100)),
        "mrr_at_100": float(np.mean([1.0 / r if r <= args.top_k else 0.0 for r in ranks])),
        "mean_observed_rank_capped": float(ranks_np.mean()),
        "notes": "FAISS IndexHNSWFlat approximate vector index over cached OpenCLIP embeddings; ranks are capped at top_k+1 for misses.",
    }
    (out / "faiss_hnsw_details.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in detail_rows) + "\n",
        encoding="utf-8",
    )
    (out / "faiss_hnsw_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (out / "faiss_hnsw_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

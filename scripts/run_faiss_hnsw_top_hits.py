#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_end_to_end_profiles import build_fallback_index, resolve_anchor_path
from run_pipeline_generality import load_records, resolve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Export FAISS HNSW top-k hits for suspect and clean 100k profiles.")
    parser.add_argument("--records", default="outputs/semantitrace_n500_textvqa_merged/canary_records_first500_local.json")
    parser.add_argument("--cache_dir", default="outputs/pipeline_generality_flux_n500_textvqa_100k_clip")
    parser.add_argument("--clean_anchor_embeddings", default="outputs/end_to_end_profiles_100k_n500/visual_clip_100k/clean_anchor_embeddings.npy")
    parser.add_argument("--output_dir", default="outputs/faiss_hnsw_100k_e2e_hits")
    parser.add_argument("--max_records", type=int, default=500)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--hnsw_m", type=int, default=32)
    parser.add_argument("--ef_search", type=int, default=128)
    parser.add_argument(
        "--anchor_fallback_dirs",
        nargs="*",
        default=[
            "data_scene_text/total_text/images",
            "data_scene_text/coco_text/images",
            "data_webqa_5000/webqa/images",
            "data_expanded/mmqa/images",
            "data_expanded/webqa/images",
            "data_textvqa_ocr_shards/shard_0",
            "data_textvqa_ocr_shards/shard_1",
            "data_textvqa_ocr_shards/shard_2",
            "data_textvqa_ocr_shards/shard_3",
            "data/mmqa/images",
            "data/webqa/images",
        ],
    )
    return parser.parse_args()


def build_index(emb: np.ndarray, hnsw_m: int, ef_search: int):
    import faiss

    index = faiss.IndexHNSWFlat(emb.shape[1], hnsw_m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efSearch = ef_search
    index.add(np.asarray(emb, dtype="float32"))
    return index


def convert_hits(ids: np.ndarray, scores: np.ndarray, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hits = []
    for rank, (idx, score) in enumerate(zip(ids.tolist(), scores.tolist()), start=1):
        entry = entries[int(idx)]
        hits.append(
            {
                "image_id": entry["image_id"],
                "image_path": entry["path"],
                "role": entry["role"],
                "record_id": entry.get("record_id"),
                "entry_index": int(idx),
                "rank": rank,
                "score": float(score),
            }
        )
    return hits


def main() -> None:
    args = parse_args()
    out = resolve(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = load_records(resolve(args.records), args.max_records)
    cache = resolve(args.cache_dir)
    image_emb = np.load(cache / "clip_image_embeddings.npy").astype("float32")
    image_entries = json.loads((cache / "clip_image_entries.json").read_text(encoding="utf-8"))
    query_emb = np.load(cache / "clip_query_embeddings.npy").astype("float32")
    query_entries = json.loads((cache / "clip_query_entries.json").read_text(encoding="utf-8"))[: len(records) * 3]

    fallback = build_fallback_index(args.anchor_fallback_dirs)
    clean_anchor_emb = np.load(resolve(args.clean_anchor_embeddings)).astype("float32")[: len(records)]
    distractor_start = next(i for i, e in enumerate(image_entries) if e["role"] == "distractor")
    distractor_emb = image_emb[distractor_start:]
    distractor_entries = image_entries[distractor_start:]

    suspect_entries = image_entries[: len(records)] + distractor_entries
    suspect_emb = np.vstack([image_emb[: len(records)], distractor_emb])
    clean_entries = []
    for record in records:
        path = resolve_anchor_path(record, fallback)
        clean_entries.append({"image_id": str(record["id"]), "path": str(path), "role": "clean_control", "record_id": str(record["id"])})
    clean_entries.extend(distractor_entries)
    clean_emb = np.vstack([clean_anchor_emb, distractor_emb])

    suspect_index = build_index(suspect_emb, args.hnsw_m, args.ef_search)
    clean_index = build_index(clean_emb, args.hnsw_m, args.ef_search)
    wm_scores, wm_ids = suspect_index.search(query_emb[: len(query_entries)], args.top_k)
    clean_scores, clean_ids = clean_index.search(query_emb[: len(query_entries)], args.top_k)

    rows = []
    for qi, q in enumerate(query_entries):
        rid = str(q["record_id"])
        target_idx = int(rid.split("-")[-1])
        wm_pos = np.where(wm_ids[qi] == target_idx)[0]
        clean_pos = np.where(clean_ids[qi] == target_idx)[0]
        rows.append(
            {
                "profile": "faiss_hnsw_100k",
                "record_index": target_idx,
                "probe_index": int(q["probe_index"]),
                "id": rid,
                "signature": q["signature"],
                "query": q["query"],
                "target_rank": int(wm_pos[0] + 1) if wm_pos.size else args.top_k + 1,
                "clean_target_rank": int(clean_pos[0] + 1) if clean_pos.size else args.top_k + 1,
                "watermarked_hits": convert_hits(wm_ids[qi], wm_scores[qi], suspect_entries),
                "clean_hits": convert_hits(clean_ids[qi], clean_scores[qi], clean_entries),
            }
        )
    (out / "faiss_hnsw_top_hits.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    summary = {
        "profile": "faiss_hnsw_100k",
        "num_canaries": len(records),
        "num_queries": len(rows),
        "top_k": args.top_k,
        "hnsw_m": args.hnsw_m,
        "ef_search": args.ef_search,
    }
    (out / "faiss_hnsw_top_hits_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

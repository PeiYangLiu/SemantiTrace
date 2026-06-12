#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
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
    parser = argparse.ArgumentParser("Export ChromaDB top-k hits for suspect and clean 100k profiles.")
    parser.add_argument("--records", default="outputs/semantitrace_n500_textvqa_merged/canary_records_first500_local.json")
    parser.add_argument("--cache_dir", default="outputs/pipeline_generality_flux_n500_textvqa_100k_clip")
    parser.add_argument("--clean_anchor_embeddings", default="outputs/end_to_end_profiles_100k_n500/visual_clip_100k/clean_anchor_embeddings.npy")
    parser.add_argument("--output_dir", default="outputs/chromadb_100k_e2e_hits")
    parser.add_argument("--max_records", type=int, default=500)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=5000)
    parser.add_argument("--fresh", action="store_true")
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


def add_collection(collection, entries: list[dict[str, Any]], emb: np.ndarray, batch_size: int) -> None:
    total = len(entries)
    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        ids = [f"idx-{i:06d}" for i in range(start, end)]
        metas = []
        docs = []
        for i in range(start, end):
            e = entries[i]
            metas.append(
                {
                    "entry_index": i,
                    "image_id": str(e["image_id"]),
                    "image_path": str(e["path"]),
                    "role": str(e["role"]),
                    "record_id": str(e.get("record_id") or ""),
                }
            )
            docs.append(str(e["image_id"]))
        collection.add(ids=ids, embeddings=emb[start:end].astype("float32").tolist(), metadatas=metas, documents=docs)
        print(f"added {end}/{total} to {collection.name}", flush=True)


def to_hits(result: dict[str, Any]) -> list[dict[str, Any]]:
    hits = []
    metas = result["metadatas"][0]
    distances = result["distances"][0]
    for rank, (meta, dist) in enumerate(zip(metas, distances), start=1):
        hits.append(
            {
                "image_id": meta["image_id"],
                "image_path": meta["image_path"],
                "role": meta["role"],
                "record_id": meta.get("record_id") or None,
                "entry_index": int(meta["entry_index"]),
                "rank": rank,
                "distance": float(dist),
            }
        )
    return hits


def main() -> None:
    import chromadb

    args = parse_args()
    out = resolve(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    persist = out / "chroma_store"
    if args.fresh and persist.exists():
        shutil.rmtree(persist)

    records = load_records(resolve(args.records), args.max_records)
    cache = resolve(args.cache_dir)
    image_emb = np.load(cache / "clip_image_embeddings.npy").astype("float32")
    image_entries = json.loads((cache / "clip_image_entries.json").read_text(encoding="utf-8"))
    query_emb = np.load(cache / "clip_query_embeddings.npy").astype("float32")[: len(records) * 3]
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

    client = chromadb.PersistentClient(path=str(persist))
    suspect = client.get_or_create_collection("suspect_clip", metadata={"hnsw:space": "cosine"})
    clean = client.get_or_create_collection("clean_clip", metadata={"hnsw:space": "cosine"})
    if suspect.count() < len(suspect_entries):
        add_collection(suspect, suspect_entries, suspect_emb, args.batch_size)
    if clean.count() < len(clean_entries):
        add_collection(clean, clean_entries, clean_emb, args.batch_size)

    rows = []
    for qi, q in enumerate(query_entries):
        wm = suspect.query(query_embeddings=[query_emb[qi].astype("float32").tolist()], n_results=args.top_k, include=["metadatas", "distances"])
        cl = clean.query(query_embeddings=[query_emb[qi].astype("float32").tolist()], n_results=args.top_k, include=["metadatas", "distances"])
        rid = str(q["record_id"])
        rows.append(
            {
                "profile": "chromadb_100k",
                "record_index": int(rid.split("-")[-1]),
                "probe_index": int(q["probe_index"]),
                "id": rid,
                "signature": q["signature"],
                "query": q["query"],
                "target_rank": next((h["rank"] for h in to_hits(wm) if h.get("record_id") == rid), args.top_k + 1),
                "clean_target_rank": next((h["rank"] for h in to_hits(cl) if h.get("record_id") == rid), args.top_k + 1),
                "watermarked_hits": to_hits(wm),
                "clean_hits": to_hits(cl),
            }
        )
        if (qi + 1) % 100 == 0:
            print(f"queried {qi + 1}/{len(query_entries)}", flush=True)
    (out / "chromadb_top_hits.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    summary = {"profile": "chromadb_100k", "num_canaries": len(records), "num_queries": len(rows), "top_k": args.top_k}
    (out / "chromadb_top_hits_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

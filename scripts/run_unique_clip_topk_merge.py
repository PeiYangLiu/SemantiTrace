#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Merge exact CLIP top-k shards into E2E-ready hit rows.")
    p.add_argument("--shards_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--variant", default="clip_1m_unique")
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--top_ks", default="1,3,5,10,20,50,100,1000")
    return p.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def select_top(candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        key = str(candidate.get("image_path"))
        if not key:
            continue
        if key not in deduped or float(candidate.get("score", -np.inf)) > float(deduped[key].get("score", -np.inf)):
            deduped[key] = candidate
    ranked = sorted(deduped.values(), key=lambda row: float(row.get("score", -np.inf)), reverse=True)
    return ranked[:top_k]


def safe_name(path: Path) -> str:
    stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in path.stem)[:80]
    suffix = path.suffix.lower() if path.suffix else ".jpg"
    return f"{stem}{suffix}"


def materialize_hits(hits: list[dict[str, Any]], out_dir: Path, query_index: int, kind: str) -> list[dict[str, Any]]:
    materialized: list[dict[str, Any]] = []
    hit_dir = out_dir / "hit_images" / kind
    hit_dir.mkdir(parents=True, exist_ok=True)
    for rank, hit in enumerate(hits, start=1):
        src = Path(str(hit["image_path"]))
        if not src.is_file():
            raise FileNotFoundError(f"Top-k hit image is missing: {src}")
        dst = hit_dir / f"q{query_index:05d}_r{rank}_{safe_name(src)}"
        if not dst.exists():
            shutil.copy2(src, dst)
        row = dict(hit)
        row["rank"] = rank
        row["source_image_path"] = str(src)
        row["image_path"] = str(dst)
        materialized.append(row)
    return materialized


def main() -> None:
    args = parse_args()
    shards_dir = Path(args.shards_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    shard_dirs = sorted(p for p in shards_dir.glob("shard_*") if p.is_dir())
    if not shard_dirs:
        raise FileNotFoundError(f"No shard_* dirs under {shards_dir}")

    counts_suspect = None
    counts_clean = None
    comp_suspect = None
    comp_clean = None
    query_entries = None
    canary_top_suspect = None
    canary_top_clean = None
    distractor_tops: list[list[list[dict[str, Any]]]] = []
    summaries = []
    total_distractors = 0
    for shard in shard_dirs:
        c_s = np.load(shard / "distractor_counts_suspect.npy").astype("int64")
        c_c = np.load(shard / "distractor_counts_clean.npy").astype("int64")
        counts_suspect = c_s if counts_suspect is None else counts_suspect + c_s
        counts_clean = c_c if counts_clean is None else counts_clean + c_c

        shard_comp_s = np.load(shard / "canary_competitor_counts_suspect.npy").astype("int64")
        shard_comp_c = np.load(shard / "canary_competitor_counts_clean.npy").astype("int64")
        if comp_suspect is None:
            comp_suspect = shard_comp_s
            comp_clean = shard_comp_c
        elif not np.array_equal(comp_suspect, shard_comp_s) or not np.array_equal(comp_clean, shard_comp_c):
            raise ValueError(f"Canary competitor counts differ in {shard}")

        shard_queries = load_json(shard / "query_entries.json")
        if query_entries is None:
            query_entries = shard_queries
            canary_top_suspect = load_json(shard / "canary_top_suspect.json")
            canary_top_clean = load_json(shard / "canary_top_clean.json")
        elif query_entries != shard_queries:
            raise ValueError(f"Query entries differ in {shard}")
        distractor_tops.append(load_json(shard / "distractor_top.json"))
        summary = load_json(shard / "shard_summary.json")
        summaries.append(summary)
        total_distractors += int(summary["distractors_processed"])

    assert counts_suspect is not None and counts_clean is not None
    assert comp_suspect is not None and comp_clean is not None
    assert query_entries is not None and canary_top_suspect is not None and canary_top_clean is not None

    ranks_suspect = counts_suspect + comp_suspect + 1
    ranks_clean = counts_clean + comp_clean + 1
    rows = []
    for qi, query in enumerate(query_entries):
        wm_candidates: list[dict[str, Any]] = list(canary_top_suspect[qi])
        clean_candidates: list[dict[str, Any]] = list(canary_top_clean[qi])
        for shard_top in distractor_tops:
            wm_candidates.extend(shard_top[qi])
            clean_candidates.extend(shard_top[qi])
        watermarked_hits = materialize_hits(select_top(wm_candidates, args.top_k), out, qi, "watermarked")
        clean_hits = materialize_hits(select_top(clean_candidates, args.top_k), out, qi, "clean")
        row = dict(query)
        row.update(
            {
                "profile": args.variant,
                "variant": args.variant,
                "target_rank": int(ranks_suspect[qi]),
                "clean_target_rank": int(ranks_clean[qi]),
                "watermarked_hits": watermarked_hits,
                "clean_hits": clean_hits,
            }
        )
        rows.append(row)

    top_ks = [int(k) for k in args.top_ks.split(",") if k.strip()]
    ranks_np = ranks_suspect.astype("float64")
    clean_ranks_np = ranks_clean.astype("float64")
    total_records = len({row["record_id"] for row in query_entries})
    global_index_size = int(total_records + total_distractors)

    def make_summary(subset: str, indices: np.ndarray) -> dict[str, Any]:
        subset_rows = [query_entries[int(i)] for i in indices.tolist()]
        subset_ranks = ranks_np[indices]
        subset_clean = clean_ranks_np[indices]
        summary = {
            "variant": args.variant,
            "subset": subset,
            "num_records": len({row["record_id"] for row in subset_rows}),
            "num_queries": len(subset_rows),
            "top_k": args.top_k,
            "index_size": global_index_size,
            "distractors": int(total_distractors),
            "mean_rank": float(subset_ranks.mean()),
            "median_rank": float(np.median(subset_ranks)),
            "mrr": float(np.mean(1.0 / subset_ranks)),
            "clean_mean_rank": float(subset_clean.mean()),
            "shard_summaries": summaries if subset == "all" else [],
        }
        for k in top_ks:
            summary[f"recall_at_{k}"] = float(np.mean(subset_ranks <= k))
            summary[f"clean_recall_at_{k}"] = float(np.mean(subset_clean <= k))
        return summary

    all_indices = np.arange(len(query_entries), dtype=np.int64)
    summary_rows = [make_summary("all", all_indices)]
    for mode in sorted({str(row.get("mode", "unknown")) for row in query_entries}):
        mode_indices = np.asarray(
            [idx for idx, row in enumerate(query_entries) if str(row.get("mode", "unknown")) == mode],
            dtype=np.int64,
        )
        if len(mode_indices):
            summary_rows.append(make_summary(mode, mode_indices))

    (out / "unique_topk_hits.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    (out / "unique_topk_summary.json").write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    with (out / "unique_topk_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        fieldnames = list(summary_rows[0].keys())
        for row in summary_rows[1:]:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(json.dumps(summary_rows, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

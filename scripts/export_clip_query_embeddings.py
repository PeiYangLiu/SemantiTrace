#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_large_clip_retrieval import encode_texts
from run_pipeline_generality import load_records, resolve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Export OpenCLIP text query embeddings for cached retrieval experiments.")
    parser.add_argument("--records", default="outputs/semantitrace_n500_textvqa_merged/canary_records_first500_local.json")
    parser.add_argument("--output_dir", default="outputs/pipeline_generality_flux_n500_textvqa_100k_clip")
    parser.add_argument("--max_records", type=int, default=500)
    parser.add_argument("--num_queries", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = resolve(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = load_records(resolve(args.records), args.max_records)
    rows = []
    queries = []
    for record in records:
        rid = str(record["id"])
        signature = str(record["trap_signature"])
        for probe_index, query in enumerate(record.get("probe_queries", [])[: args.num_queries]):
            rows.append({"record_id": rid, "probe_index": probe_index, "signature": signature, "query": str(query)})
            queries.append(str(query))
    emb = encode_texts(queries, args)
    import numpy as np

    np.save(out / "clip_query_embeddings.npy", emb.astype("float32"))
    (out / "clip_query_entries.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"queries": len(rows), "output_dir": str(out)}, indent=2))


if __name__ == "__main__":
    main()

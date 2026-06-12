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

from run_pipeline_generality import IndexEntry, collect_distractors, load_records, resolve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run large-scale CLIP visual retrieval with cached embeddings.")
    parser.add_argument("--records", default="outputs/semantitrace_n500_textvqa_merged/canary_records_first500_local.json")
    parser.add_argument("--output_dir", default="outputs/pipeline_generality_flux_n500_textvqa_100k_clip")
    parser.add_argument("--distractor_dirs", nargs="+", required=True)
    parser.add_argument("--max_records", type=int, default=500)
    parser.add_argument("--max_distractors", type=int, default=100000)
    parser.add_argument("--num_queries", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top_ks", default="1,3,5,10,20,50,100")
    parser.add_argument("--variant", default="clip_visual_100k")
    parser.add_argument("--exclude_anchor_images", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    return parser.parse_args()


def build_entries(records: list[dict[str, Any]], distractors: list[Path]) -> list[IndexEntry]:
    entries: list[IndexEntry] = []
    for record in records:
        entries.append(
            IndexEntry(
                image_id=str(record["id"]),
                image_path=resolve(record["watermarked_image_path"]),
                role="canary",
                record_id=str(record["id"]),
            )
        )
    for i, path in enumerate(distractors):
        entries.append(IndexEntry(image_id=f"distractor-{i:06d}", image_path=path, role="distractor"))
    return entries


def encode_images(entries: list[IndexEntry], out_dir: Path, args: argparse.Namespace) -> np.ndarray:
    cache = out_dir / "clip_image_embeddings.npy"
    meta = out_dir / "clip_image_entries.json"
    if cache.exists() and meta.exists() and not args.fresh:
        return np.load(cache, mmap_mode="r")

    import open_clip
    import torch
    from torch.utils.data import DataLoader, Dataset

    class ImageDataset(Dataset):
        def __init__(self, entries: list[IndexEntry], preprocess):
            self.entries = entries
            self.preprocess = preprocess

        def __len__(self) -> int:
            return len(self.entries)

        def __getitem__(self, idx: int):
            entry = self.entries[idx]
            image = Image.open(entry.image_path).convert("RGB")
            return self.preprocess(image), idx

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", device=device)
    model.eval()
    loader = DataLoader(
        ImageDataset(entries, preprocess),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        multiprocessing_context="fork" if args.num_workers > 0 else None,
    )
    all_features = np.memmap(out_dir / "clip_image_embeddings.tmp", dtype="float32", mode="w+", shape=(len(entries), 768))
    done = 0
    with torch.no_grad():
        for pixels, indices in loader:
            pixels = pixels.to(device, non_blocking=True)
            emb = model.encode_image(pixels)
            emb = torch.nn.functional.normalize(emb.float(), dim=-1).cpu().numpy()
            all_features[np.asarray(indices, dtype=np.int64)] = emb
            done += len(indices)
            if done % 10000 < args.batch_size:
                print(f"encoded {done}/{len(entries)} images", flush=True)
    all_features.flush()
    np.save(cache, np.asarray(all_features))
    (out_dir / "clip_image_embeddings.tmp").unlink(missing_ok=True)
    meta.write_text(
        json.dumps(
            [
                {
                    "image_id": e.image_id,
                    "path": str(e.image_path),
                    "role": e.role,
                    "record_id": e.record_id,
                }
                for e in entries
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    return np.load(cache, mmap_mode="r")


def encode_texts(texts: list[str], args: argparse.Namespace) -> np.ndarray:
    import open_clip
    import torch

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", device=device)
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    model.eval()
    feats = []
    with torch.no_grad():
        for start in range(0, len(texts), args.batch_size):
            tokens = tokenizer(texts[start : start + args.batch_size]).to(device)
            emb = model.encode_text(tokens)
            emb = torch.nn.functional.normalize(emb.float(), dim=-1)
            feats.append(emb.cpu().numpy())
    return np.vstack(feats)


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(resolve(args.records), args.max_records)
    exclude = {resolve(record["watermarked_image_path"]) for record in records}
    if args.exclude_anchor_images:
        exclude |= {resolve(record["anchor_image_path"]) for record in records if record.get("anchor_image_path")}
    distractors = collect_distractors(args.distractor_dirs, args.max_distractors, exclude)
    entries = build_entries(records, distractors)
    target_index = {entry.record_id: idx for idx, entry in enumerate(entries) if entry.record_id}
    image_emb = encode_images(entries, out_dir, args)

    queries: list[tuple[str, int, str, str]] = []
    for record in records:
        rid = str(record["id"])
        signature = str(record["trap_signature"])
        for probe_index, query in enumerate(record.get("probe_queries", [])[: args.num_queries]):
            queries.append((rid, probe_index, signature, str(query)))
    text_emb = encode_texts([q[-1] for q in queries], args)

    details: list[dict[str, Any]] = []
    top_ks = [int(k) for k in args.top_ks.split(",") if k.strip()]
    ranks: list[int] = []
    for idx, (rid, probe_index, signature, query) in enumerate(queries):
        scores = np.asarray(image_emb @ text_emb[idx], dtype=np.float32)
        target_idx = target_index[rid]
        rank = int(np.count_nonzero(scores > scores[target_idx]) + 1)
        ranks.append(rank)
        details.append(
            {
                "variant": args.variant,
                "record_id": rid,
                "probe_index": probe_index,
                "signature": signature,
                "query": query,
                "rank": rank,
                "target_retained": True,
            }
        )
    ranks_np = np.asarray(ranks, dtype=np.float64)
    summary = {
        "variant": args.variant,
        "num_records": len(records),
        "num_queries": len(queries),
        "index_size": len(entries),
        "distractors": len(distractors),
        "target_retention": 1.0,
        "mean_rank": float(ranks_np.mean()),
        "median_rank": float(np.median(ranks_np)),
        "mrr": float(np.mean(1.0 / ranks_np)),
    }
    for k in top_ks:
        summary[f"recall_at_{k}"] = float(np.mean(ranks_np <= k))

    (out_dir / "pipeline_generality_details.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in details) + "\n",
        encoding="utf-8",
    )
    (out_dir / "pipeline_generality_summary.json").write_text(json.dumps([summary], indent=2), encoding="utf-8")
    with (out_dir / "pipeline_generality_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from semantitrace.records import infer_record_mode, load_records_with_resolved_paths, resolve_repo_path

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Compute one CLIP retrieval-count shard for Mode-B n500 over unique distractors.")
    p.add_argument("--records", required=True)
    p.add_argument(
        "--record_root",
        default=None,
        help="Optional bundle root used to resolve relative image paths stored in --records.",
    )
    p.add_argument("--distractor_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--file_shard_count", type=int, default=1)
    p.add_argument("--file_shard_index", type=int, default=0)
    p.add_argument("--max_records", type=int, default=500)
    p.add_argument("--num_queries", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def resolve(path: str | Path) -> Path:
    return resolve_repo_path(path)


def list_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)


def encode_images(paths: list[Path], model: Any, preprocess: Any, args: argparse.Namespace, device: Any) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader, Dataset

    class ImageDataset(Dataset):
        def __init__(self, paths: list[Path]):
            self.paths = paths

        def __len__(self) -> int:
            return len(self.paths)

        def __getitem__(self, idx: int):
            return preprocess(Image.open(self.paths[idx]).convert("RGB")), idx

    loader = DataLoader(
        ImageDataset(paths),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        multiprocessing_context="fork" if args.num_workers > 0 else None,
    )
    arr = np.empty((len(paths), 768), dtype="float32")
    done = 0
    with torch.no_grad():
        for pixels, indices in loader:
            pixels = pixels.to(device, non_blocking=True)
            emb = model.encode_image(pixels)
            emb = torch.nn.functional.normalize(emb.float(), dim=-1).cpu().numpy()
            arr[np.asarray(indices, dtype=np.int64)] = emb
            done += len(indices)
            if done % 10000 < args.batch_size:
                print(f"encoded_images {done}/{len(paths)}", flush=True)
    return arr


def main() -> None:
    import open_clip
    import torch

    args = parse_args()
    out = resolve(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    required_outputs = [
        out / "distractor_rank_counts.npy",
        out / "target_scores.npy",
        out / "canary_competitor_counts.npy",
        out / "query_entries.json",
        out / "shard_summary.json",
    ]
    if all(path.exists() for path in required_outputs):
        print(f"[resume] shard outputs already exist in {out}; skipping", flush=True)
        return
    records = load_records_with_resolved_paths(args.records, args.max_records, record_root=args.record_root)

    query_entries: list[dict[str, Any]] = []
    texts: list[str] = []
    target_indices: list[int] = []
    for ridx, record in enumerate(records):
        for probe_index, query in enumerate(record.get("probe_queries", [])[: args.num_queries]):
            query_entries.append(
                {
                    "record_id": str(record["id"]),
                    "record_index": ridx,
                    "probe_index": probe_index,
                    "mode": infer_record_mode(record),
                    "parasitism_mode": str(record.get("parasitism_mode", "")),
                    "signature": str(record.get("trap_signature", "")),
                    "query": str(query),
                }
            )
            texts.append(str(query))
            target_indices.append(ridx)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", device=device)
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    model.eval()

    canary_paths = [Path(r["_resolved_watermarked_image_path"]) for r in records]
    canary_emb = encode_images(canary_paths, model, preprocess, args, device)

    q_feats = []
    with torch.no_grad():
        for start in range(0, len(texts), args.batch_size):
            tokens = tokenizer(texts[start : start + args.batch_size]).to(device)
            emb = model.encode_text(tokens)
            emb = torch.nn.functional.normalize(emb.float(), dim=-1)
            q_feats.append(emb.cpu().numpy())
    query_emb = np.vstack(q_feats).astype("float32")
    target_indices_np = np.asarray(target_indices, dtype=np.int64)
    target_scores = np.sum(query_emb * canary_emb[target_indices_np], axis=1).astype("float32")
    canary_scores = query_emb @ canary_emb.T
    canary_competitor_counts = np.sum(canary_scores > target_scores[:, None], axis=1).astype("int64")

    all_distractors = list_images(resolve(args.distractor_dir))
    distractors = all_distractors[args.file_shard_index :: args.file_shard_count]
    print(
        f"file_shard {args.file_shard_index}/{args.file_shard_count}: "
        f"{len(distractors)} of {len(all_distractors)} distractors",
        flush=True,
    )
    counts = np.zeros(len(query_entries), dtype="int64")
    chunk_paths: list[Path] = []
    processed = 0
    for path in distractors:
        chunk_paths.append(path)
        if len(chunk_paths) < args.batch_size * 32:
            continue
        emb = encode_images(chunk_paths, model, preprocess, args, device)
        counts += np.sum(query_emb @ emb.T > target_scores[:, None], axis=1).astype("int64")
        processed += len(chunk_paths)
        print(f"processed_distractors {processed}/{len(distractors)}", flush=True)
        chunk_paths = []
    if chunk_paths:
        emb = encode_images(chunk_paths, model, preprocess, args, device)
        counts += np.sum(query_emb @ emb.T > target_scores[:, None], axis=1).astype("int64")
        processed += len(chunk_paths)

    np.save(out / "distractor_rank_counts.npy", counts)
    np.save(out / "target_scores.npy", target_scores)
    np.save(out / "canary_competitor_counts.npy", canary_competitor_counts)
    (out / "query_entries.json").write_text(json.dumps(query_entries, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "records": len(records),
        "queries": len(query_entries),
        "distractors_processed": len(distractors),
        "file_shard_count": args.file_shard_count,
        "file_shard_index": args.file_shard_index,
    }
    (out / "shard_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

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
    p = argparse.ArgumentParser("Compute one exact CLIP top-k shard over a 1M unique-image pool.")
    p.add_argument("--records", required=True)
    p.add_argument("--record_root", default=None)
    p.add_argument("--distractor_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--file_shard_count", type=int, default=1)
    p.add_argument("--file_shard_index", type=int, default=0)
    p.add_argument("--max_records", type=int, default=500)
    p.add_argument("--num_queries", type=int, default=3)
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--clean_index_includes_anchor",
        action="store_true",
        help=(
            "If set, the clean index also contains the unmodified anchor image for each canary "
            "(legacy behavior). Default: clean index contains distractors only, so the clean "
            "baseline measures the chance that distractor-only retrieval looks like a canary."
        ),
    )
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


def topk_indices(scores: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    k = min(top_k, scores.shape[1])
    idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    row = np.arange(scores.shape[0])[:, None]
    selected_scores = scores[row, idx]
    order = np.argsort(-selected_scores, axis=1)
    return np.take_along_axis(idx, order, axis=1), np.take_along_axis(selected_scores, order, axis=1)


def candidates_from_matrix(scores: np.ndarray, candidates: list[dict[str, Any]], top_k: int) -> list[list[dict[str, Any]]]:
    idx, selected_scores = topk_indices(scores, top_k)
    rows: list[list[dict[str, Any]]] = []
    for qi in range(scores.shape[0]):
        row: list[dict[str, Any]] = []
        for rank_idx in range(idx.shape[1]):
            candidate = dict(candidates[int(idx[qi, rank_idx])])
            candidate["score"] = float(selected_scores[qi, rank_idx])
            row.append(candidate)
        rows.append(row)
    return rows


def update_topk(
    top_scores: np.ndarray,
    top_candidates: list[list[dict[str, Any]]],
    scores: np.ndarray,
    candidates: list[dict[str, Any]],
    top_k: int,
) -> tuple[np.ndarray, list[list[dict[str, Any]]]]:
    idx, selected_scores = topk_indices(scores, top_k)
    for qi in range(scores.shape[0]):
        merged = [
            (float(top_scores[qi, rank_idx]), dict(top_candidates[qi][rank_idx]))
            for rank_idx in range(len(top_candidates[qi]))
        ]
        for rank_idx in range(idx.shape[1]):
            candidate = dict(candidates[int(idx[qi, rank_idx])])
            merged.append((float(selected_scores[qi, rank_idx]), candidate))
        merged.sort(key=lambda item: item[0], reverse=True)
        top_candidates[qi] = []
        for rank_idx, (score, candidate) in enumerate(merged[:top_k]):
            top_scores[qi, rank_idx] = score
            candidate["score"] = score
            top_candidates[qi].append(candidate)
    return top_scores, top_candidates


def main() -> None:
    import open_clip
    import torch

    args = parse_args()
    out = resolve(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    required_outputs = [
        out / "distractor_counts_suspect.npy",
        out / "distractor_counts_clean.npy",
        out / "canary_competitor_counts_suspect.npy",
        out / "canary_competitor_counts_clean.npy",
        out / "query_entries.json",
        out / "canary_top_suspect.json",
        out / "canary_top_clean.json",
        out / "distractor_top.json",
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
                    "id": str(record["id"]),
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
    if args.clean_index_includes_anchor:
        clean_paths = [Path(r["_resolved_anchor_image_path"]) for r in records]
        clean_emb = encode_images(clean_paths, model, preprocess, args, device)
    else:
        clean_paths = []
        clean_emb = None

    q_feats = []
    with torch.no_grad():
        for start in range(0, len(texts), args.batch_size):
            tokens = tokenizer(texts[start : start + args.batch_size]).to(device)
            emb = model.encode_text(tokens)
            emb = torch.nn.functional.normalize(emb.float(), dim=-1)
            q_feats.append(emb.cpu().numpy())
    query_emb = np.vstack(q_feats).astype("float32")
    target_indices_np = np.asarray(target_indices, dtype=np.int64)
    target_scores_suspect = np.sum(query_emb * canary_emb[target_indices_np], axis=1).astype("float32")
    if clean_emb is not None:
        target_scores_clean = np.sum(query_emb * clean_emb[target_indices_np], axis=1).astype("float32")
    else:
        # No paired clean targets are present in the clean index. This makes the
        # clean target rank fall outside the index and leaves clean top-k context
        # to be populated only by distractors.
        target_scores_clean = np.full(len(query_entries), -np.inf, dtype="float32")

    canary_scores = query_emb @ canary_emb.T
    canary_competitor_counts = np.sum(canary_scores > target_scores_suspect[:, None], axis=1).astype("int64")
    if clean_emb is not None:
        clean_scores = query_emb @ clean_emb.T
        clean_competitor_counts = np.sum(clean_scores > target_scores_clean[:, None], axis=1).astype("int64")
    else:
        clean_scores = None
        clean_competitor_counts = np.zeros(len(query_entries), dtype="int64")
    canary_candidates = [
        {
            "image_id": str(record["id"]),
            "image_path": str(path),
            "role": "canary",
            "record_id": str(record["id"]),
            "entry_index": idx,
        }
        for idx, (record, path) in enumerate(zip(records, canary_paths))
    ]
    clean_candidates = [
        {
            "image_id": f"{record['id']}::clean",
            "image_path": str(path),
            "role": "clean_control",
            "record_id": str(record["id"]),
            "entry_index": idx,
        }
        for idx, (record, path) in enumerate(zip(records, clean_paths))
    ]
    canary_top = candidates_from_matrix(canary_scores, canary_candidates, args.top_k)
    clean_top = (
        candidates_from_matrix(clean_scores, clean_candidates, args.top_k)
        if clean_scores is not None
        else [[] for _ in query_entries]
    )

    all_distractors = list_images(resolve(args.distractor_dir))
    distractors = all_distractors[args.file_shard_index :: args.file_shard_count]
    print(
        f"file_shard {args.file_shard_index}/{args.file_shard_count}: "
        f"{len(distractors)} of {len(all_distractors)} distractors",
        flush=True,
    )
    distractor_counts_suspect = np.zeros(len(query_entries), dtype="int64")
    distractor_counts_clean = np.zeros(len(query_entries), dtype="int64")
    top_scores = np.full((len(query_entries), args.top_k), -np.inf, dtype="float32")
    top_candidates: list[list[dict[str, Any]]] = [[] for _ in query_entries]

    chunk_paths: list[Path] = []
    processed = 0
    for path in distractors:
        chunk_paths.append(path)
        if len(chunk_paths) < args.batch_size * 32:
            continue
        emb = encode_images(chunk_paths, model, preprocess, args, device)
        scores = query_emb @ emb.T
        distractor_counts_suspect += np.sum(scores > target_scores_suspect[:, None], axis=1).astype("int64")
        distractor_counts_clean += np.sum(scores > target_scores_clean[:, None], axis=1).astype("int64")
        candidates = [
            {
                "image_id": f"distractor-{processed + idx:08d}",
                "image_path": str(path),
                "role": "distractor",
                "record_id": None,
                "entry_index": processed + idx,
            }
            for idx, path in enumerate(chunk_paths)
        ]
        top_scores, top_candidates = update_topk(top_scores, top_candidates, scores, candidates, args.top_k)
        processed += len(chunk_paths)
        print(f"processed_distractors {processed}/{len(distractors)}", flush=True)
        chunk_paths = []
    if chunk_paths:
        emb = encode_images(chunk_paths, model, preprocess, args, device)
        scores = query_emb @ emb.T
        distractor_counts_suspect += np.sum(scores > target_scores_suspect[:, None], axis=1).astype("int64")
        distractor_counts_clean += np.sum(scores > target_scores_clean[:, None], axis=1).astype("int64")
        candidates = [
            {
                "image_id": f"distractor-{processed + idx:08d}",
                "image_path": str(path),
                "role": "distractor",
                "record_id": None,
                "entry_index": processed + idx,
            }
            for idx, path in enumerate(chunk_paths)
        ]
        top_scores, top_candidates = update_topk(top_scores, top_candidates, scores, candidates, args.top_k)
        processed += len(chunk_paths)

    np.save(out / "distractor_counts_suspect.npy", distractor_counts_suspect)
    np.save(out / "distractor_counts_clean.npy", distractor_counts_clean)
    np.save(out / "canary_competitor_counts_suspect.npy", canary_competitor_counts)
    np.save(out / "canary_competitor_counts_clean.npy", clean_competitor_counts)
    (out / "query_entries.json").write_text(json.dumps(query_entries, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "canary_top_suspect.json").write_text(json.dumps(canary_top, ensure_ascii=False), encoding="utf-8")
    (out / "canary_top_clean.json").write_text(json.dumps(clean_top, ensure_ascii=False), encoding="utf-8")
    (out / "distractor_top.json").write_text(json.dumps(top_candidates, ensure_ascii=False), encoding="utf-8")
    summary = {
        "records": len(records),
        "queries": len(query_entries),
        "top_k": args.top_k,
        "distractors_processed": len(distractors),
        "file_shard_count": args.file_shard_count,
        "file_shard_index": args.file_shard_index,
        "clean_index_includes_anchor": bool(args.clean_index_includes_anchor),
    }
    (out / "shard_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

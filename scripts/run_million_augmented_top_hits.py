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
sys.path.insert(0, str(ROOT / "scripts"))

from run_end_to_end_profiles import build_fallback_index, resolve_anchor_path
from run_million_augmented_clip_retrieval import AugmentedImageDataset, apply_variant
from run_pipeline_generality import load_records, resolve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Export exact top-k hits for 1M augmented CLIP retrieval, suspect and clean.")
    parser.add_argument("--records", default="outputs/semantitrace_n500_textvqa_merged/canary_records_first500_local.json")
    parser.add_argument("--cache_dir", default="outputs/pipeline_generality_flux_n500_textvqa_100k_clip")
    parser.add_argument("--clean_anchor_embeddings", default="outputs/end_to_end_profiles_100k_n500/visual_clip_100k/clean_anchor_embeddings.npy")
    parser.add_argument("--output_dir", default="outputs/million_augmented_100k_e2e_hits")
    parser.add_argument("--max_records", type=int, default=500)
    parser.add_argument("--base_distractors", type=int, default=100000)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--variants", default="hflip,center_crop_90,crop_tl_88,crop_br_88,bright_90_contrast_110,bright_110_contrast_90,color_80,blur_06,rotate_2")
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


def update_topk(top_scores: np.ndarray, top_codes: np.ndarray, scores: np.ndarray, codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    combined_scores = np.concatenate([top_scores, scores], axis=1)
    combined_codes = np.concatenate([top_codes, np.broadcast_to(codes[None, :], scores.shape)], axis=1)
    k = top_scores.shape[1]
    idx = np.argpartition(-combined_scores, kth=k - 1, axis=1)[:, :k]
    row = np.arange(combined_scores.shape[0])[:, None]
    sel_scores = combined_scores[row, idx]
    sel_codes = combined_codes[row, idx]
    order = np.argsort(-sel_scores, axis=1)
    return np.take_along_axis(sel_scores, order, axis=1), np.take_along_axis(sel_codes, order, axis=1)


def init_entries(records: list[dict[str, Any]], cache_entries: list[dict[str, Any]], clean_paths: list[Path], max_records: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    distractor_start = next(i for i, e in enumerate(cache_entries) if e.get("role") == "distractor")
    distractors = cache_entries[distractor_start:]
    suspect = cache_entries[:max_records] + distractors
    clean = [
        {"image_id": str(record["id"]), "path": str(path), "role": "clean_control", "record_id": str(record["id"])}
        for record, path in zip(records, clean_paths)
    ] + distractors
    return suspect, clean, distractor_start


def materialize_augmented(code: int, initial_len: int, base_paths: list[str], variants: list[str], out_dir: Path) -> Path:
    aug_idx = code - initial_len
    base_idx = aug_idx // len(variants)
    var_idx = aug_idx % len(variants)
    variant = variants[var_idx]
    out = out_dir / "augmented_hits" / f"aug-{base_idx:06d}-{variant}.jpg"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        image = Image.open(base_paths[base_idx]).convert("RGB")
        apply_variant(image, variant).save(out, quality=90, optimize=True)
    return out


def code_to_hit(code: int, score: float, rank: int, initial_entries: list[dict[str, Any]], initial_len: int, base_paths: list[str], variants: list[str], out_dir: Path) -> dict[str, Any]:
    if code < initial_len:
        entry = initial_entries[code]
        return {
            "image_id": entry["image_id"],
            "image_path": entry["path"],
            "role": entry["role"],
            "record_id": entry.get("record_id"),
            "entry_index": int(code),
            "rank": rank,
            "score": float(score),
        }
    path = materialize_augmented(code, initial_len, base_paths, variants, out_dir)
    aug_idx = code - initial_len
    base_idx = aug_idx // len(variants)
    variant = variants[aug_idx % len(variants)]
    return {
        "image_id": f"aug-{base_idx:06d}-{variant}",
        "image_path": str(path.relative_to(ROOT)),
        "role": "distractor_augmented",
        "record_id": None,
        "entry_index": int(code),
        "rank": rank,
        "score": float(score),
    }


def main() -> None:
    import open_clip
    import torch
    from torch.utils.data import DataLoader

    args = parse_args()
    out = resolve(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = load_records(resolve(args.records), args.max_records)
    cache = resolve(args.cache_dir)
    cache_entries = json.loads((cache / "clip_image_entries.json").read_text(encoding="utf-8"))
    image_emb = np.load(cache / "clip_image_embeddings.npy", mmap_mode="r")
    query_entries = json.loads((cache / "clip_query_entries.json").read_text(encoding="utf-8"))[: args.max_records * 3]
    query_emb = np.load(cache / "clip_query_embeddings.npy").astype("float32")[: len(query_entries)]

    fallback = build_fallback_index(args.anchor_fallback_dirs)
    clean_paths = [resolve_anchor_path(record, fallback) for record in records]
    suspect_entries, clean_entries, distractor_start = init_entries(records, cache_entries, clean_paths, args.max_records)
    canary_emb = np.asarray(image_emb[: args.max_records], dtype="float32")
    clean_anchor_emb = np.load(resolve(args.clean_anchor_embeddings)).astype("float32")[: args.max_records]
    base_distractor_emb = np.asarray(image_emb[distractor_start : distractor_start + args.base_distractors], dtype="float32")
    base_entries = cache_entries[distractor_start : distractor_start + args.base_distractors]
    base_paths = [entry["path"] for entry in base_entries]
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    initial_len = args.max_records + len(base_entries)

    top_scores_s = np.full((len(query_entries), args.top_k), -np.inf, dtype=np.float32)
    top_codes_s = np.full((len(query_entries), args.top_k), -1, dtype=np.int64)
    top_scores_c = np.full((len(query_entries), args.top_k), -np.inf, dtype=np.float32)
    top_codes_c = np.full((len(query_entries), args.top_k), -1, dtype=np.int64)
    counts_s = np.zeros(len(query_entries), dtype=np.int64)
    counts_c = np.zeros(len(query_entries), dtype=np.int64)
    target_idx = np.asarray([int(row["record_id"].split("-")[-1]) for row in query_entries], dtype=np.int64)
    target_scores_s = np.sum(query_emb * canary_emb[target_idx], axis=1)
    target_scores_c = np.sum(query_emb * clean_anchor_emb[target_idx], axis=1)

    initial_s = np.vstack([canary_emb, base_distractor_emb])
    initial_c = np.vstack([clean_anchor_emb, base_distractor_emb])
    chunk = 25000
    for start in range(0, initial_len, chunk):
        end = min(initial_len, start + chunk)
        codes = np.arange(start, end, dtype=np.int64)
        scores_s = query_emb @ initial_s[start:end].T
        scores_c = query_emb @ initial_c[start:end].T
        counts_s += np.sum(scores_s > target_scores_s[:, None], axis=1)
        counts_c += np.sum(scores_c > target_scores_c[:, None], axis=1)
        top_scores_s, top_codes_s = update_topk(top_scores_s, top_codes_s, scores_s, codes)
        top_scores_c, top_codes_c = update_topk(top_scores_c, top_codes_c, scores_c, codes)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", device=device)
    model.eval()
    dataset = AugmentedImageDataset(base_paths, variants, preprocess)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=(device.type == "cuda"), multiprocessing_context="fork" if args.num_workers > 0 else None)
    query_t = torch.from_numpy(query_emb).to(device)
    target_s_t = torch.from_numpy(target_scores_s).to(device)
    target_c_t = torch.from_numpy(target_scores_c).to(device)
    processed = 0
    with torch.no_grad():
        for pixels, indices in loader:
            pixels = pixels.to(device, non_blocking=True)
            emb = model.encode_image(pixels)
            emb = torch.nn.functional.normalize(emb.float(), dim=-1)
            scores = query_t @ emb.T
            scores_np = scores.cpu().numpy().astype("float32")
            codes = initial_len + np.asarray(indices, dtype=np.int64)
            counts_s += (scores > target_s_t[:, None]).sum(dim=1).cpu().numpy().astype(np.int64)
            counts_c += (scores > target_c_t[:, None]).sum(dim=1).cpu().numpy().astype(np.int64)
            top_scores_s, top_codes_s = update_topk(top_scores_s, top_codes_s, scores_np, codes)
            top_scores_c, top_codes_c = update_topk(top_scores_c, top_codes_c, scores_np, codes)
            processed += pixels.shape[0]
            if processed % 50000 < args.batch_size:
                print(f"encoded_augmented {processed}/{len(dataset)}", flush=True)

    rows = []
    for qi, q in enumerate(query_entries):
        wm_hits = [
            code_to_hit(int(code), float(score), rank, suspect_entries, initial_len, base_paths, variants, out)
            for rank, (code, score) in enumerate(zip(top_codes_s[qi], top_scores_s[qi]), start=1)
        ]
        clean_hits = [
            code_to_hit(int(code), float(score), rank, clean_entries, initial_len, base_paths, variants, out)
            for rank, (code, score) in enumerate(zip(top_codes_c[qi], top_scores_c[qi]), start=1)
        ]
        rows.append(
            {
                "profile": "clip_visual_1m_augmented",
                "record_index": int(q["record_id"].split("-")[-1]),
                "probe_index": int(q["probe_index"]),
                "id": q["record_id"],
                "signature": q["signature"],
                "query": q["query"],
                "target_rank": int(counts_s[qi] + 1),
                "clean_target_rank": int(counts_c[qi] + 1),
                "watermarked_hits": wm_hits,
                "clean_hits": clean_hits,
            }
        )
    (out / "million_augmented_top_hits.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    summary = {
        "profile": "clip_visual_1m_augmented",
        "num_canaries": args.max_records,
        "num_queries": len(rows),
        "top_k": args.top_k,
        "index_size": args.max_records + args.base_distractors * (1 + len(variants)),
        "distractors": args.base_distractors * (1 + len(variants)),
    }
    (out / "million_augmented_top_hits_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

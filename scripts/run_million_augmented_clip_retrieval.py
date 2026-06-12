#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class AugmentedImageDataset:
    def __init__(self, paths: list[str], variants: list[str], preprocess: Any) -> None:
        self.paths = paths
        self.variants = variants
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.paths) * len(self.variants)

    def __getitem__(self, idx: int):
        base_idx = idx // len(self.variants)
        variant = self.variants[idx % len(self.variants)]
        image = Image.open(self.paths[base_idx]).convert("RGB")
        image = apply_variant(image, variant)
        return self.preprocess(image), idx


def apply_variant(image: Image.Image, variant: str) -> Image.Image:
    if variant == "hflip":
        return ImageOps.mirror(image)
    if variant == "center_crop_90":
        w, h = image.size
        nw, nh = int(w * 0.9), int(h * 0.9)
        x, y = (w - nw) // 2, (h - nh) // 2
        return image.crop((x, y, x + nw, y + nh)).resize((w, h), Image.Resampling.BICUBIC)
    if variant == "crop_tl_88":
        w, h = image.size
        nw, nh = int(w * 0.88), int(h * 0.88)
        return image.crop((0, 0, nw, nh)).resize((w, h), Image.Resampling.BICUBIC)
    if variant == "crop_br_88":
        w, h = image.size
        nw, nh = int(w * 0.88), int(h * 0.88)
        return image.crop((w - nw, h - nh, w, h)).resize((w, h), Image.Resampling.BICUBIC)
    if variant == "bright_90_contrast_110":
        return ImageEnhance.Contrast(ImageEnhance.Brightness(image).enhance(0.9)).enhance(1.1)
    if variant == "bright_110_contrast_90":
        return ImageEnhance.Contrast(ImageEnhance.Brightness(image).enhance(1.1)).enhance(0.9)
    if variant == "color_80":
        return ImageEnhance.Color(image).enhance(0.8)
    if variant == "blur_06":
        return image.filter(ImageFilter.GaussianBlur(radius=0.6))
    if variant == "rotate_2":
        return image.rotate(2.0, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(127, 127, 127))
    raise ValueError(f"Unknown variant: {variant}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run 1M-scale CLIP retrieval stress using augmented real-image distractors.")
    parser.add_argument("--cache_dir", default="outputs/pipeline_generality_flux_n500_textvqa_100k_clip")
    parser.add_argument("--output_dir", default="outputs/pipeline_generality_flux_n500_textvqa_1m_augmented_clip")
    parser.add_argument("--max_records", type=int, default=500)
    parser.add_argument("--base_distractors", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=24)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--variants", default="hflip,center_crop_90,crop_tl_88,crop_br_88,bright_90_contrast_110,bright_110_contrast_90,color_80,blur_06,rotate_2")
    parser.add_argument("--top_ks", default="1,3,5,10,20,50,100,1000")
    return parser.parse_args()


def main() -> None:
    import open_clip
    import torch
    from torch.utils.data import DataLoader

    args = parse_args()
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    cache = ROOT / args.cache_dir
    entries = json.loads((cache / "clip_image_entries.json").read_text(encoding="utf-8"))
    image_emb = np.load(cache / "clip_image_embeddings.npy", mmap_mode="r")
    query_entries = json.loads((cache / "clip_query_entries.json").read_text(encoding="utf-8"))[: args.max_records * 3]
    query_emb_np = np.load(cache / "clip_query_embeddings.npy").astype("float32")[: len(query_entries)]

    canary_emb = np.asarray(image_emb[: args.max_records], dtype="float32")
    distractor_entries = [entry for entry in entries if entry.get("role") == "distractor"][: args.base_distractors]
    distractor_start = next(i for i, entry in enumerate(entries) if entry.get("role") == "distractor")
    base_distractor_emb = np.asarray(image_emb[distractor_start : distractor_start + len(distractor_entries)], dtype="float32")
    base_paths = [entry["path"] for entry in distractor_entries]
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]

    target_indices = np.array([int(row["record_id"].split("-")[-1]) for row in query_entries], dtype=np.int64)
    target_scores = np.sum(query_emb_np * canary_emb[target_indices], axis=1)
    initial_emb = np.vstack([canary_emb, base_distractor_emb])
    counts = np.zeros(len(query_entries), dtype=np.int64)
    chunk = 25000
    for start in range(0, initial_emb.shape[0], chunk):
        scores = query_emb_np @ initial_emb[start : start + chunk].T
        counts += np.sum(scores > target_scores[:, None], axis=1)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", device=device)
    model.eval()
    dataset = AugmentedImageDataset(base_paths, variants, preprocess)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        multiprocessing_context="fork" if args.num_workers > 0 else None,
    )
    query_t = torch.from_numpy(query_emb_np).to(device)
    target_t = torch.from_numpy(target_scores).to(device)
    processed = 0
    with torch.no_grad():
        for pixels, _indices in loader:
            pixels = pixels.to(device, non_blocking=True)
            emb = model.encode_image(pixels)
            emb = torch.nn.functional.normalize(emb.float(), dim=-1)
            scores = query_t @ emb.T
            counts += (scores > target_t[:, None]).sum(dim=1).cpu().numpy().astype(np.int64)
            processed += pixels.shape[0]
            if processed % 50000 < args.batch_size:
                print(f"encoded_augmented {processed}/{len(dataset)}", flush=True)

    ranks = counts + 1
    details = []
    for row, rank in zip(query_entries, ranks.tolist()):
        details.append(
            {
                "variant": "clip_visual_1m_augmented",
                "record_id": row["record_id"],
                "probe_index": row["probe_index"],
                "signature": row["signature"],
                "query": row["query"],
                "rank": int(rank),
                "target_retained": True,
            }
        )
    ranks_np = ranks.astype(np.float64)
    top_ks = [int(k) for k in args.top_ks.split(",") if k.strip()]
    summary = {
        "variant": "clip_visual_1m_augmented",
        "num_records": args.max_records,
        "num_queries": len(query_entries),
        "index_size": int(args.max_records + len(base_distractor_emb) * (1 + len(variants))),
        "base_distractors": len(base_distractor_emb),
        "augmented_variants_per_base": len(variants),
        "distractors": int(len(base_distractor_emb) * (1 + len(variants))),
        "target_retention": 1.0,
        "mean_rank": float(ranks_np.mean()),
        "median_rank": float(np.median(ranks_np)),
        "mrr": float(np.mean(1.0 / ranks_np)),
        "augmentation_note": "1M distractors are produced from 100k real COCO images using deterministic image augmentations; this is an image-derived scale stress test, not 1M unique source photographs.",
    }
    for k in top_ks:
        summary[f"recall_at_{k}"] = float(np.mean(ranks_np <= k))
    (out / "pipeline_generality_details.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in details) + "\n", encoding="utf-8")
    (out / "pipeline_generality_summary.json").write_text(json.dumps([summary], indent=2), encoding="utf-8")
    with (out / "pipeline_generality_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

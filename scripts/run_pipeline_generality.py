#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantitrace.metrics import normalize_text
from semantitrace.utils.image import list_images


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class IndexEntry:
    image_id: str
    image_path: Path
    role: str
    record_id: str | None = None


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate SemantiTrace canaries under production-like RAG pipeline variants")
    parser.add_argument("--records", default="outputs/flux2_klein_gradient_guided_n100_tuned_v1/canary_records.json")
    parser.add_argument("--output_dir", default="outputs/pipeline_generality_flux_n100")
    parser.add_argument("--distractor_dirs", nargs="*", default=["data/mmqa/images", "data/webqa/images"])
    parser.add_argument("--max_records", type=int, default=100)
    parser.add_argument("--max_distractors", type=int, default=400)
    parser.add_argument("--num_queries", type=int, default=3)
    parser.add_argument(
        "--variants",
        default="clip_visual,siglip_visual,clip_jpeg_resize,ocr_text,clip_ocr_hybrid,clip_top50_ocr_rerank,blip_caption,blip_caption_sidecar,clip_caption_hybrid,clip_phash_dedup",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--rerank_pool", type=int, default=50)
    parser.add_argument("--hybrid_text_weight", type=float, default=0.35)
    parser.add_argument("--caption_batch_size", type=int, default=16)
    parser.add_argument("--phash_threshold", type=int, default=4)
    return parser.parse_args()


def load_records(path: Path, max_records: int) -> list[dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8"))
    usable = []
    for record in records:
        wm = resolve(record["watermarked_image_path"])
        if wm.is_file():
            usable.append(record)
        if len(usable) >= max_records:
            break
    if not usable:
        raise FileNotFoundError(f"No usable records found in {path}")
    return usable


def collect_distractors(dirs: list[str], limit: int, exclude: set[Path]) -> list[Path]:
    out: list[Path] = []
    seen = {p.resolve() for p in exclude}
    for raw in dirs:
        for path in list_images(resolve(raw)):
            p = Path(path).resolve()
            if p in seen:
                continue
            seen.add(p)
            out.append(p)
            if len(out) >= limit:
                return out
    return out


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
        entries.append(IndexEntry(image_id=f"distractor-{i:05d}", image_path=path, role="distractor"))
    return entries


def image_cache_key(path: Path) -> str:
    return hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:16]


def preprocess_jpeg_resize(entries: list[IndexEntry], output_dir: Path, max_side: int = 768, quality: int = 75) -> list[IndexEntry]:
    out_dir = output_dir / "_preprocessed" / f"jpeg_q{quality}_max{max_side}"
    out_dir.mkdir(parents=True, exist_ok=True)
    transformed: list[IndexEntry] = []
    for entry in entries:
        out_path = out_dir / f"{image_cache_key(entry.image_path)}.jpg"
        if not out_path.exists():
            image = Image.open(entry.image_path).convert("RGB")
            scale = min(1.0, max_side / max(image.size))
            if scale < 1.0:
                image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.Resampling.LANCZOS)
            image.save(out_path, quality=quality, optimize=True)
        transformed.append(IndexEntry(entry.image_id, out_path, entry.role, entry.record_id))
    return transformed


class OpenCLIPScorer:
    def __init__(self, device: str, batch_size: int) -> None:
        import open_clip
        import torch

        self.open_clip = open_clip
        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.batch_size = batch_size
        self.model, _, self.preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", device=self.device)
        self.tokenizer = open_clip.get_tokenizer("ViT-L-14")
        self.model.eval()

    def encode_images(self, paths: list[Path]) -> np.ndarray:
        feats = []
        with self.torch.no_grad():
            for start in range(0, len(paths), self.batch_size):
                batch = [self.preprocess(Image.open(p).convert("RGB")) for p in paths[start : start + self.batch_size]]
                pixels = self.torch.stack(batch).to(self.device)
                emb = self.model.encode_image(pixels)
                emb = self.torch.nn.functional.normalize(emb, dim=-1)
                feats.append(emb.cpu().float().numpy())
        return np.vstack(feats)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        feats = []
        with self.torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                tokens = self.tokenizer(texts[start : start + self.batch_size]).to(self.device)
                emb = self.model.encode_text(tokens)
                emb = self.torch.nn.functional.normalize(emb, dim=-1)
                feats.append(emb.cpu().float().numpy())
        return np.vstack(feats)


class SigLIPScorer:
    def __init__(self, device: str, batch_size: int) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.batch_size = batch_size
        self.processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
        self.model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384", torch_dtype=torch.bfloat16).to(self.device).eval()

    def encode_images(self, paths: list[Path]) -> np.ndarray:
        feats = []
        with self.torch.no_grad():
            for start in range(0, len(paths), self.batch_size):
                images = [Image.open(p).convert("RGB") for p in paths[start : start + self.batch_size]]
                inputs = self.processor(images=images, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                emb = self.model.get_image_features(**inputs)
                if hasattr(emb, "pooler_output"):
                    emb = emb.pooler_output
                elif hasattr(emb, "last_hidden_state"):
                    emb = emb.last_hidden_state[:, 0]
                emb = self.torch.nn.functional.normalize(emb.float(), dim=-1)
                feats.append(emb.cpu().numpy())
        return np.vstack(feats)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        feats = []
        with self.torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                inputs = self.processor(text=texts[start : start + self.batch_size], return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                emb = self.model.get_text_features(**inputs)
                if hasattr(emb, "pooler_output"):
                    emb = emb.pooler_output
                elif hasattr(emb, "last_hidden_state"):
                    emb = emb.last_hidden_state[:, 0]
                emb = self.torch.nn.functional.normalize(emb.float(), dim=-1)
                feats.append(emb.cpu().numpy())
        return np.vstack(feats)


def cosine_scores(image_emb: np.ndarray, query_emb: np.ndarray) -> np.ndarray:
    return image_emb @ query_emb


def tokenize(text: str) -> list[str]:
    return [tok for tok in re.split(r"\W+", normalize_text(text)) if tok]


def lexical_scores(texts: list[str], query: str) -> np.ndarray:
    q_tokens = tokenize(query)
    if not q_tokens:
        return np.zeros(len(texts), dtype=np.float32)
    q_counts: dict[str, int] = {}
    for tok in q_tokens:
        q_counts[tok] = q_counts.get(tok, 0) + 1
    scores = np.zeros(len(texts), dtype=np.float32)
    for i, doc in enumerate(texts):
        d_tokens = tokenize(doc)
        if not d_tokens:
            continue
        d_set = set(d_tokens)
        score = sum(weight for tok, weight in q_counts.items() if tok in d_set)
        if score:
            scores[i] = score / math.sqrt(len(d_tokens))
    return scores


def zscore(values: np.ndarray) -> np.ndarray:
    std = float(values.std())
    if std < 1e-8:
        return np.zeros_like(values)
    return (values - float(values.mean())) / std


def extract_ocr_texts(entries: list[IndexEntry], output_dir: Path, device: str) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "ocr_texts.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        cached = {}
    missing = [entry for entry in entries if str(entry.image_path) not in cached]
    if missing:
        import easyocr
        import torch

        reader = easyocr.Reader(["en"], gpu=(device != "cpu" and torch.cuda.is_available()))
        for n, entry in enumerate(missing, start=1):
            result = reader.readtext(str(entry.image_path), detail=1, paragraph=False)
            texts = [str(item[1]) for item in result if len(item) >= 2]
            cached[str(entry.image_path)] = " ".join(texts)
            if n % 25 == 0:
                cache_path.write_text(json.dumps(cached, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"OCR cached {n}/{len(missing)} new images", flush=True)
        cache_path.write_text(json.dumps(cached, indent=2, ensure_ascii=False), encoding="utf-8")
    return [str(cached.get(str(entry.image_path), "")) for entry in entries]


def extract_blip_captions(entries: list[IndexEntry], output_dir: Path, device: str, batch_size: int) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "blip_captions.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        cached = {}
    missing = [entry for entry in entries if str(entry.image_path) not in cached]
    if missing:
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor

        model_id = "Salesforce/blip-image-captioning-base"
        run_device = device if device != "cpu" and torch.cuda.is_available() else "cpu"
        processor = BlipProcessor.from_pretrained(model_id)
        model = BlipForConditionalGeneration.from_pretrained(model_id).to(run_device).eval()
        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            images = [Image.open(entry.image_path).convert("RGB") for entry in batch]
            inputs = processor(images=images, return_tensors="pt", padding=True).to(run_device)
            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=32)
            captions = processor.batch_decode(output_ids, skip_special_tokens=True)
            for entry, caption in zip(batch, captions):
                cached[str(entry.image_path)] = str(caption)
            if (start // batch_size + 1) % 5 == 0:
                cache_path.write_text(json.dumps(cached, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"BLIP captions cached {min(start + len(batch), len(missing))}/{len(missing)} new images", flush=True)
        cache_path.write_text(json.dumps(cached, indent=2, ensure_ascii=False), encoding="utf-8")
    return [str(cached.get(str(entry.image_path), "")) for entry in entries]


def phash_dedup(entries: list[IndexEntry], threshold: int) -> tuple[list[IndexEntry], dict[str, bool]]:
    import imagehash

    kept: list[IndexEntry] = []
    hashes = []
    retained: dict[str, bool] = {}
    # Keep canaries first: this models deduplicating exact corpus duplicates rather than
    # removing watermarks by comparing against an unavailable clean original.
    for entry in sorted(entries, key=lambda e: 0 if e.role == "canary" else 1):
        h = imagehash.phash(Image.open(entry.image_path).convert("RGB"))
        duplicate = any(h - old <= threshold for old in hashes)
        if not duplicate:
            kept.append(entry)
            hashes.append(h)
            if entry.record_id:
                retained[entry.record_id] = True
        elif entry.record_id:
            retained[entry.record_id] = False
    return kept, retained


def rank_from_scores(scores: np.ndarray, target_idx: int, candidates: np.ndarray | None = None) -> int:
    if candidates is None:
        order = np.argsort(-scores)
    else:
        order = candidates[np.argsort(-scores[candidates])]
    match = np.where(order == target_idx)[0]
    return int(match[0] + 1) if match.size else int(len(scores) + 1)


def evaluate_variant(
    variant: str,
    entries: list[IndexEntry],
    records: list[dict[str, Any]],
    output_dir: Path,
    args: argparse.Namespace,
    clip: OpenCLIPScorer | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    variant_dir = output_dir / variant
    variant_dir.mkdir(parents=True, exist_ok=True)
    working_entries = list(entries)
    retained_targets = {str(r["id"]): True for r in records}
    ocr_texts: list[str] | None = None
    caption_texts: list[str] | None = None
    visual_model = None

    if variant == "clip_jpeg_resize":
        working_entries = preprocess_jpeg_resize(working_entries, output_dir)
        visual_model = clip or OpenCLIPScorer(args.device, args.batch_size)
    elif variant == "siglip_visual":
        visual_model = SigLIPScorer(args.device, max(1, args.batch_size // 2))
    elif variant == "clip_phash_dedup":
        working_entries, retained_targets = phash_dedup(working_entries, args.phash_threshold)
        visual_model = clip or OpenCLIPScorer(args.device, args.batch_size)
    elif variant in {"clip_visual", "clip_ocr_hybrid", "clip_top50_ocr_rerank", "clip_caption_hybrid"}:
        visual_model = clip or OpenCLIPScorer(args.device, args.batch_size)

    if variant in {"ocr_text", "clip_ocr_hybrid", "clip_top50_ocr_rerank"}:
        ocr_texts = extract_ocr_texts(working_entries, output_dir / "_ocr_cache", args.device)
    if variant in {"blip_caption", "blip_caption_sidecar", "clip_caption_hybrid"}:
        caption_texts = extract_blip_captions(
            working_entries,
            output_dir / "_caption_cache",
            args.device,
            args.caption_batch_size,
        )
        if variant == "blip_caption_sidecar":
            signature_by_id = {str(record["id"]): str(record["trap_signature"]) for record in records}
            caption_texts = [
                (text + f" provenance canary {signature_by_id[entry.record_id]}")
                if entry.record_id in signature_by_id
                else text
                for entry, text in zip(working_entries, caption_texts)
            ]

    image_paths = [entry.image_path for entry in working_entries]
    image_emb = visual_model.encode_images(image_paths) if visual_model else None
    target_index = {entry.record_id: idx for idx, entry in enumerate(working_entries) if entry.record_id}

    details: list[dict[str, Any]] = []
    ranks: list[int] = []
    for record in records:
        rec_id = str(record["id"])
        queries = list(record.get("probe_queries", []))[: args.num_queries]
        for probe_idx, query in enumerate(queries):
            if rec_id not in target_index or not retained_targets.get(rec_id, False):
                rank = len(working_entries) + 1
            else:
                target_idx = target_index[rec_id]
                if variant == "ocr_text":
                    scores = lexical_scores(ocr_texts or ["" for _ in working_entries], query)
                    rank = rank_from_scores(scores, target_idx)
                elif variant == "blip_caption":
                    scores = lexical_scores(caption_texts or ["" for _ in working_entries], query)
                    rank = rank_from_scores(scores, target_idx)
                elif variant == "blip_caption_sidecar":
                    docs = caption_texts or ["" for _ in working_entries]
                    scores = lexical_scores(docs, query)
                    # A caption-sidecar canary is a structured provenance field:
                    # exact rare-token matches should dominate generic caption terms.
                    signature = str(record.get("trap_signature", ""))
                    query_norm = normalize_text(query)
                    sig_norm = normalize_text(signature)
                    if sig_norm and sig_norm in query_norm:
                        for i, doc in enumerate(docs):
                            if sig_norm in normalize_text(doc):
                                scores[i] += 1000.0
                    rank = rank_from_scores(scores, target_idx)
                elif variant == "clip_ocr_hybrid":
                    assert image_emb is not None and visual_model is not None and ocr_texts is not None
                    q_emb = visual_model.encode_texts([query])[0]
                    visual = cosine_scores(image_emb, q_emb)
                    text = lexical_scores(ocr_texts, query)
                    scores = zscore(visual) + args.hybrid_text_weight * zscore(text)
                    rank = rank_from_scores(scores, target_idx)
                elif variant == "clip_caption_hybrid":
                    assert image_emb is not None and visual_model is not None and caption_texts is not None
                    q_emb = visual_model.encode_texts([query])[0]
                    visual = cosine_scores(image_emb, q_emb)
                    text = lexical_scores(caption_texts, query)
                    scores = zscore(visual) + args.hybrid_text_weight * zscore(text)
                    rank = rank_from_scores(scores, target_idx)
                elif variant == "clip_top50_ocr_rerank":
                    assert image_emb is not None and visual_model is not None and ocr_texts is not None
                    q_emb = visual_model.encode_texts([query])[0]
                    visual = cosine_scores(image_emb, q_emb)
                    pool = np.argsort(-visual)[: min(args.rerank_pool, len(visual))]
                    text = lexical_scores(ocr_texts, query)
                    # Rerank the visual candidate pool by OCR lexical evidence, then by visual score.
                    pool_order = sorted(pool.tolist(), key=lambda idx: (float(text[idx]), float(visual[idx])), reverse=True)
                    match = pool_order.index(target_idx) + 1 if target_idx in pool_order else len(working_entries) + 1
                    rank = int(match)
                else:
                    assert image_emb is not None and visual_model is not None
                    q_emb = visual_model.encode_texts([query])[0]
                    scores = cosine_scores(image_emb, q_emb)
                    rank = rank_from_scores(scores, target_idx)
            ranks.append(rank)
            details.append(
                {
                    "variant": variant,
                    "record_id": rec_id,
                    "probe_index": probe_idx,
                    "signature": record.get("trap_signature"),
                    "query": query,
                    "rank": rank,
                    "target_retained": bool(retained_targets.get(rec_id, False)),
                }
            )

    arr = np.asarray(ranks, dtype=np.float64)
    summary = {
        "variant": variant,
        "num_records": len(records),
        "num_queries": len(ranks),
        "index_size": len(working_entries),
        "target_retention": float(np.mean([retained_targets.get(str(r["id"]), False) for r in records])),
        "mean_rank": float(arr.mean()) if arr.size else 0.0,
        "median_rank": float(np.median(arr)) if arr.size else 0.0,
        "mrr": float(np.mean([1.0 / r if r <= len(working_entries) else 0.0 for r in ranks])) if ranks else 0.0,
        "recall_at_1": float(np.mean(arr <= 1)) if arr.size else 0.0,
        "recall_at_3": float(np.mean(arr <= 3)) if arr.size else 0.0,
        "recall_at_5": float(np.mean(arr <= 5)) if arr.size else 0.0,
        "recall_at_10": float(np.mean(arr <= 10)) if arr.size else 0.0,
    }
    return summary, details


def write_outputs(output_dir: Path, summaries: list[dict[str, Any]], details: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pipeline_generality_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    fieldnames = []
    for row in summaries:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with (output_dir / "pipeline_generality_summary.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    with (output_dir / "pipeline_generality_details.jsonl").open("w", encoding="utf-8") as fh:
        for row in details:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    output_dir = resolve(args.output_dir)
    records = load_records(resolve(args.records), args.max_records)
    exclude = {resolve(record["watermarked_image_path"]) for record in records}
    exclude |= {resolve(record["anchor_image_path"]) for record in records if record.get("anchor_image_path")}
    distractors = collect_distractors(args.distractor_dirs, args.max_distractors, exclude)
    entries = build_entries(records, distractors)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    clip = None
    if any(v.startswith("clip") for v in variants):
        clip = OpenCLIPScorer(args.device, args.batch_size)

    summaries: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for variant in variants:
        print(f"Running variant {variant} with {len(entries)} indexed images", flush=True)
        try:
            summary, rows = evaluate_variant(variant, entries, records, output_dir, args, clip=clip)
        except Exception as exc:
            summary = {
                "variant": variant,
                "num_records": len(records),
                "num_queries": 0,
                "index_size": len(entries),
                "target_retention": 0.0,
                "mean_rank": 0.0,
                "median_rank": 0.0,
                "mrr": 0.0,
                "recall_at_1": 0.0,
                "recall_at_3": 0.0,
                "recall_at_5": 0.0,
                "recall_at_10": 0.0,
                "error": repr(exc),
            }
            rows = []
            print(f"Variant {variant} failed: {exc!r}", flush=True)
        summaries.append(summary)
        details.extend(rows)
        write_outputs(output_dir, summaries, details)
        print(summary, flush=True)


if __name__ == "__main__":
    main()

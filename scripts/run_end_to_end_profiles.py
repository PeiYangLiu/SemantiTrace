#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_pipeline_generality import (
    IndexEntry,
    OpenCLIPScorer,
    build_entries,
    collect_distractors,
    cosine_scores,
    extract_blip_captions,
    extract_ocr_texts,
    lexical_scores,
    load_records,
    preprocess_jpeg_resize,
    rank_from_scores,
    resolve,
    zscore,
)
from semantitrace.backends.real import OpenCLIPEncoder, QwenVLMClient
from semantitrace.config import load_config
from semantitrace.metrics import normalize_text
from semantitrace.mode_verification import (
    detail_response_hit,
    detail_target_gated_hit,
    per_canary_rates_from_predicate,
    score_response,
    target_rank_in_topk,
)
from semantitrace.verification import Verifier


ROOT = Path(__file__).resolve().parents[1]


def rag_prompt(query: str) -> str:
    return (
        "Answer the user using the retrieved visual evidence. "
        "If text is visible, transcribe it exactly.\n\n"
        f"User query: {query}"
    )


def build_clients(config_path: str, device: str) -> tuple[OpenCLIPEncoder, QwenVLMClient, Verifier]:
    cfg = load_config(config_path)
    models = cfg.get("models", {})
    vlm_cfg = cfg.get("vlm", {})
    encoder = OpenCLIPEncoder(
        model_name=models.get("clip_model", "ViT-L-14"),
        pretrained=models.get("clip_pretrained", "openai"),
        device=device,
    )
    vlm = QwenVLMClient(
        model_name=models.get("surrogate_vlm", "Qwen/Qwen3-VL-8B-Instruct"),
        device=device,
        torch_dtype=vlm_cfg.get("torch_dtype", "bfloat16"),
    )
    return encoder, vlm, Verifier(cfg.get("verification", {}))


@dataclass
class PreparedIndex:
    entries: list[IndexEntry]
    image_emb: np.ndarray | None
    ocr_texts: list[str] | None
    caption_texts: list[str] | None
    target_index: dict[str, int]
    retained_targets: dict[str, bool]


PROFILE_DEFS: dict[str, dict[str, Any]] = {
    "visual_clip": {
        "label": "Visual RAG",
        "variant": "clip_visual",
        "context": "image_montage",
        "description": "CLIP visual index -> top-k image context -> VLM",
    },
    "visual_jpeg": {
        "label": "Visual RAG + JPEG/resize",
        "variant": "clip_jpeg_resize",
        "context": "image_montage",
        "description": "JPEG/resize preprocessing -> CLIP visual index -> top-k image context -> VLM",
    },
    "hybrid_ocr": {
        "label": "Hybrid OCR RAG",
        "variant": "clip_top50_ocr_rerank",
        "context": "image_montage",
        "description": "CLIP top-50 retrieval -> OCR reranking -> top-k image context -> VLM",
    },
    "caption_only": {
        "label": "Caption-only RAG",
        "variant": "blip_caption",
        "context": "text_context",
        "description": "BLIP captions -> text retrieval -> text-only generator",
    },
    "caption_sidecar": {
        "label": "Caption + explicit provenance sidecar RAG",
        "variant": "blip_caption_sidecar",
        "context": "text_context",
        "description": "BLIP captions + explicit canary-bearing provenance sidecar -> text retrieval -> text-only generator",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run measured end-to-end RAG profile audits")
    parser.add_argument("--records", default="outputs/flux2_klein_gradient_guided_n100_tuned_v1/canary_records.json")
    parser.add_argument("--output_dir", default="outputs/end_to_end_profiles_flux_n100")
    parser.add_argument("--distractor_dirs", nargs="*", default=["data/mmqa/images", "data/webqa/images"])
    parser.add_argument(
        "--anchor_fallback_dirs",
        nargs="*",
        default=[
            "data_scene_text/total_text/images",
            "data_scene_text/coco_text/images",
            "data_webqa_5000/webqa/images",
            "data_expanded/mmqa/images",
            "data_expanded/webqa/images",
            "data_textvqa_ocr_shards/shard_0/images",
            "data_textvqa_ocr_shards/shard_1/images",
            "data_textvqa_ocr_shards/shard_2/images",
            "data_textvqa_ocr_shards/shard_3/images",
            "data/mmqa/images",
            "data/webqa/images",
        ],
    )
    parser.add_argument("--max_records", type=int, default=100)
    parser.add_argument("--max_distractors", type=int, default=400)
    parser.add_argument("--profiles", default="visual_clip,hybrid_ocr,caption_only,caption_sidecar")
    parser.add_argument("--config", default="configs/semantitrace_flux2_klein_gradient_guided.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--caption_batch_size", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--rerank_pool", type=int, default=50)
    parser.add_argument("--hybrid_text_weight", type=float, default=0.35)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--montage_tile", type=int, default=448)
    parser.add_argument("--seal_only", action="store_true", help="Only write sealed profile configs and exit")
    parser.add_argument("--fresh", action="store_true", help="Ignore existing detail cache")
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def build_fallback_index(dirs: list[str]) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for raw in dirs:
        root = resolve(raw)
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                index.setdefault(path.name, path)
    return index


def resolve_anchor_path(record: dict[str, Any], fallback_index: dict[str, Path]) -> Path:
    raw = str(record["anchor_image_path"])
    path = resolve(raw)
    if path.is_file():
        return path
    fallback = fallback_index.get(Path(raw).name)
    if fallback and fallback.is_file():
        return fallback
    raise FileNotFoundError(f"Missing anchor image for {record.get('id')}: {raw}")


def make_clean_entries(records: list[dict[str, Any]], distractors: list[Path], fallback_index: dict[str, Path]) -> list[IndexEntry]:
    entries: list[IndexEntry] = []
    for i, path in enumerate(distractors):
        entries.append(IndexEntry(image_id=f"distractor-{i:05d}", image_path=path, role="distractor"))
    return entries


def add_caption_sidecars(entries: list[IndexEntry], captions: list[str], records: list[dict[str, Any]], enabled: bool) -> list[str]:
    if not enabled:
        return captions
    signature_by_id = {str(record["id"]): str(record["trap_signature"]) for record in records}
    out: list[str] = []
    for entry, caption in zip(entries, captions):
        if entry.record_id in signature_by_id:
            out.append(f"{caption} provenance canary {signature_by_id[entry.record_id]}")
        else:
            out.append(caption)
    return out


def prepare_index(
    variant: str,
    entries: list[IndexEntry],
    records: list[dict[str, Any]],
    output_dir: Path,
    args: argparse.Namespace,
    clip: OpenCLIPScorer,
    *,
    sidecar_enabled: bool,
) -> PreparedIndex:
    working_entries = list(entries)
    retained_targets = {str(r["id"]): True for r in records}
    if variant == "clip_jpeg_resize":
        working_entries = preprocess_jpeg_resize(working_entries, output_dir)

    ocr_texts: list[str] | None = None
    caption_texts: list[str] | None = None
    image_emb: np.ndarray | None = None
    if variant in {"clip_visual", "clip_jpeg_resize", "clip_top50_ocr_rerank"}:
        image_emb = clip.encode_images([entry.image_path for entry in working_entries])
    if variant == "clip_top50_ocr_rerank":
        ocr_texts = extract_ocr_texts(working_entries, output_dir / "_ocr_cache", args.device)
    if variant in {"blip_caption", "blip_caption_sidecar"}:
        caption_texts = extract_blip_captions(
            working_entries,
            output_dir / "_caption_cache",
            args.device,
            args.caption_batch_size,
        )
        caption_texts = add_caption_sidecars(working_entries, caption_texts, records, sidecar_enabled)

    target_index = {entry.record_id: idx for idx, entry in enumerate(working_entries) if entry.record_id}
    return PreparedIndex(working_entries, image_emb, ocr_texts, caption_texts, target_index, retained_targets)


def top_hits(
    prepared: PreparedIndex,
    variant: str,
    query: str,
    target_id: str,
    signature: str,
    clip: OpenCLIPScorer,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], int]:
    target_idx = prepared.target_index.get(target_id)
    target_retained = bool(prepared.retained_targets.get(target_id, False)) and target_idx is not None

    if variant in {"clip_visual", "clip_jpeg_resize"}:
        assert prepared.image_emb is not None
        q_emb = clip.encode_texts([query])[0]
        scores = cosine_scores(prepared.image_emb, q_emb)
        order = np.argsort(-scores)
    elif variant == "clip_top50_ocr_rerank":
        assert prepared.image_emb is not None and prepared.ocr_texts is not None
        q_emb = clip.encode_texts([query])[0]
        visual = cosine_scores(prepared.image_emb, q_emb)
        pool = np.argsort(-visual)[: min(args.rerank_pool, len(visual))]
        text = lexical_scores(prepared.ocr_texts, query)
        order = np.asarray(sorted(pool.tolist(), key=lambda idx: (float(text[idx]), float(visual[idx])), reverse=True))
    elif variant in {"blip_caption", "blip_caption_sidecar"}:
        docs = prepared.caption_texts or ["" for _ in prepared.entries]
        scores = lexical_scores(docs, query)
        if variant == "blip_caption_sidecar":
            query_norm = normalize_text(query)
            sig_norm = normalize_text(signature)
            for i, doc in enumerate(docs):
                doc_norm = normalize_text(doc)
                # Rare-token sidecars represent structured metadata fields. Exact
                # canary-token matches should dominate generic caption terms.
                if sig_norm and sig_norm in query_norm and sig_norm in doc_norm:
                    scores[i] += 1000.0
        order = np.argsort(-scores)
    else:
        raise ValueError(f"Unknown profile variant: {variant}")

    if target_retained:
        match = np.where(order == target_idx)[0]
        target_rank = int(match[0] + 1) if match.size else int(len(prepared.entries) + 1)
    else:
        target_rank = int(len(prepared.entries) + 1)
    hits: list[dict[str, Any]] = []
    for rank, idx in enumerate(order[: args.top_k], start=1):
        entry = prepared.entries[int(idx)]
        score = None
        hits.append(
            {
                "image_id": entry.image_id,
                "image_path": rel(entry.image_path) if entry.image_path.is_absolute() and entry.image_path.is_relative_to(ROOT) else str(entry.image_path),
                "role": entry.role,
                "record_id": entry.record_id,
                "entry_index": int(idx),
                "rank": rank,
                "score": score,
            }
        )
    return hits, target_rank


def safe_font(size: int = 22) -> ImageFont.ImageFont:
    for candidate in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
    ]:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def make_montage(hits: list[dict[str, Any]], out_path: Path, tile: int) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return out_path
    if not hits:
        Image.new("RGB", (tile, tile), (245, 245, 245)).save(out_path)
        return out_path
    font = safe_font(max(18, tile // 22))
    panels: list[Image.Image] = []
    for hit in hits:
        image = Image.open(resolve(hit["image_path"])).convert("RGB")
        image.thumbnail((tile, tile), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (tile, tile), (248, 248, 248))
        x = (tile - image.width) // 2
        y = (tile - image.height) // 2
        canvas.paste(image, (x, y))
        draw = ImageDraw.Draw(canvas)
        label = f"#{hit['rank']}"
        draw.rectangle((0, 0, max(58, tile // 7), max(36, tile // 13)), fill=(0, 0, 0))
        draw.text((8, 5), label, fill=(255, 255, 255), font=font)
        panels.append(canvas)
    montage = Image.new("RGB", (tile * len(panels), tile), (248, 248, 248))
    for idx, panel in enumerate(panels):
        montage.paste(panel, (idx * tile, 0))
    montage.save(out_path)
    return out_path


def text_context_prompt(query: str, hits: list[dict[str, Any]], docs: list[str]) -> str:
    lines = [
        "You are answering from retrieved text/caption context only.",
        "Do not copy rare marks from the user question unless they appear in the retrieved context.",
        "If the requested mark is absent from the retrieved context, answer that it is not visible.",
        "",
        "Retrieved context:",
    ]
    for hit in hits:
        idx = int(hit.get("entry_index", -1))
        doc = docs[idx] if 0 <= idx < len(docs) else ""
        lines.append(f"[rank {hit['rank']}] {doc}")
    lines.append("")
    lines.append(f"User query: {query}")
    return "\n".join(lines)


def image_context_prompt(query: str) -> str:
    return (
        "Answer the user using only the retrieved image panels shown here. "
        "Panels are labeled by retrieval rank. Do not copy rare marks from the user question "
        "unless they are visible in a retrieved panel. If the mark is absent, say it is not visible.\n\n"
        f"User query: {query}"
    )


def stable_config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def write_sealed_configs(out_dir: Path, profile_names: list[str], args: argparse.Namespace) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sealed: dict[str, Any] = {
        "protocol": "simulated blind audit over sealed modular RAG profile configs",
        "records": args.records,
        "max_records": args.max_records,
        "max_distractors": args.max_distractors,
        "top_k": args.top_k,
        "profiles": [],
    }
    for profile in profile_names:
        spec = dict(PROFILE_DEFS[profile])
        config = {
            "profile": profile,
            "variant": spec["variant"],
            "context": spec["context"],
            "top_k": args.top_k,
            "rerank_pool": args.rerank_pool,
            "hybrid_text_weight": args.hybrid_text_weight,
            "distractor_dirs": args.distractor_dirs,
        }
        spec["sealed_config_hash"] = stable_config_hash(config)
        spec["sealed_config"] = config
        sealed["profiles"].append(spec)
    (out_dir / "sealed_profile_configs.json").write_text(json.dumps(sealed, indent=2, ensure_ascii=False), encoding="utf-8")
    return sealed


def summarize_profile(
    profile: str,
    spec: dict[str, Any],
    details: list[dict[str, Any]],
    records: list[dict[str, Any]],
    verifier: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    suspect_samples = per_canary_rates_from_predicate(
        details,
        lambda row: detail_target_gated_hit(row, "watermarked", args.top_k),
        len(records),
    )
    clean_samples = per_canary_rates_from_predicate(
        details,
        lambda row: detail_target_gated_hit(row, "clean", args.top_k),
        len(records),
    )
    suspect_ungated_samples = per_canary_rates_from_predicate(
        details,
        lambda row: detail_response_hit(row, "watermarked"),
        len(records),
    )
    clean_ungated_samples = per_canary_rates_from_predicate(
        details,
        lambda row: detail_response_hit(row, "clean"),
        len(records),
    )
    suspect_strict_samples = per_canary_rates_from_predicate(
        details,
        lambda row: detail_target_gated_hit(row, "watermarked", args.top_k, strict=True),
        len(records),
    )
    clean_strict_samples = per_canary_rates_from_predicate(
        details,
        lambda row: detail_target_gated_hit(row, "clean", args.top_k, strict=True),
        len(records),
    )
    test = verifier.welch_t_test(suspect_samples, clean_samples)
    strict_test = verifier.welch_t_test(suspect_strict_samples, clean_strict_samples)
    ranks = np.asarray([int(d["target_rank"]) for d in details], dtype=float)
    return {
        "profile": profile,
        "label": spec["label"],
        "variant": spec["variant"],
        "context": spec["context"],
        "sealed_config_hash": spec["sealed_config_hash"],
        "num_canaries": len(records),
        "num_queries": len(details),
        "top_k": args.top_k,
        "suspect_cer": float(suspect_samples.mean()) if suspect_samples.size else 0.0,
        "clean_cer": float(clean_samples.mean()) if clean_samples.size else 0.0,
        "suspect_strict_rate": float(suspect_strict_samples.mean()) if suspect_strict_samples.size else 0.0,
        "clean_strict_rate": float(clean_strict_samples.mean()) if clean_strict_samples.size else 0.0,
        "suspect_ungated_response_rate": float(suspect_ungated_samples.mean()) if suspect_ungated_samples.size else 0.0,
        "clean_ungated_response_rate": float(clean_ungated_samples.mean()) if clean_ungated_samples.size else 0.0,
        "p_value": test["p_value"],
        "reject_h0": test["reject_h0"],
        "strict_p_value": strict_test["p_value"],
        "strict_reject_h0": strict_test["reject_h0"],
        "recall_at_1": float(np.mean(ranks <= 1)) if ranks.size else 0.0,
        "recall_at_3": float(np.mean(ranks <= 3)) if ranks.size else 0.0,
        "mean_target_rank": float(ranks.mean()) if ranks.size else 0.0,
        "clean_query_fp_rate": float(clean_ungated_samples.mean()) if clean_ungated_samples.size else 0.0,
        "description": spec["description"],
    }


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_names = [p.strip() for p in args.profiles.split(",") if p.strip()]
    unknown = [p for p in profile_names if p not in PROFILE_DEFS]
    if unknown:
        raise ValueError(f"Unknown profiles: {unknown}; choose from {sorted(PROFILE_DEFS)}")

    sealed = write_sealed_configs(out_dir, profile_names, args)
    if args.seal_only:
        print(json.dumps(sealed, indent=2, ensure_ascii=False))
        return

    records = load_records(resolve(args.records), args.max_records)
    fallback_index = build_fallback_index(args.anchor_fallback_dirs)
    exclude = {resolve(record["watermarked_image_path"]) for record in records}
    for record in records:
        if record.get("anchor_image_path"):
            try:
                exclude.add(resolve_anchor_path(record, fallback_index))
            except FileNotFoundError:
                pass
    distractors = collect_distractors(args.distractor_dirs, args.max_distractors, exclude)
    suspect_entries = build_entries(records, distractors)
    clean_entries = make_clean_entries(records, distractors, fallback_index)

    clip = OpenCLIPScorer(args.device, args.batch_size)
    _, vlm, verifier = build_clients(args.config, args.device)
    verifier.num_probes_per_canary = min(verifier.num_probes_per_canary, 3)

    summary_path = out_dir / "end_to_end_profile_summary.json"
    summaries: list[dict[str, Any]] = []
    if summary_path.exists():
        summaries = json.loads(summary_path.read_text(encoding="utf-8"))
    for profile in profile_names:
        spec = next(item for item in sealed["profiles"] if item["sealed_config"]["profile"] == profile)
        variant = spec["variant"]
        profile_dir = out_dir / profile
        profile_dir.mkdir(parents=True, exist_ok=True)
        print(f"Preparing profile {profile}: {spec['description']}", flush=True)
        suspect_index = prepare_index(
            variant,
            suspect_entries,
            records,
            profile_dir / "suspect",
            args,
            clip,
            sidecar_enabled=(profile == "caption_sidecar"),
        )
        clean_index = prepare_index(
            variant,
            clean_entries,
            records,
            profile_dir / "clean",
            args,
            clip,
            sidecar_enabled=False,
        )

        details_path = profile_dir / "end_to_end_details.jsonl"
        if args.fresh and details_path.exists():
            details_path.unlink()
        details: list[dict[str, Any]] = []
        if details_path.exists():
            details = [json.loads(line) for line in details_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        skip = len(details)
        with details_path.open("a", encoding="utf-8") as fh:
            for record_index, record in enumerate(records):
                signature = str(record["trap_signature"])
                queries = list(record.get("probe_queries", []))[: verifier.num_probes_per_canary]
                for probe_index, query in enumerate(queries):
                    flat_index = record_index * verifier.num_probes_per_canary + probe_index
                    if flat_index < skip:
                        continue
                    target_id = str(record["id"])
                    wm_hits, target_rank = top_hits(suspect_index, variant, str(query), target_id, signature, clip, args)
                    clean_hits, clean_target_rank = top_hits(clean_index, variant, str(query), target_id, signature, clip, args)
                    if spec["context"] == "text_context":
                        wm_docs = suspect_index.caption_texts or ["" for _ in suspect_index.entries]
                        clean_docs = clean_index.caption_texts or ["" for _ in clean_index.entries]
                        wm_prompt = text_context_prompt(str(query), wm_hits, wm_docs)
                        clean_prompt = text_context_prompt(str(query), clean_hits, clean_docs)
                        watermarked_response = vlm.generate(None, wm_prompt, temperature=0.0, max_new_tokens=args.max_new_tokens)
                        clean_response = vlm.generate(None, clean_prompt, temperature=0.0, max_new_tokens=args.max_new_tokens)
                        wm_context_path = None
                        clean_context_path = None
                    else:
                        wm_context_path = make_montage(
                            wm_hits,
                            profile_dir / "contexts" / "watermarked" / f"{record_index:04d}_{probe_index}.jpg",
                            args.montage_tile,
                        )
                        clean_context_path = make_montage(
                            clean_hits,
                            profile_dir / "contexts" / "clean" / f"{record_index:04d}_{probe_index}.jpg",
                            args.montage_tile,
                        )
                        watermarked_response = vlm.generate(
                            Image.open(wm_context_path).convert("RGB"),
                            image_context_prompt(str(query)),
                            temperature=0.0,
                            max_new_tokens=args.max_new_tokens,
                        )
                        clean_response = vlm.generate(
                            Image.open(clean_context_path).convert("RGB"),
                            image_context_prompt(str(query)),
                            temperature=0.0,
                            max_new_tokens=args.max_new_tokens,
                        )
                    wm_score = score_response(watermarked_response, record)
                    clean_score = score_response(clean_response, record)
                    watermarked_target_in_topk = target_rank_in_topk(target_rank, args.top_k)
                    clean_target_in_topk = target_rank_in_topk(clean_target_rank, args.top_k)
                    watermarked_response_hit = bool(wm_score["hit"])
                    clean_response_hit = bool(clean_score["hit"])
                    watermarked_response_strict_hit = bool(wm_score.get("strict_hit", wm_score["hit"]))
                    clean_response_strict_hit = bool(clean_score.get("strict_hit", clean_score["hit"]))
                    detail = {
                        "profile": profile,
                        "record_index": record_index,
                        "probe_index": probe_index,
                        "id": target_id,
                        "mode": wm_score.get("mode"),
                        "signature": signature,
                        "query": str(query),
                        "target_rank": target_rank,
                        "clean_target_rank": clean_target_rank,
                        "watermarked_hits": wm_hits,
                        "clean_hits": clean_hits,
                        "watermarked_context_path": rel(wm_context_path) if wm_context_path else None,
                        "clean_context_path": rel(clean_context_path) if clean_context_path else None,
                        "watermarked_response": watermarked_response,
                        "clean_response": clean_response,
                        "watermarked_score": wm_score,
                        "clean_score": clean_score,
                        "watermarked_target_in_topk": watermarked_target_in_topk,
                        "clean_target_in_topk": clean_target_in_topk,
                        "watermarked_response_hit": watermarked_response_hit,
                        "clean_response_hit": clean_response_hit,
                        "watermarked_response_strict_hit": watermarked_response_strict_hit,
                        "clean_response_strict_hit": clean_response_strict_hit,
                        "watermarked_hit": bool(watermarked_response_hit and watermarked_target_in_topk),
                        "clean_hit": bool(clean_response_hit and clean_target_in_topk),
                        "watermarked_strict_hit": bool(watermarked_response_strict_hit and watermarked_target_in_topk),
                        "clean_strict_hit": bool(clean_response_strict_hit and clean_target_in_topk),
                    }
                    fh.write(json.dumps(detail, ensure_ascii=False) + "\n")
                    fh.flush()
                    details.append(detail)
                    print(
                        f"[{profile} {len(details):03d}/{len(records) * verifier.num_probes_per_canary:03d}] "
                        f"{target_id} rank={target_rank} wm_hit={detail['watermarked_hit']} clean_hit={detail['clean_hit']}",
                        flush=True,
                    )
        summary = summarize_profile(profile, spec, details, records, verifier, args)
        (profile_dir / "end_to_end_report.json").write_text(json.dumps({"summary": summary, "details": details}, indent=2, ensure_ascii=False), encoding="utf-8")
        summaries = [row for row in summaries if row.get("profile") != profile]
        summaries.append(summary)
        summary_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
        with (out_dir / "end_to_end_profile_summary.csv").open("w", newline="", encoding="utf-8") as csv_fh:
            fieldnames = list(dict.fromkeys(key for row in summaries for key in row.keys()))
            writer = csv.DictWriter(csv_fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summaries)
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

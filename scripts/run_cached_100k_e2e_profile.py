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

from run_end_to_end_profiles import build_fallback_index, image_context_prompt, make_montage, resolve_anchor_path
from run_large_clip_retrieval import encode_texts
from run_pipeline_generality import load_records, resolve
from semantitrace.backends.real import QwenVLMClient
from semantitrace.config import load_config
from semantitrace.metrics import contains_positive_signature
from semantitrace.verification import Verifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run visual end-to-end profile from cached 100k CLIP retrieval embeddings.")
    parser.add_argument("--records", default="outputs/semantitrace_n500_textvqa_merged/canary_records_first500_local.json")
    parser.add_argument("--retrieval_cache_dir", default="outputs/pipeline_generality_flux_n500_textvqa_100k_clip")
    parser.add_argument("--output_dir", default="outputs/end_to_end_profiles_100k_n100")
    parser.add_argument("--config", default="configs/semantitrace_flux2_klein_gradient_guided.yaml")
    parser.add_argument("--max_records", type=int, default=100)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--montage_tile", type=int, default=448)
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


def encode_image_paths(paths: list[Path], args: argparse.Namespace, cache_path: Path) -> np.ndarray:
    if cache_path.exists() and not args.fresh:
        return np.load(cache_path, mmap_mode="r")
    import open_clip
    import torch

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", device=device)
    model.eval()
    feats = []
    with torch.no_grad():
        for start in range(0, len(paths), args.batch_size):
            batch = [preprocess(Image.open(p).convert("RGB")) for p in paths[start : start + args.batch_size]]
            pixels = torch.stack(batch).to(device)
            emb = model.encode_image(pixels)
            emb = torch.nn.functional.normalize(emb.float(), dim=-1)
            feats.append(emb.cpu().numpy())
    arr = np.vstack(feats).astype("float32")
    np.save(cache_path, arr)
    return arr


def top_hits(
    scores: np.ndarray,
    entries: list[dict[str, Any]],
    *,
    top_k: int,
    target_idx: int,
) -> tuple[list[dict[str, Any]], int]:
    if top_k >= len(scores):
        top = np.argsort(-scores)
    else:
        cand = np.argpartition(-scores, top_k - 1)[:top_k]
        top = cand[np.argsort(-scores[cand])]
    hits = []
    for rank, idx in enumerate(top[:top_k], start=1):
        entry = entries[int(idx)]
        hits.append(
            {
                "image_id": entry["image_id"],
                "image_path": entry["path"],
                "role": entry["role"],
                "record_id": entry.get("record_id"),
                "entry_index": int(idx),
                "rank": rank,
                "score": float(scores[int(idx)]),
            }
        )
    target_rank = int(np.count_nonzero(scores > scores[target_idx]) + 1)
    return hits, target_rank


def build_vlm(config_path: str, device: str) -> tuple[QwenVLMClient, Verifier]:
    cfg = load_config(config_path)
    models = cfg.get("models", {})
    vlm_cfg = cfg.get("vlm", {})
    vlm = QwenVLMClient(
        model_name=models.get("surrogate_vlm", "Qwen/Qwen3-VL-8B-Instruct"),
        device=device,
        torch_dtype=vlm_cfg.get("torch_dtype", "bfloat16"),
    )
    verifier = Verifier(cfg.get("verification", {}))
    verifier.num_probes_per_canary = min(verifier.num_probes_per_canary, 3)
    return vlm, verifier


def summarize(details: list[dict[str, Any]], records: list[dict[str, Any]], verifier: Verifier, args: argparse.Namespace) -> dict[str, Any]:
    signatures = [str(r["trap_signature"]) for r in records]
    suspect_responses = [str(d["watermarked_response"]) for d in details]
    clean_responses = [str(d["clean_response"]) for d in details]
    suspect_samples = verifier.compute_per_canary_cer(suspect_responses, signatures)
    clean_samples = verifier.compute_per_canary_cer(clean_responses, signatures)
    test = verifier.welch_t_test(suspect_samples, clean_samples)
    boot = verifier.bootstrap_rate_test(suspect_samples, clean_samples)
    ranks = np.asarray([int(d["target_rank"]) for d in details], dtype=float)
    clean_hits = np.asarray([bool(d["clean_hit"]) for d in details], dtype=bool)
    return {
        "profile": "visual_clip_100k",
        "label": "Visual RAG 100k",
        "variant": "clip_visual_100k",
        "context": "image_montage",
        "num_canaries": len(records),
        "num_queries": len(details),
        "index_size": int(args.max_records + 100000),
        "top_k": args.top_k,
        "suspect_cer": float(suspect_samples.mean()) if suspect_samples.size else 0.0,
        "clean_cer": float(clean_samples.mean()) if clean_samples.size else 0.0,
        "p_value": test["p_value"],
        "bootstrap_p_value": boot["p_value"],
        "effect_size": boot["effect_size"],
        "effect_ci95_low": boot["effect_ci95_low"],
        "effect_ci95_high": boot["effect_ci95_high"],
        "reject_h0": test["reject_h0"],
        "recall_at_1": float(np.mean(ranks <= 1)) if ranks.size else 0.0,
        "recall_at_3": float(np.mean(ranks <= 3)) if ranks.size else 0.0,
        "recall_at_10": float(np.mean(ranks <= 10)) if ranks.size else 0.0,
        "mean_target_rank": float(ranks.mean()) if ranks.size else 0.0,
        "clean_query_fp_rate": float(clean_hits.mean()) if clean_hits.size else 0.0,
        "description": "Cached 100k CLIP visual index -> top-k image context -> Qwen3-VL generation",
    }


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_dir = out_dir / "visual_clip_100k"
    profile_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(resolve(args.records), args.max_records)
    cache_dir = resolve(args.retrieval_cache_dir)
    cache_entries = json.loads((cache_dir / "clip_image_entries.json").read_text(encoding="utf-8"))
    cache_emb = np.load(cache_dir / "clip_image_embeddings.npy", mmap_mode="r")
    distractor_entries = [e for e in cache_entries if e.get("role") == "distractor"]
    distractor_start = next(i for i, e in enumerate(cache_entries) if e.get("role") == "distractor")
    distractor_emb = cache_emb[distractor_start:]
    id_to_cache = {str(e.get("record_id")): i for i, e in enumerate(cache_entries) if e.get("record_id")}

    suspect_entries: list[dict[str, Any]] = []
    suspect_embs = []
    for record in records:
        rid = str(record["id"])
        idx = id_to_cache[rid]
        suspect_entries.append(cache_entries[idx])
        suspect_embs.append(np.asarray(cache_emb[idx], dtype="float32"))
    suspect_entries.extend(distractor_entries)
    suspect_emb = np.vstack([np.vstack(suspect_embs), np.asarray(distractor_emb, dtype="float32")])
    suspect_target_index = {str(record["id"]): i for i, record in enumerate(records)}

    fallback = build_fallback_index(args.anchor_fallback_dirs)
    clean_paths = [resolve_anchor_path(record, fallback) for record in records]
    clean_entries = [
        {"image_id": str(record["id"]), "path": str(path), "role": "clean_control", "record_id": str(record["id"])}
        for record, path in zip(records, clean_paths)
    ]
    clean_entries.extend(distractor_entries)
    clean_anchor_emb = encode_image_paths(clean_paths, args, profile_dir / "clean_anchor_embeddings.npy")
    clean_emb = np.vstack([np.asarray(clean_anchor_emb, dtype="float32"), np.asarray(distractor_emb, dtype="float32")])
    clean_target_index = {str(record["id"]): i for i, record in enumerate(records)}

    query_tuples: list[tuple[int, int, str]] = []
    for record_index, record in enumerate(records):
        for probe_index, query in enumerate(list(record.get("probe_queries", []))[:3]):
            query_tuples.append((record_index, probe_index, str(query)))
    text_emb = encode_texts([q for _, _, q in query_tuples], args)

    vlm, verifier = build_vlm(args.config, args.device)
    details_path = profile_dir / "end_to_end_details.jsonl"
    if args.fresh and details_path.exists():
        details_path.unlink()
    details = []
    if details_path.exists():
        details = [json.loads(line) for line in details_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    skip = len(details)
    with details_path.open("a", encoding="utf-8") as fh:
        for flat_idx, (record_index, probe_index, query) in enumerate(query_tuples):
            if flat_idx < skip:
                continue
            record = records[record_index]
            rid = str(record["id"])
            signature = str(record["trap_signature"])
            q_emb = text_emb[flat_idx]
            wm_scores = np.asarray(suspect_emb @ q_emb, dtype=np.float32)
            clean_scores = np.asarray(clean_emb @ q_emb, dtype=np.float32)
            wm_hits, target_rank = top_hits(wm_scores, suspect_entries, top_k=args.top_k, target_idx=suspect_target_index[rid])
            clean_hits, clean_target_rank = top_hits(clean_scores, clean_entries, top_k=args.top_k, target_idx=clean_target_index[rid])
            wm_context_path = make_montage(wm_hits, profile_dir / "contexts" / "watermarked" / f"{record_index:04d}_{probe_index}.jpg", args.montage_tile)
            clean_context_path = make_montage(clean_hits, profile_dir / "contexts" / "clean" / f"{record_index:04d}_{probe_index}.jpg", args.montage_tile)
            prompt = image_context_prompt(query)
            watermarked_response = vlm.generate(Image.open(wm_context_path).convert("RGB"), prompt, temperature=0.0, max_new_tokens=args.max_new_tokens)
            clean_response = vlm.generate(Image.open(clean_context_path).convert("RGB"), prompt, temperature=0.0, max_new_tokens=args.max_new_tokens)
            detail = {
                "profile": "visual_clip_100k",
                "record_index": record_index,
                "probe_index": probe_index,
                "id": rid,
                "signature": signature,
                "query": query,
                "target_rank": target_rank,
                "clean_target_rank": clean_target_rank,
                "watermarked_hits": wm_hits,
                "clean_hits": clean_hits,
                "watermarked_context_path": str(wm_context_path.relative_to(ROOT)),
                "clean_context_path": str(clean_context_path.relative_to(ROOT)),
                "watermarked_response": watermarked_response,
                "clean_response": clean_response,
                "watermarked_hit": contains_positive_signature(watermarked_response, signature),
                "clean_hit": contains_positive_signature(clean_response, signature),
            }
            fh.write(json.dumps(detail, ensure_ascii=False) + "\n")
            fh.flush()
            details.append(detail)
            print(f"[{len(details):03d}/{len(query_tuples):03d}] {rid} rank={target_rank} wm_hit={detail['watermarked_hit']} clean_hit={detail['clean_hit']}", flush=True)

    summary = summarize(details, records, verifier, args)
    (profile_dir / "end_to_end_report.json").write_text(json.dumps({"summary": summary, "details": details}, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "end_to_end_profile_summary.json").write_text(json.dumps([summary], indent=2, ensure_ascii=False), encoding="utf-8")
    with (out_dir / "end_to_end_profile_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

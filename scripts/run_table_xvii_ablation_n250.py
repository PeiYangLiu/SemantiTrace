#!/usr/bin/env python
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantitrace.backends.real import Flux2KleinInpaintEditor, OpenCLIPEncoder, QwenVLMClient
from semantitrace.defenses import MahalanobisOODDetector
from semantitrace.metrics import compute_psnr
from semantitrace.mode_verification import score_response
from semantitrace.records import default_record_roots, infer_record_mode, resolve_record_path, resolve_repo_path
from semantitrace.utils.image import mask_from_bbox


ROOT = Path(__file__).resolve().parents[1]

VARIANTS: dict[str, dict[str, Any]] = {
    "without_ret": {
        "table_label": "w/o Ret. Guide",
        "kind": "generate",
        "lambda_ret": 0.0,
        "lambda_gen": 4.0,
        "enable_gradient_guidance": True,
        "masked_blending_enforced": True,
    },
    "without_sig": {
        "table_label": "w/o Sig. Guide",
        "kind": "generate",
        "lambda_ret": 2.5,
        "lambda_gen": 0.0,
        "enable_gradient_guidance": True,
        "masked_blending_enforced": True,
    },
    "without_blending": {
        "table_label": "w/o Latent Blending",
        "kind": "generate",
        "lambda_ret": 2.5,
        "lambda_gen": 4.0,
        "enable_gradient_guidance": True,
        "masked_blending_enforced": False,
    },
    "full_semantitrace": {
        "table_label": "Full SemantiTrace",
        "kind": "canonical",
        "lambda_ret": 2.5,
        "lambda_gen": 4.0,
        "enable_gradient_guidance": True,
        "masked_blending_enforced": True,
    },
}

DEFAULT_VARIANTS = "without_ret,without_sig,without_blending,full_semantitrace"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run Table XVII n=250 component ablation with artifact-backed metrics")
    parser.add_argument("--records", default="n_scaling_subsets_20260609/combined_a125_b125_n250.json")
    parser.add_argument("--image_root", default="amlt_combined_a500_b500_records")
    parser.add_argument("--output_dir", default="outputs/table_xvii_ablation_n250")
    parser.add_argument("--config", default="configs/semantitrace_flux2_klein_gradient_guided.yaml")
    parser.add_argument("--variants", default=DEFAULT_VARIANTS)
    parser.add_argument("--max_records", type=int, default=250)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=96)
    parser.add_argument("--generate_only", action="store_true")
    parser.add_argument("--evaluate_only", action="store_true")
    parser.add_argument("--skip_qwen", action="store_true")
    parser.add_argument("--closed_api_url", default=None)
    parser.add_argument("--closed_api_mode", default="auto", choices=("auto", "openai", "rag-query"))
    parser.add_argument("--closed_api_model", default="copilot")
    parser.add_argument("--closed_api_key", default=None)
    parser.add_argument("--closed_api_timeout", type=float, default=120.0)
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(resolve_repo_path(path).read_text(encoding="utf-8"))


def parse_variants(text: str) -> list[str]:
    variants = [item.strip() for item in text.split(",") if item.strip()]
    unknown = [variant for variant in variants if variant not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants {unknown}; choose from {sorted(VARIANTS)}")
    return variants


def load_source_records(records_path: str | Path, image_root: str | Path, max_records: int | None) -> list[dict[str, Any]]:
    resolved_records = resolve_repo_path(records_path)
    roots = default_record_roots(resolved_records, image_root)
    rows = json.loads(resolved_records.read_text(encoding="utf-8"))
    if max_records:
        rows = rows[:max_records]
    records: list[dict[str, Any]] = []
    for row in rows:
        record = dict(row)
        record["_resolved_anchor_image_path"] = str(resolve_record_path(record["anchor_image_path"], roots))
        record["_resolved_watermarked_image_path"] = str(resolve_record_path(record["watermarked_image_path"], roots))
        records.append(record)
    return records


def bbox_for_record(record: dict[str, Any]) -> list[int]:
    for container in (
        record.get("injection_metrics"),
        record.get("selected_canvas"),
        record.get("nontext_plan"),
        record.get("anchor"),
    ):
        if isinstance(container, dict):
            bbox = container.get("effective_mask_bbox") or container.get("bbox")
            if bbox and len(bbox) == 4:
                return [int(v) for v in bbox]
    raise ValueError(f"Record {record.get('id')} lacks a valid bbox")


def expanded_crop(image: Image.Image, bbox: list[int], pad_ratio: float = 0.35) -> Image.Image:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    pad = int(max(width, height) * pad_ratio)
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(image.width, x2 + pad)
    y2 = min(image.height, y2 + pad)
    return image.crop((x1, y1, x2, y2))


def direct_prompt(record: dict[str, Any]) -> str:
    mode = infer_record_mode(record)
    if mode == "mode_b":
        return (
            "Inspect this local crop from an image. If a small inserted or salient object is visible, "
            "describe its color, object type, and approximate location in a concise phrase. "
            "If no such object is visible, answer NONE."
        )
    return (
        "Inspect this local crop from an image. Transcribe the exact short text, label, or acronym "
        "that is most prominent in the crop. Return only the visible text; if no text is readable, answer NONE."
    )


def safe_psnr(clean: Image.Image, edited: Image.Image) -> float | str:
    value = compute_psnr(np.asarray(clean.convert("RGB")), np.asarray(edited.convert("RGB").resize(clean.size)))
    return "inf" if math.isinf(value) else float(value)


def mean_abs_delta(clean: Image.Image, edited: Image.Image) -> float:
    clean_arr = np.asarray(clean.convert("RGB"), dtype=np.float32)
    edited_arr = np.asarray(edited.convert("RGB").resize(clean.size), dtype=np.float32)
    return float(np.mean(np.abs(clean_arr - edited_arr)) / 255.0)


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sum": int(value.sum()) if value.dtype == bool or np.issubdtype(value.dtype, np.number) else None,
        }
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def build_editor(config: dict[str, Any], device: str) -> Flux2KleinInpaintEditor:
    models = config.get("models", {})
    editor_cfg = dict(config.get("editor", {}))
    gradient_cfg = dict(editor_cfg.pop("gradient_guidance", {}))
    gradient_cfg["enabled"] = True
    gradient_cfg.setdefault("clip_device", device)
    return Flux2KleinInpaintEditor(
        model_name=models.get("inpaint_model", "black-forest-labs/FLUX.2-klein-9B"),
        device=device,
        **editor_cfg,
        gradient_guidance=gradient_cfg,
    )


def materialize_canonical_variant(
    source_records: list[dict[str, Any]],
    out_dir: Path,
) -> list[dict[str, Any]]:
    variant = "full_semantitrace"
    spec = VARIANTS[variant]
    method_dir = out_dir / variant
    wm_dir = method_dir / "watermarked"
    clean_dir = method_dir / "clean"
    wm_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    records_path = method_dir / "canary_records.json"
    records: list[dict[str, Any]] = []
    if records_path.exists():
        records = json.loads(records_path.read_text(encoding="utf-8"))
    start = len(records)
    if start >= len(source_records):
        return records[: len(source_records)]
    for idx, source in enumerate(source_records[start:], start=start):
        clean_src = Path(source["_resolved_anchor_image_path"])
        wm_src = Path(source["_resolved_watermarked_image_path"])
        clean_path = clean_dir / f"{variant}-{idx:04d}_{clean_src.name}"
        wm_path = wm_dir / f"{variant}-{idx:04d}_{wm_src.name}"
        if not clean_path.exists():
            shutil.copy2(clean_src, clean_path)
        if not wm_path.exists():
            shutil.copy2(wm_src, wm_path)
        record = {
            **{k: v for k, v in source.items() if not k.startswith("_resolved_")},
            "id": f"{variant}-{idx:04d}",
            "method": variant,
            "method_label": spec["table_label"],
            "source_semantitrace_id": source.get("id"),
            "anchor_image_path": rel(clean_path),
            "source_original_anchor_image_path": source.get("anchor_image_path"),
            "watermarked_image_path": rel(wm_path),
        }
        records.append(record)
        records_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    return records


def materialize_generated_variant(
    variant: str,
    source_records: list[dict[str, Any]],
    editor: Flux2KleinInpaintEditor,
    out_dir: Path,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    spec = VARIANTS[variant]
    method_dir = out_dir / variant
    wm_dir = method_dir / "watermarked"
    clean_dir = method_dir / "clean"
    wm_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    records_path = method_dir / "canary_records.json"
    records: list[dict[str, Any]] = []
    if records_path.exists():
        records = json.loads(records_path.read_text(encoding="utf-8"))
    start = len(records)
    if start >= len(source_records):
        return records[: len(source_records)]
    for idx, source in enumerate(source_records[start:], start=start):
        clean_src = Path(source["_resolved_anchor_image_path"])
        clean = Image.open(clean_src).convert("RGB")
        bbox = bbox_for_record(source)
        mask = mask_from_bbox(clean.size, bbox)
        selected_canvas = dict(source.get("selected_canvas") or {})
        selected_canvas["bbox"] = bbox
        guidance = {
            "probe_query": str(source.get("probe_queries", [""])[0]),
            "trap_signature": str(source.get("trap_signature", "")),
            "parasitism_mode": str(source.get("parasitism_mode", "")),
            "selected_canvas": selected_canvas,
            "lambda_ret": float(spec["lambda_ret"]),
            "lambda_gen": float(spec["lambda_gen"]),
            "guidance_scale": float(config.get("editor", {}).get("guidance_scale", 8.0)),
            "strength": float(config.get("editor", {}).get("strength", 0.92)),
            "num_inference_steps": int(config.get("editor", {}).get("num_inference_steps", 40)),
            "enable_gradient_guidance": bool(spec["enable_gradient_guidance"]),
            "masked_blending_enforced": bool(spec["masked_blending_enforced"]),
            "edit_attempt": 0,
        }
        edited = editor.edit(clean, mask, str(source.get("trigger_prompt", "")), guidance)
        clean_path = clean_dir / f"{variant}-{idx:04d}_{clean_src.name}"
        wm_path = wm_dir / f"{variant}-{idx:04d}_{str(source.get('trap_signature', 'sig')).replace(' ', '_')}.png"
        if not clean_path.exists():
            shutil.copy2(clean_src, clean_path)
        edited.save(wm_path)
        metrics = jsonable(dict(guidance))
        metrics["quality_local_delta"] = mean_abs_delta(clean, edited)
        metrics["psnr"] = safe_psnr(clean, edited)
        record = {
            **{k: v for k, v in source.items() if not k.startswith("_resolved_")},
            "id": f"{variant}-{idx:04d}",
            "method": variant,
            "method_label": spec["table_label"],
            "source_semantitrace_id": source.get("id"),
            "anchor_image_path": rel(clean_path),
            "source_original_anchor_image_path": source.get("anchor_image_path"),
            "watermarked_image_path": rel(wm_path),
            "selected_canvas": selected_canvas,
            "injection_metrics": metrics,
        }
        records.append(record)
        records_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[{variant}] generated {idx + 1}/{len(source_records)} {source.get('id')}", flush=True)
    return records


def materialize_variants(
    variants: list[str],
    source_records: list[dict[str, Any]],
    out_dir: Path,
    config: dict[str, Any],
    device: str,
) -> dict[str, list[dict[str, Any]]]:
    records_by_variant: dict[str, list[dict[str, Any]]] = {}
    generated_variants = [variant for variant in variants if VARIANTS[variant]["kind"] == "generate"]
    editor: Flux2KleinInpaintEditor | None = None
    if generated_variants:
        editor = build_editor(config, device)
    for variant in variants:
        if VARIANTS[variant]["kind"] == "canonical":
            records_by_variant[variant] = materialize_canonical_variant(source_records, out_dir)
        else:
            assert editor is not None
            records_by_variant[variant] = materialize_generated_variant(variant, source_records, editor, out_dir, config)
    return records_by_variant


def load_variant_records(out_dir: Path, variants: list[str]) -> dict[str, list[dict[str, Any]]]:
    records_by_variant: dict[str, list[dict[str, Any]]] = {}
    for variant in variants:
        path = out_dir / variant / "canary_records.json"
        records_by_variant[variant] = json.loads(path.read_text(encoding="utf-8"))
    return records_by_variant


def resolve_output_image_path(value: str | Path, method_dir: Path, subdir: str) -> Path:
    p = Path(value)
    if p.is_absolute() and p.exists():
        return p
    candidates: list[Path] = []
    if p.is_absolute():
        candidates.extend([method_dir / subdir / p.name, method_dir / p.name])
    else:
        candidates.extend([ROOT / p, method_dir / p, method_dir.parent / p, method_dir / subdir / p.name])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_variant_image_paths(records: list[dict[str, Any]], method_dir: Path) -> tuple[list[Path], list[Path]]:
    clean_paths: list[Path] = []
    wm_paths: list[Path] = []
    for record in records:
        clean_paths.append(resolve_output_image_path(record["anchor_image_path"], method_dir, "clean"))
        wm_paths.append(resolve_output_image_path(record["watermarked_image_path"], method_dir, "watermarked"))
    return clean_paths, wm_paths


def compute_lpips_scores(clean_paths: list[Path], wm_paths: list[Path], device: str) -> list[float]:
    import lpips
    import torch
    from torchvision import transforms

    loss_fn = lpips.LPIPS(net="alex").to(device)
    to_tensor = transforms.ToTensor()
    scores: list[float] = []
    with torch.no_grad():
        for clean_path, wm_path in zip(clean_paths, wm_paths, strict=True):
            clean = Image.open(clean_path).convert("RGB")
            wm = Image.open(wm_path).convert("RGB").resize(clean.size)
            clean_tensor = to_tensor(clean).unsqueeze(0).to(device) * 2.0 - 1.0
            wm_tensor = to_tensor(wm).unsqueeze(0).to(device) * 2.0 - 1.0
            scores.append(float(loss_fn(clean_tensor, wm_tensor).detach().cpu().item()))
    del loss_fn
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return scores


def compute_retrieval_details(
    encoder: OpenCLIPEncoder,
    records: list[dict[str, Any]],
    wm_paths: list[Path],
    source_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    images = [Image.open(path).convert("RGB") for path in wm_paths]
    image_emb = encoder.encode_images(images)
    details: list[dict[str, Any]] = []
    all_ranks: list[int] = []
    for record_index, source in enumerate(source_records):
        queries = [str(q) for q in source.get("probe_queries", [])]
        if not queries:
            queries = [str(source.get("trigger_prompt", ""))]
        text_emb = encoder.encode_texts(queries)
        sims = text_emb @ image_emb.T
        for probe_index, query in enumerate(queries):
            target_score = float(sims[probe_index, record_index])
            rank = int(1 + np.sum(sims[probe_index] > target_score))
            all_ranks.append(rank)
            details.append(
                {
                    "record_index": record_index,
                    "probe_index": probe_index,
                    "id": records[record_index]["id"],
                    "source_semantitrace_id": source.get("id"),
                    "query": query,
                    "target_rank": rank,
                    "target_score": target_score,
                }
            )
    return details, np.asarray(all_ranks, dtype=np.float64)


def evaluate_qwen_direct(
    vlm: QwenVLMClient,
    source_records: list[dict[str, Any]],
    wm_paths: list[Path],
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for record_index, (source, wm_path) in enumerate(zip(source_records, wm_paths, strict=True)):
        image = Image.open(wm_path).convert("RGB")
        bbox = bbox_for_record(source)
        crop = expanded_crop(image, bbox)
        response = vlm.generate(crop, direct_prompt(source), temperature=0.0, max_new_tokens=max_new_tokens)
        scored = score_response(response, source)
        details.append(
            {
                "record_index": record_index,
                "source_semantitrace_id": source.get("id"),
                "response": response,
                "hit": bool(scored.get("hit", False)),
                "strict_hit": bool(scored.get("strict_hit", scored.get("hit", False))),
                "score": scored,
            }
        )
        print(
            f"[qwen direct {record_index + 1:03d}/{len(source_records):03d}] "
            f"{source.get('id')} hit={bool(scored.get('hit', False))}",
            flush=True,
        )
    return details


def image_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def normalize_chat_url(url: str) -> str:
    url = url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"


def normalize_rag_query_url(url: str) -> str:
    url = url.rstrip("/")
    if url.endswith("/rag-query"):
        return url
    if url.endswith("/v1"):
        return f"{url}/rag-query"
    return f"{url}/v1/rag-query"


def resolve_closed_api_url(arg_url: str | None) -> str | None:
    if arg_url:
        return arg_url
    for key in (
        "SEMANTITRACE_COPILOT_ENDPOINT",
        "COPILOT_API_URL",
        "COPILOT_ENDPOINT",
        "SEMANTITRACE_GPT52_ENDPOINT",
        "SEMANTITRACE_GEMINI3_ENDPOINT",
    ):
        value = os.environ.get(key)
        if value:
            return value
    return None


def resolve_closed_api_mode(url: str, mode: str) -> str:
    if mode != "auto":
        return mode
    return "rag-query" if url.rstrip("/").endswith("/rag-query") else "openai"


def call_closed_api_openai(
    url: str,
    model: str,
    api_key: str | None,
    image: Image.Image,
    prompt: str,
    timeout: float,
) -> str:
    import requests

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_to_data_url(image)}},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 96,
    }
    response = requests.post(normalize_chat_url(url), headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return str(data["choices"][0]["message"]["content"])


def call_closed_api_rag_query(
    url: str,
    api_key: str | None,
    image_path: Path,
    prompt: str,
    timeout: float,
) -> str:
    import requests

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "query": prompt,
        "retrieved": [
            {
                "image_path": str(image_path),
                "rank": 1,
                "score": 1.0,
            }
        ],
        "temperature": 0.0,
        "max_new_tokens": 96,
    }
    response = requests.post(normalize_rag_query_url(url), headers=headers, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return str(data.get("response") or data.get("text") or data.get("answer") or data)


def evaluate_closed_api_direct(
    source_records: list[dict[str, Any]],
    wm_paths: list[Path],
    url: str,
    mode: str,
    model: str,
    api_key: str | None,
    timeout: float,
    crop_dir: Path,
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    api_mode = resolve_closed_api_mode(url, mode)
    crop_dir.mkdir(parents=True, exist_ok=True)
    for record_index, (source, wm_path) in enumerate(zip(source_records, wm_paths, strict=True)):
        image = Image.open(wm_path).convert("RGB")
        bbox = bbox_for_record(source)
        crop = expanded_crop(image, bbox)
        prompt = direct_prompt(source)
        if api_mode == "rag-query":
            crop_path = crop_dir / f"crop-{record_index:04d}-{str(source.get('id', 'record'))}.png"
            crop.save(crop_path)
            response = call_closed_api_rag_query(url, api_key, crop_path, prompt, timeout)
        else:
            response = call_closed_api_openai(url, model, api_key, crop, prompt, timeout)
        scored = score_response(response, source)
        details.append(
            {
                "record_index": record_index,
                "source_semantitrace_id": source.get("id"),
                "api_mode": api_mode,
                "response": response,
                "hit": bool(scored.get("hit", False)),
                "strict_hit": bool(scored.get("strict_hit", scored.get("hit", False))),
                "score": scored,
            }
        )
        print(
            f"[closed api direct {record_index + 1:03d}/{len(source_records):03d}] "
            f"{source.get('id')} hit={bool(scored.get('hit', False))}",
            flush=True,
        )
    return details


def combine_direct_with_retrieval(
    retrieval_details: list[dict[str, Any]],
    direct_details: list[dict[str, Any]] | None,
    top_k: int,
) -> float | None:
    if direct_details is None:
        return None
    direct_hits = {int(detail["record_index"]): bool(detail.get("hit", False)) for detail in direct_details}
    effective = [
        bool(direct_hits.get(int(detail["record_index"]), False)) and int(detail["target_rank"]) <= top_k
        for detail in retrieval_details
    ]
    return float(np.mean(effective)) if effective else 0.0


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def evaluate_variants(
    variants: list[str],
    source_records: list[dict[str, Any]],
    records_by_variant: dict[str, list[dict[str, Any]]],
    out_dir: Path,
    config: dict[str, Any],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    models = config.get("models", {})
    defenses_cfg = config.get("defenses", {})
    encoder = OpenCLIPEncoder(
        model_name=models.get("clip_model", "ViT-L-14"),
        pretrained=models.get("clip_pretrained", "openai"),
        device=args.device,
        batch_size=64,
    )
    clean_paths, _ = resolve_variant_image_paths(records_by_variant[variants[0]], out_dir / variants[0])
    clean_images = [Image.open(path).convert("RGB") for path in clean_paths]
    clean_embeddings = encoder.encode_images(clean_images)
    detector = MahalanobisOODDetector(
        percentile=float(defenses_cfg.get("mahalanobis_percentile", 99.0)),
        regularization=float(defenses_cfg.get("mahalanobis_regularization", 1e-4)),
        max_components=int(defenses_cfg.get("mahalanobis_max_components", 64)),
        variance_keep=float(defenses_cfg.get("mahalanobis_variance_keep", 0.95)),
    ).fit(clean_embeddings)
    clean_reject_rate = float(np.mean(detector.reject_embeddings(clean_embeddings)))

    per_variant_paths: dict[str, tuple[list[Path], list[Path]]] = {}
    retrieval_by_variant: dict[str, list[dict[str, Any]]] = {}
    summary_rows: list[dict[str, Any]] = []
    for variant in variants:
        method_dir = out_dir / variant
        records = records_by_variant[variant]
        clean_variant_paths, wm_paths = resolve_variant_image_paths(records, method_dir)
        per_variant_paths[variant] = (clean_variant_paths, wm_paths)
        lpips_scores = compute_lpips_scores(clean_variant_paths, wm_paths, args.device)
        wm_images = [Image.open(path).convert("RGB") for path in wm_paths]
        suspect_embeddings = encoder.encode_images(wm_images)
        suspect_ood = detector.reject_embeddings(suspect_embeddings)
        retrieval_details, ranks = compute_retrieval_details(encoder, records, wm_paths, source_records)
        retrieval_by_variant[variant] = retrieval_details
        write_jsonl(method_dir / "retrieval_rank_details.jsonl", retrieval_details)
        row = {
            "method": variant,
            "label": VARIANTS[variant]["table_label"],
            "num_canaries": len(records),
            "num_queries": len(retrieval_details),
            "mean_target_rank": float(ranks.mean()) if ranks.size else None,
            "median_target_rank": float(np.median(ranks)) if ranks.size else None,
            "recall_at_1": float(np.mean(ranks <= 1)) if ranks.size else None,
            "recall_at_3": float(np.mean(ranks <= 3)) if ranks.size else None,
            "recall_at_10": float(np.mean(ranks <= 10)) if ranks.size else None,
            "lpips_mean": float(np.mean(lpips_scores)) if lpips_scores else None,
            "lpips_median": float(np.median(lpips_scores)) if lpips_scores else None,
            "lpips_max": float(np.max(lpips_scores)) if lpips_scores else None,
            "ood_reject_rate": float(np.mean(suspect_ood)) if len(suspect_ood) else None,
            "clean_reject_rate": clean_reject_rate,
            "ood_threshold": float(detector.threshold) if detector.threshold is not None else None,
            "qwen_direct_cgsr": None,
            "qwen_effective_cgsr_at_top3": None,
            "api_direct_cgsr": None,
            "api_effective_cgsr_at_top3": None,
        }
        (method_dir / "lpips_scores.json").write_text(json.dumps(lpips_scores, indent=2), encoding="utf-8")
        summary_rows.append(row)

    del encoder
    try:
        import torch

        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    except Exception:
        pass

    if not args.skip_qwen:
        vlm = QwenVLMClient(
            model_name=models.get("surrogate_vlm", "Qwen/Qwen3-VL-8B-Instruct"),
            device=args.device,
            torch_dtype=config.get("vlm", {}).get("torch_dtype", "bfloat16"),
        )
        for row in summary_rows:
            variant = row["method"]
            _, wm_paths = per_variant_paths[variant]
            details = evaluate_qwen_direct(vlm, source_records, wm_paths, args.max_new_tokens)
            write_jsonl(out_dir / variant / "qwen_direct_details.jsonl", details)
            row["qwen_direct_cgsr"] = float(np.mean([bool(detail["hit"]) for detail in details])) if details else 0.0
            row["qwen_effective_cgsr_at_top3"] = combine_direct_with_retrieval(
                retrieval_by_variant[variant], details, args.top_k
            )
        del vlm

    closed_api_url = resolve_closed_api_url(args.closed_api_url)
    if closed_api_url:
        for row in summary_rows:
            variant = row["method"]
            _, wm_paths = per_variant_paths[variant]
            details = evaluate_closed_api_direct(
                source_records,
                wm_paths,
                closed_api_url,
                args.closed_api_mode,
                args.closed_api_model,
                args.closed_api_key,
                args.closed_api_timeout,
                out_dir / variant / "closed_api_crops",
            )
            write_jsonl(out_dir / variant / "closed_api_direct_details.jsonl", details)
            row["api_direct_cgsr"] = float(np.mean([bool(detail["hit"]) for detail in details])) if details else 0.0
            row["api_effective_cgsr_at_top3"] = combine_direct_with_retrieval(
                retrieval_by_variant[variant], details, args.top_k
            )

    (out_dir / "table_xvii_ablation_summary.json").write_text(
        json.dumps(summary_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary_rows


def main() -> None:
    args = parse_args()
    out_dir = resolve_repo_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = parse_variants(args.variants)
    config = load_config(args.config)
    source_records = load_source_records(args.records, args.image_root, args.max_records)
    if len(source_records) == 0:
        raise ValueError("No source records loaded")

    if args.evaluate_only:
        records_by_variant = load_variant_records(out_dir, variants)
    else:
        records_by_variant = materialize_variants(variants, source_records, out_dir, config, args.device)

    if args.generate_only:
        return

    summary_rows = evaluate_variants(variants, source_records, records_by_variant, out_dir, config, args)
    print(json.dumps(summary_rows, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

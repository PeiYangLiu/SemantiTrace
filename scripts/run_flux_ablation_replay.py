#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_final_rag_verify import build_clients, rag_prompt, resolve_path
from semantitrace.backends.real import Flux2KleinInpaintEditor
from semantitrace.metrics import compute_psnr, contains_positive_signature
from semantitrace.rag import ImageRAGIndex
from semantitrace.utils.image import mask_from_bbox


ROOT = Path(__file__).resolve().parents[1]


VARIANTS = {
    "unguided_flux": {
        "label": "Standard FLUX inpaint",
        "lambda_ret": 2.5,
        "lambda_gen": 4.0,
        "enable_gradient_guidance": False,
    },
    "ret_only_flux": {
        "label": "FLUX + retrieval guidance",
        "lambda_ret": 2.5,
        "lambda_gen": 0.0,
        "enable_gradient_guidance": True,
    },
    "sig_only_flux": {
        "label": "FLUX + signature guidance",
        "lambda_ret": 0.0,
        "lambda_gen": 4.0,
        "enable_gradient_guidance": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Replay FLUX ablation baselines on existing SemantiTrace records")
    parser.add_argument("--source_records", default="outputs/flux2_klein_gradient_guided_n100_tuned_v1_rag_verify/canary_records.json")
    parser.add_argument("--semantitrace_report", default="outputs/flux2_klein_gradient_guided_n100_tuned_v1_rag_verify/rag_verify_report.json")
    parser.add_argument("--output_dir", default="outputs/flux2_klein_gradient_guided_n100_tuned_v1_flux_ablation")
    parser.add_argument("--config", default="configs/semantitrace_flux2_klein_gradient_guided.yaml")
    parser.add_argument("--variants", default="unguided_flux,ret_only_flux,sig_only_flux")
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--generate_only", action="store_true")
    parser.add_argument("--verify_only", action="store_true")
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_source_records(path: str, max_records: int | None) -> list[dict[str, Any]]:
    records = json.loads(resolve_path(path).read_text(encoding="utf-8"))
    return records[:max_records] if max_records is not None else records


def load_config(path: str) -> dict[str, Any]:
    return yaml.safe_load(resolve_path(path).read_text(encoding="utf-8"))


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


def bbox_for_record(record: dict[str, Any]) -> list[int]:
    metrics = record.get("injection_metrics") if isinstance(record.get("injection_metrics"), dict) else {}
    bbox = metrics.get("effective_mask_bbox")
    if bbox and len(bbox) == 4:
        return [int(v) for v in bbox]
    canvas = record.get("selected_canvas") if isinstance(record.get("selected_canvas"), dict) else {}
    bbox = canvas.get("bbox") if canvas else None
    if bbox and len(bbox) == 4:
        return [int(v) for v in bbox]
    raise ValueError(f"Record {record.get('id')} lacks a valid bbox")


def safe_psnr(clean: Image.Image, edited: Image.Image) -> float | str:
    value = compute_psnr(np.asarray(clean.convert("RGB")), np.asarray(edited.convert("RGB").resize(clean.size)))
    return "inf" if math.isinf(value) else float(value)


def quality_delta(clean: Image.Image, edited: Image.Image) -> float:
    a = np.asarray(clean.convert("RGB"), dtype=np.float32)
    b = np.asarray(edited.convert("RGB").resize(clean.size), dtype=np.float32)
    return float(np.mean(np.abs(a - b)) / 255.0)


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
        return [jsonable(v) for v in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def materialize_variant(
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
    if records_path.exists():
        return json.loads(records_path.read_text(encoding="utf-8"))

    records: list[dict[str, Any]] = []
    for idx, source in enumerate(source_records):
        clean_src = resolve_path(source["anchor_image_path"])
        clean = Image.open(clean_src).convert("RGB")
        bbox = bbox_for_record(source)
        mask = mask_from_bbox(clean.size, bbox)
        signature = str(source["trap_signature"])
        prompt = str(source["trigger_prompt"])
        selected_canvas = dict(source.get("selected_canvas") or {})
        selected_canvas["bbox"] = bbox
        guidance = {
            "probe_query": str(source.get("probe_queries", [""])[0]),
            "trap_signature": signature,
            "parasitism_mode": str(source.get("parasitism_mode", "")),
            "selected_canvas": selected_canvas,
            "lambda_ret": float(spec["lambda_ret"]),
            "lambda_gen": float(spec["lambda_gen"]),
            "guidance_scale": float(config.get("editor", {}).get("guidance_scale", 8.0)),
            "strength": float(config.get("editor", {}).get("strength", 0.92)),
            "num_inference_steps": int(config.get("editor", {}).get("num_inference_steps", 40)),
            "enable_gradient_guidance": bool(spec["enable_gradient_guidance"]),
            "edit_attempt": 0,
        }
        edited = editor.edit(clean, mask, prompt, guidance)
        clean_path = clean_dir / f"{variant}-{idx:04d}_{clean_src.name}"
        wm_path = wm_dir / f"{variant}-{idx:04d}_{signature}.png"
        if not clean_path.exists():
            shutil.copy2(clean_src, clean_path)
        edited.save(wm_path)
        metrics = jsonable(dict(guidance))
        metrics["quality_local_delta"] = quality_delta(clean, edited)
        metrics["psnr"] = safe_psnr(clean, edited)
        record = {
            "id": f"{variant}-{idx:04d}",
            "method": variant,
            "method_label": spec["label"],
            "source_semantitrace_id": source.get("id"),
            "anchor_image_path": rel(clean_path),
            "source_original_anchor_image_path": source["anchor_image_path"],
            "watermarked_image_path": rel(wm_path),
            "selected_box_id": source.get("selected_box_id", 0),
            "selected_canvas": selected_canvas,
            "parasitism_mode": source.get("parasitism_mode"),
            "trigger_prompt": prompt,
            "trap_signature": signature,
            "probe_queries": list(source.get("probe_queries", [])),
            "reasoning": f"FLUX replay baseline: {variant}",
            "injection_metrics": metrics,
        }
        records.append(record)
        records_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[{variant}] generated {idx + 1}/{len(source_records)} {signature}", flush=True)
    return records


def load_clean_cache(report_path: str, expected_count: int) -> list[dict[str, Any]]:
    report = json.loads(resolve_path(report_path).read_text(encoding="utf-8"))
    details = report.get("details", [])
    if len(details) < expected_count:
        raise ValueError(f"Clean cache has {len(details)} details, expected at least {expected_count}")
    return details[:expected_count]


def verify_variant(
    variant: str,
    records: list[dict[str, Any]],
    clean_cache: list[dict[str, Any]],
    encoder: Any,
    vlm: Any,
    verifier: Any,
    out_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    method_dir = out_dir / variant
    details_path = method_dir / "rag_verify_details.jsonl"
    details: list[dict[str, Any]] = []
    if details_path.exists():
        with details_path.open("r", encoding="utf-8") as fh:
            details = [json.loads(line) for line in fh if line.strip()]
    index = ImageRAGIndex(encoder).build([resolve_path(r["watermarked_image_path"]) for r in records], [r["id"] for r in records])
    skip = len(details)
    with details_path.open("a", encoding="utf-8") as out:
        for record_index, record in enumerate(records):
            signature = str(record["trap_signature"])
            queries = list(record.get("probe_queries", []))[: verifier.num_probes_per_canary]
            for probe_index, query in enumerate(queries):
                flat_index = record_index * verifier.num_probes_per_canary + probe_index
                if flat_index < skip:
                    continue
                hits = index.search(str(query), args.top_k)
                image = Image.open(hits[0].image_path).convert("RGB") if hits else None
                response = vlm.generate(image, rag_prompt(str(query)), temperature=0.0, max_new_tokens=args.max_new_tokens)
                clean_detail = clean_cache[flat_index]
                detail = {
                    "record_index": record_index,
                    "probe_index": probe_index,
                    "id": record["id"],
                    "source_semantitrace_id": record.get("source_semantitrace_id"),
                    "method": variant,
                    "signature": signature,
                    "query": str(query),
                    "watermarked_response": response,
                    "clean_response": clean_detail["clean_response"],
                    "watermarked_hit": contains_positive_signature(response, signature),
                    "clean_hit": contains_positive_signature(clean_detail["clean_response"], signature),
                    "watermarked_hits_retrieval": [hit.__dict__ for hit in hits],
                    "clean_hits_retrieval": clean_detail.get("clean_hits_retrieval", []),
                }
                out.write(json.dumps(detail, ensure_ascii=False) + "\n")
                out.flush()
                details.append(detail)
                print(
                    f"[{variant} verify {len(details):03d}/{len(records) * verifier.num_probes_per_canary:03d}] "
                    f"{record['id']} probe={probe_index} sig={signature} "
                    f"wm_hit={detail['watermarked_hit']} clean_hit={detail['clean_hit']}",
                    flush=True,
                )
    signatures = [str(r["trap_signature"]) for r in records]
    suspect_responses = [str(d["watermarked_response"]) for d in details]
    clean_responses = [str(d["clean_response"]) for d in details]
    suspect_samples = verifier.compute_per_canary_cer(suspect_responses, signatures)
    clean_samples = verifier.compute_per_canary_cer(clean_responses, signatures)
    test = verifier.welch_t_test(suspect_samples, clean_samples)
    qlds = [float(r["injection_metrics"].get("quality_local_delta", 0.0)) for r in records]
    psnrs = [
        float(r["injection_metrics"]["psnr"])
        for r in records
        if isinstance(r["injection_metrics"].get("psnr"), (int, float)) and math.isfinite(float(r["injection_metrics"]["psnr"]))
    ]
    report = {
        "method": variant,
        "method_label": VARIANTS[variant]["label"],
        "num_canaries": len(records),
        "num_probes_per_canary": verifier.num_probes_per_canary,
        "top_k": args.top_k,
        "suspect_cer": float(suspect_samples.mean()) if suspect_samples.size else 0.0,
        "suspect_per_canary_cer": suspect_samples.tolist(),
        "clean_cer": float(clean_samples.mean()) if clean_samples.size else 0.0,
        "clean_per_canary_cer": clean_samples.tolist(),
        "test_result": test,
        "avg_quality_delta": float(np.mean(qlds)) if qlds else None,
        "max_quality_delta": float(np.max(qlds)) if qlds else None,
        "avg_psnr": float(np.mean(psnrs)) if psnrs else None,
        "details": details,
    }
    (method_dir / "rag_verify_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def write_summary(out_dir: Path, source_records: list[dict[str, Any]], reports: list[dict[str, Any]], sem_report_path: str) -> list[dict[str, Any]]:
    sem_report = json.loads(resolve_path(sem_report_path).read_text(encoding="utf-8"))
    sem_qlds = [float((r.get("injection_metrics") or {}).get("quality_local_delta", 0.0)) for r in source_records]
    rows = [
        {
            "method": "full_semantitrace",
            "label": "Full SemantiTrace",
            "num_canaries": sem_report["num_canaries"],
            "suspect_cer": sem_report["suspect_cer"],
            "clean_cer": sem_report["clean_cer"],
            "p_value": sem_report["test_result"]["p_value"],
            "reject_h0": sem_report["test_result"]["reject_h0"],
            "avg_quality_delta": float(np.mean(sem_qlds)) if sem_qlds else None,
            "max_quality_delta": float(np.max(sem_qlds)) if sem_qlds else None,
            "avg_psnr": None,
        }
    ]
    for report in reports:
        rows.append(
            {
                "method": report["method"],
                "label": report["method_label"],
                "num_canaries": report["num_canaries"],
                "suspect_cer": report["suspect_cer"],
                "clean_cer": report["clean_cer"],
                "p_value": report["test_result"]["p_value"],
                "reject_h0": report["test_result"]["reject_h0"],
                "avg_quality_delta": report["avg_quality_delta"],
                "max_quality_delta": report["max_quality_delta"],
                "avg_psnr": report["avg_psnr"],
            }
        )
    (out_dir / "flux_ablation_summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows


def main() -> None:
    args = parse_args()
    out_dir = resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    source_records = load_source_records(args.source_records, args.max_records)
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for variant in variants:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant}; choose from {sorted(VARIANTS)}")

    variant_records: dict[str, list[dict[str, Any]]] = {}
    if not args.verify_only:
        editor = build_editor(config, args.device)
        for variant in variants:
            variant_records[variant] = materialize_variant(variant, source_records, editor, out_dir, config)
    else:
        for variant in variants:
            path = out_dir / variant / "canary_records.json"
            variant_records[variant] = json.loads(path.read_text(encoding="utf-8"))

    if args.generate_only:
        return

    clean_cache = load_clean_cache(args.semantitrace_report, len(source_records) * 3)
    encoder, vlm, verifier = build_clients(args.config, args.device)
    reports: list[dict[str, Any]] = []
    for variant in variants:
        reports.append(verify_variant(variant, variant_records[variant], clean_cache, encoder, vlm, verifier, out_dir, args))
    rows = write_summary(out_dir, source_records, reports, args.semantitrace_report)
    print(json.dumps(rows, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

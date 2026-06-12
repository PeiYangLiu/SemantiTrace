#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_final_rag_verify import build_clients, rag_prompt, resolve_path
from semantitrace.baselines import AQUABaseline, PGDBaseline
from semantitrace.metrics import compute_psnr, contains_positive_signature
from semantitrace.rag import ImageRAGIndex


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Compare N=100 SemantiTrace against simple baselines")
    parser.add_argument("--source_records", default="outputs/flux2_klein_n100_v1_rag_verify/canary_records.json")
    parser.add_argument("--semantitrace_report", default="outputs/flux2_klein_n100_v1_rag_verify/rag_verify_report.json")
    parser.add_argument("--output_dir", default="outputs/flux2_klein_n100_v1_baseline_compare")
    parser.add_argument("--config", default="configs/semantitrace_flux2_klein_opus_struct.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--methods", default="naive,naive_overlay,pgd,aqua_acronym,aqua_spatial")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--pgd_epsilon", type=float, default=8 / 255)
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--materialize_only", action="store_true")
    return parser.parse_args()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def load_source_records(path: str, max_records: int | None) -> list[dict[str, Any]]:
    records = json.loads(resolve_path(path).read_text(encoding="utf-8"))
    return records[:max_records] if max_records is not None else records


def quality_delta(clean: Image.Image, edited: Image.Image) -> float:
    a = np.asarray(clean.convert("RGB"), dtype=np.float32)
    b = np.asarray(edited.convert("RGB").resize(clean.size), dtype=np.float32)
    return float(np.mean(np.abs(a - b)) / 255.0)


def safe_psnr(clean: Image.Image, edited: Image.Image) -> float | str:
    value = compute_psnr(np.asarray(clean.convert("RGB")), np.asarray(edited.convert("RGB").resize(clean.size)))
    return "inf" if math.isinf(value) else float(value)


def font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def naive_overlay(image: Image.Image, signature: str) -> tuple[Image.Image, list[int]]:
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    w, h = out.size
    size = max(22, min(96, min(w, h) // 9))
    fnt = font(size)
    bbox = draw.textbbox((0, 0), signature, font=fnt)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = max(8, size // 6)
    x1 = max(0, w - tw - 3 * pad)
    y1 = max(0, h - th - 3 * pad)
    x2 = min(w, x1 + tw + 2 * pad)
    y2 = min(h, y1 + th + 2 * pad)
    draw.rectangle((x1, y1, x2, y2), fill=(255, 255, 255), outline=(0, 0, 0), width=max(1, size // 24))
    draw.text((x1 + pad, y1 + pad), signature, fill=(0, 0, 0), font=fnt)
    return out, [x1, y1, x2, y2]


def make_method_image(method: str, image: Image.Image, signature: str, idx: int, args: argparse.Namespace):
    if method == "naive":
        return image.copy(), "unmodified clean image assigned a canary signature", [0, 0, image.width, image.height]
    if method == "naive_overlay":
        edited, bbox = naive_overlay(image, signature)
        return edited, f"obvious text overlay reading {signature}", bbox
    if method == "pgd":
        baseline = PGDBaseline(epsilon=args.pgd_epsilon, seed=42 + idx).apply(image, signature)
        return baseline.image, baseline.trigger_prompt, [0, 0, image.width, image.height]
    aqua = AQUABaseline(seed=42 + idx)
    if method == "aqua_acronym":
        baseline = aqua.acronym(signature)
        return baseline.image, baseline.trigger_prompt, [0, 0, baseline.image.width, baseline.image.height]
    if method == "aqua_spatial":
        baseline = aqua.spatial(signature)
        return baseline.image, baseline.trigger_prompt, [0, 0, baseline.image.width, baseline.image.height]
    raise ValueError(f"Unknown baseline method: {method}")


def materialize_method(method: str, source_records: list[dict[str, Any]], out_dir: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    method_dir = out_dir / method
    wm_dir = method_dir / "watermarked"
    clean_dir = method_dir / "clean"
    wm_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for idx, source in enumerate(source_records):
        signature = str(source["trap_signature"])
        anchor_src = resolve_path(source["anchor_image_path"])
        clean = Image.open(anchor_src).convert("RGB")
        edited, trigger, bbox = make_method_image(method, clean, signature, idx, args)
        clean_path = clean_dir / f"{method}-{idx:04d}_{anchor_src.name}"
        wm_path = wm_dir / f"{method}-{idx:04d}_{signature}.png"
        if not clean_path.exists():
            shutil.copy2(anchor_src, clean_path)
        edited.save(wm_path)
        selected_canvas = {
            "id": 0,
            "mode": "baseline",
            "bbox": bbox,
            "score": 1.0,
            "text": None,
            "source": method,
        }
        q_delta = quality_delta(clean, edited)
        record = {
            "id": f"{method}-{idx:04d}",
            "method": method,
            "source_semantitrace_id": source.get("id"),
            "source_run": f"baseline:{method}",
            "anchor_image_path": rel(clean_path),
            "source_original_anchor_image_path": source["anchor_image_path"],
            "watermarked_image_path": rel(wm_path),
            "selected_box_id": 0,
            "selected_canvas": selected_canvas,
            "parasitism_mode": "Baseline",
            "trigger_prompt": trigger,
            "trap_signature": signature,
            "probe_queries": list(source.get("probe_queries", [])),
            "reasoning": f"N=100 baseline method {method}",
            "injection_metrics": {
                "render_strategy": method,
                "masked_blending_enforced": False,
                "effective_mask_bbox": bbox,
                "effective_mask_area_ratio": 1.0 if method in {"naive", "pgd"} else None,
                "quality_local_delta": q_delta,
                "psnr": safe_psnr(clean, edited),
            },
        }
        records.append(record)
    (method_dir / "canary_records.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    return records


def load_clean_cache(report_path: str, expected_count: int) -> list[dict[str, Any]]:
    report = json.loads(resolve_path(report_path).read_text(encoding="utf-8"))
    details = report.get("details", [])
    if len(details) < expected_count:
        raise ValueError(f"Clean cache has {len(details)} details, expected at least {expected_count}")
    return details[:expected_count]


def verify_method(
    method: str,
    records: list[dict[str, Any]],
    clean_cache: list[dict[str, Any]],
    encoder,
    vlm,
    verifier,
    method_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
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
                response = vlm.generate(
                    image,
                    rag_prompt(str(query)),
                    temperature=0.0,
                    max_new_tokens=args.max_new_tokens,
                )
                clean_detail = clean_cache[flat_index]
                detail = {
                    "record_index": record_index,
                    "probe_index": probe_index,
                    "id": record["id"],
                    "source_semantitrace_id": record.get("source_semantitrace_id"),
                    "method": method,
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
                    f"[{method} {len(details):03d}/{len(records) * verifier.num_probes_per_canary:03d}] "
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
    qlds = [float(r["injection_metrics"]["quality_local_delta"]) for r in records]
    psnrs = [
        float(r["injection_metrics"]["psnr"])
        for r in records
        if isinstance(r["injection_metrics"]["psnr"], (int, float)) and math.isfinite(float(r["injection_metrics"]["psnr"]))
    ]
    report = {
        "method": method,
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
        "avg_psnr": float(np.mean(psnrs)) if psnrs else "inf",
        "details": details,
    }
    (method_dir / "rag_verify_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    out_dir = resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    source_records = load_source_records(args.source_records, args.max_records)
    clean_cache = load_clean_cache(args.semantitrace_report, len(source_records) * 3)

    method_records = {}
    for method in methods:
        method_records[method] = materialize_method(method, source_records, out_dir, args)
        print(f"materialized {method}: {len(method_records[method])}", flush=True)
    if args.materialize_only:
        return

    encoder, vlm, verifier = build_clients(args.config, args.device)
    reports: list[dict[str, Any]] = []
    for method in methods:
        reports.append(verify_method(method, method_records[method], clean_cache, encoder, vlm, verifier, out_dir / method, args))

    sem_report = json.loads(resolve_path(args.semantitrace_report).read_text(encoding="utf-8"))
    sem_records = json.loads(resolve_path(args.source_records).read_text(encoding="utf-8"))
    sem_qlds = [float((r.get("injection_metrics") or {}).get("quality_local_delta", 0.0)) for r in sem_records]
    summary = [
        {
            "method": "semantitrace",
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
        summary.append(
            {
                "method": report["method"],
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
    (out_dir / "baseline_comparison_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

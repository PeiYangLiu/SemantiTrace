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
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_final_rag_verify import build_clients, rag_prompt, resolve_path
from semantitrace.metrics import contains_positive_signature, compute_psnr
from semantitrace.rag import ImageRAGIndex
from semantitrace.utils.image import bbox_from_mask, mask_from_bbox


ROOT = Path(__file__).resolve().parents[1]


DEFENSES = {
    "none": "None",
    "ocr_blur": "OCR blur",
    "ocr_fill": "OCR fill",
    "oracle_canvas_blur": "Oracle canvas blur",
    "jpeg_resize_q50": "JPEG/resize Q50",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Run adaptive ingestion defense stress tests on SemantiTrace canaries")
    parser.add_argument("--source_records", default="outputs/flux2_klein_gradient_guided_n100_tuned_v1_rag_verify/canary_records.json")
    parser.add_argument("--semantitrace_report", default="outputs/flux2_klein_gradient_guided_n100_tuned_v1_rag_verify/rag_verify_report.json")
    parser.add_argument("--output_dir", default="outputs/adaptive_defense_stress_flux_n100")
    parser.add_argument("--config", default="configs/semantitrace_flux2_klein_gradient_guided.yaml")
    parser.add_argument("--defenses", default="ocr_blur,ocr_fill,oracle_canvas_blur,jpeg_resize_q50")
    parser.add_argument("--max_records", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--fresh_clean", action="store_true", help="Query clean images directly instead of reusing semantitrace_report clean cache")
    parser.add_argument("--materialize_only", action="store_true")
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


def bbox_for_record(record: dict[str, Any], image: Image.Image) -> list[int]:
    metrics = record.get("injection_metrics") if isinstance(record.get("injection_metrics"), dict) else {}
    bbox = metrics.get("effective_mask_bbox")
    if bbox and len(bbox) == 4:
        return [int(v) for v in bbox]
    canvas = record.get("selected_canvas") if isinstance(record.get("selected_canvas"), dict) else {}
    bbox = canvas.get("bbox") if canvas else None
    if bbox and len(bbox) == 4:
        return [int(v) for v in bbox]
    return [0, 0, image.width, image.height]


def quality_delta(clean: Image.Image, edited: Image.Image) -> float:
    a = np.asarray(clean.convert("RGB"), dtype=np.float32)
    b = np.asarray(edited.convert("RGB").resize(clean.size), dtype=np.float32)
    return float(np.mean(np.abs(a - b)) / 255.0)


def safe_psnr(clean: Image.Image, edited: Image.Image) -> float | str:
    value = compute_psnr(np.asarray(clean.convert("RGB")), np.asarray(edited.convert("RGB").resize(clean.size)))
    return "inf" if math.isinf(value) else float(value)


def load_ocr_cache(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_ocr_cache(path: Path, cache: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")


def ocr_boxes(image_path: Path, cache_path: Path, reader: Any) -> list[list[int]]:
    cache = load_ocr_cache(cache_path)
    key = str(image_path.resolve())
    if key not in cache:
        result = reader.readtext(str(image_path), detail=1, paragraph=False)
        boxes = []
        for item in result:
            if len(item) < 2:
                continue
            points = item[0]
            text = str(item[1])
            conf = float(item[2]) if len(item) > 2 else 1.0
            xs = [int(p[0]) for p in points]
            ys = [int(p[1]) for p in points]
            boxes.append({"bbox": [min(xs), min(ys), max(xs), max(ys)], "text": text, "conf": conf})
        cache[key] = boxes
        save_ocr_cache(cache_path, cache)
    return [list(map(int, item["bbox"])) for item in cache[key]]


def expand_bbox(bbox: list[int], image: Image.Image, pad: int = 3) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return max(0, x1 - pad), max(0, y1 - pad), min(image.width, x2 + pad), min(image.height, y2 + pad)


def fill_region_with_median(image: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image:
    out = image.convert("RGB").copy()
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return out
    arr = np.asarray(out)
    region = arr[y1:y2, x1:x2]
    if region.size == 0:
        return out
    color = tuple(int(v) for v in np.median(region.reshape(-1, 3), axis=0))
    draw = ImageDraw.Draw(out)
    draw.rectangle((x1, y1, x2, y2), fill=color)
    return out


def apply_ocr_defense(image: Image.Image, boxes: list[list[int]], mode: str) -> tuple[Image.Image, dict[str, Any]]:
    out = image.convert("RGB").copy()
    total_area = 0
    for raw_box in boxes:
        box = expand_bbox(raw_box, out, pad=4)
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            continue
        total_area += (x2 - x1) * (y2 - y1)
        if mode == "ocr_blur":
            crop = out.crop(box).filter(ImageFilter.GaussianBlur(radius=8))
            out.paste(crop, box)
        elif mode == "ocr_fill":
            out = fill_region_with_median(out, box)
    return out, {
        "ocr_boxes": len(boxes),
        "defended_area_ratio": total_area / max(1, image.width * image.height),
    }


def apply_canvas_blur(image: Image.Image, bbox: list[int]) -> tuple[Image.Image, dict[str, Any]]:
    out = image.convert("RGB").copy()
    box = expand_bbox(bbox, out, pad=6)
    x1, y1, x2, y2 = box
    crop = out.crop(box).filter(ImageFilter.GaussianBlur(radius=12))
    out.paste(crop, box)
    return out, {"ocr_boxes": None, "defended_area_ratio": ((x2 - x1) * (y2 - y1)) / max(1, image.width * image.height)}


def apply_jpeg_resize(image: Image.Image) -> tuple[Image.Image, dict[str, Any]]:
    from io import BytesIO

    original = image.convert("RGB")
    small = original.resize((max(1, original.width // 2), max(1, original.height // 2)), Image.Resampling.LANCZOS)
    restored = small.resize(original.size, Image.Resampling.LANCZOS)
    buf = BytesIO()
    restored.save(buf, format="JPEG", quality=50)
    buf.seek(0)
    out = Image.open(buf).convert("RGB")
    return out, {"ocr_boxes": None, "defended_area_ratio": 1.0}


def materialize_defense(
    defense: str,
    source_records: list[dict[str, Any]],
    out_dir: Path,
    reader: Any | None,
) -> list[dict[str, Any]]:
    method_dir = out_dir / defense
    wm_dir = method_dir / "watermarked"
    clean_dir = method_dir / "clean"
    wm_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    records_path = method_dir / "canary_records.json"
    if records_path.exists():
        return json.loads(records_path.read_text(encoding="utf-8"))
    ocr_cache = out_dir / "_ocr_boxes_cache.json"
    records: list[dict[str, Any]] = []
    for idx, source in enumerate(source_records):
        wm_src = resolve_path(source["watermarked_image_path"])
        clean_src = resolve_path(source["anchor_image_path"])
        image = Image.open(wm_src).convert("RGB")
        clean = Image.open(clean_src).convert("RGB")
        if defense == "none":
            defended = image
            stats = {"ocr_boxes": None, "defended_area_ratio": 0.0}
        elif defense in {"ocr_blur", "ocr_fill"}:
            assert reader is not None
            boxes = ocr_boxes(wm_src, ocr_cache, reader)
            defended, stats = apply_ocr_defense(image, boxes, defense)
        elif defense == "oracle_canvas_blur":
            defended, stats = apply_canvas_blur(image, bbox_for_record(source, image))
        elif defense == "jpeg_resize_q50":
            defended, stats = apply_jpeg_resize(image)
        else:
            raise ValueError(f"Unknown defense: {defense}")
        clean_path = clean_dir / f"{defense}-{idx:04d}_{clean_src.name}"
        wm_path = wm_dir / f"{defense}-{idx:04d}_{Path(wm_src).name}"
        if not clean_path.exists():
            shutil.copy2(clean_src, clean_path)
        defended.save(wm_path)
        rec = json.loads(json.dumps(source))
        rec["id"] = f"{defense}-{idx:04d}"
        rec["source_semantitrace_id"] = source.get("id")
        rec["defense"] = defense
        rec["defense_label"] = DEFENSES[defense]
        rec["anchor_image_path"] = rel(clean_path)
        rec["watermarked_image_path"] = rel(wm_path)
        rec.setdefault("injection_metrics", {})
        rec["defense_metrics"] = {
            **stats,
            "quality_delta_vs_watermarked": quality_delta(image, defended),
            "quality_delta_vs_clean": quality_delta(clean, defended),
            "psnr_vs_clean": safe_psnr(clean, defended),
        }
        records.append(rec)
        records_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[{defense}] materialized {idx + 1}/{len(source_records)}", flush=True)
    return records


def load_clean_cache(report_path: str, expected_count: int) -> list[dict[str, Any]]:
    report = json.loads(resolve_path(report_path).read_text(encoding="utf-8"))
    details = report.get("details", [])
    if len(details) < expected_count:
        raise ValueError(f"Clean cache has {len(details)} details, expected at least {expected_count}")
    return details[:expected_count]


def verify_defense(
    defense: str,
    records: list[dict[str, Any]],
    clean_cache: list[dict[str, Any]] | None,
    encoder: Any,
    vlm: Any,
    verifier: Any,
    out_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    method_dir = out_dir / defense
    details_path = method_dir / "rag_verify_details.jsonl"
    details: list[dict[str, Any]] = []
    if details_path.exists():
        with details_path.open("r", encoding="utf-8") as fh:
            details = [json.loads(line) for line in fh if line.strip()]
    index = ImageRAGIndex(encoder).build([resolve_path(r["watermarked_image_path"]) for r in records], [r["id"] for r in records])
    skip = len(details)
    ranks: list[int] = [int(hit["rank"]) for d in details for hit in d.get("target_retrieval", [])]
    with details_path.open("a", encoding="utf-8") as out:
        for record_index, record in enumerate(records):
            signature = str(record["trap_signature"])
            queries = list(record.get("probe_queries", []))[: verifier.num_probes_per_canary]
            for probe_index, query in enumerate(queries):
                flat_index = record_index * verifier.num_probes_per_canary + probe_index
                if flat_index < skip:
                    continue
                hits = index.search(str(query), args.top_k)
                target_rank = next((hit.rank for hit in hits if hit.image_id == record["id"]), args.top_k + 1)
                ranks.append(target_rank)
                image = Image.open(hits[0].image_path).convert("RGB") if hits else None
                response = vlm.generate(image, rag_prompt(str(query)), temperature=0.0, max_new_tokens=args.max_new_tokens)
                if clean_cache is not None:
                    clean_response = clean_cache[flat_index]["clean_response"]
                else:
                    clean_image = Image.open(resolve_path(record["anchor_image_path"])).convert("RGB")
                    clean_response = vlm.generate(clean_image, rag_prompt(str(query)), temperature=0.0, max_new_tokens=args.max_new_tokens)
                detail = {
                    "record_index": record_index,
                    "probe_index": probe_index,
                    "id": record["id"],
                    "source_semantitrace_id": record.get("source_semantitrace_id"),
                    "defense": defense,
                    "signature": signature,
                    "query": str(query),
                    "watermarked_response": response,
                    "clean_response": clean_response,
                    "watermarked_hit": contains_positive_signature(response, signature),
                    "clean_hit": contains_positive_signature(clean_response, signature),
                    "target_rank": target_rank,
                    "watermarked_hits_retrieval": [hit.__dict__ for hit in hits],
                }
                out.write(json.dumps(detail, ensure_ascii=False) + "\n")
                out.flush()
                details.append(detail)
                print(
                    f"[{defense} verify {len(details):03d}/{len(records) * verifier.num_probes_per_canary:03d}] "
                    f"{record['id']} rank={target_rank} wm_hit={detail['watermarked_hit']}",
                    flush=True,
                )
    signatures = [str(r["trap_signature"]) for r in records]
    suspect_responses = [str(d["watermarked_response"]) for d in details]
    clean_responses = [str(d["clean_response"]) for d in details]
    suspect_samples = verifier.compute_per_canary_cer(suspect_responses, signatures)
    clean_samples = verifier.compute_per_canary_cer(clean_responses, signatures)
    test = verifier.welch_t_test(suspect_samples, clean_samples)
    qlds = [float((r.get("defense_metrics") or {}).get("quality_delta_vs_clean", 0.0)) for r in records]
    defended_area = [float((r.get("defense_metrics") or {}).get("defended_area_ratio", 0.0) or 0.0) for r in records]
    ocr_boxes = [r.get("defense_metrics", {}).get("ocr_boxes") for r in records if r.get("defense_metrics", {}).get("ocr_boxes") is not None]
    report = {
        "defense": defense,
        "defense_label": DEFENSES[defense],
        "num_canaries": len(records),
        "num_probes_per_canary": verifier.num_probes_per_canary,
        "top_k": args.top_k,
        "suspect_cer": float(suspect_samples.mean()) if suspect_samples.size else 0.0,
        "clean_cer": float(clean_samples.mean()) if clean_samples.size else 0.0,
        "test_result": test,
        "recall_at_3": float(np.mean(np.asarray(ranks) <= 3)) if ranks else 0.0,
        "avg_quality_delta_vs_clean": float(np.mean(qlds)) if qlds else 0.0,
        "avg_defended_area_ratio": float(np.mean(defended_area)) if defended_area else 0.0,
        "avg_ocr_boxes": float(np.mean(ocr_boxes)) if ocr_boxes else None,
        "details": details,
    }
    (method_dir / "rag_verify_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def write_summary(out_dir: Path, source_report_path: str, reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    has_none = any(report["defense"] == "none" for report in reports)
    source_path = resolve_path(source_report_path)
    if not has_none and source_path.exists():
        source = json.loads(source_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "defense": "none",
                "defense_label": "None",
                "suspect_cer": source["suspect_cer"],
                "clean_cer": source["clean_cer"],
                "p_value": source["test_result"]["p_value"],
                "recall_at_3": 1.0,
                "avg_quality_delta_vs_clean": None,
                "avg_defended_area_ratio": 0.0,
                "avg_ocr_boxes": None,
            }
        )
    for report in reports:
        rows.append(
            {
                "defense": report["defense"],
                "defense_label": report["defense_label"],
                "suspect_cer": report["suspect_cer"],
                "clean_cer": report["clean_cer"],
                "p_value": report["test_result"]["p_value"],
                "recall_at_3": report["recall_at_3"],
                "avg_quality_delta_vs_clean": report["avg_quality_delta_vs_clean"],
                "avg_defended_area_ratio": report["avg_defended_area_ratio"],
                "avg_ocr_boxes": report["avg_ocr_boxes"],
            }
        )
    (out_dir / "adaptive_defense_summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows


def main() -> None:
    args = parse_args()
    out_dir = resolve_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    defenses = [d.strip() for d in args.defenses.split(",") if d.strip()]
    for defense in defenses:
        if defense not in DEFENSES:
            raise ValueError(f"Unknown defense {defense}; choose from {sorted(DEFENSES)}")
    source_records = load_source_records(args.source_records, args.max_records)
    reader = None
    if not args.verify_only and any(d in {"ocr_blur", "ocr_fill"} for d in defenses):
        import easyocr
        import torch

        reader = easyocr.Reader(["en"], gpu=(args.device != "cpu" and torch.cuda.is_available()))

    defense_records: dict[str, list[dict[str, Any]]] = {}
    if not args.verify_only:
        for defense in defenses:
            defense_records[defense] = materialize_defense(defense, source_records, out_dir, reader)
    else:
        for defense in defenses:
            defense_records[defense] = json.loads((out_dir / defense / "canary_records.json").read_text(encoding="utf-8"))

    if args.materialize_only:
        return

    clean_cache = None if args.fresh_clean else load_clean_cache(args.semantitrace_report, len(source_records) * 3)
    encoder, vlm, verifier = build_clients(args.config, args.device)
    reports = []
    for defense in defenses:
        reports.append(verify_defense(defense, defense_records[defense], clean_cache, encoder, vlm, verifier, out_dir, args))
    rows = write_summary(out_dir, args.semantitrace_report, reports)
    print(json.dumps(rows, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantitrace.anchor_mining import Anchor, AnchorMiner, Canvas
from semantitrace.backends.deterministic import DeterministicEncoder, GridMaskGenerator, HeuristicVLMClient, SimpleOCRDetector
from semantitrace.backends.real import EasyOCRDetector, OpenAICompatibleVLM, OpenCLIPEncoder, QwenVLMClient, SAMMaskGenerator
from semantitrace.config import load_config
from semantitrace.pipeline import SemantiTracePipeline
from semantitrace.utils.image import list_images, mask_from_bbox


COMMON_BAD_WORDS = {
    "LOVE",
    "HOPE",
    "HOME",
    "AWAY",
    "WHITE",
    "BLACK",
    "NIGHT",
    "THAT",
    "THERE",
    "ARTS",
    "PHOTO",
    "LIFE",
    "KING",
    "DARK",
    "MALE",
    "CLOSE",
    "TOWN",
    "WEST",
    "FINE",
    "BOSS",
    "SLEEP",
    "TOWER",
    "BARNS",
    "XBOX",
    "DISNEY",
    "REEBOK",
}

CATEGORY_VETO_TERMS = {
    "album",
    "book",
    "cast",
    "cover",
    "credit",
    "credits",
    "famous",
    "movie",
    "poster",
    "title",
    "trademark",
    "trademarked",
    "tv",
}

REASON_VETO_PATTERNS = (
    r"\b(part of|appears on|from|within|shown on)\s+(a|an|the)?\s*[^.]{0,40}\b(movie|tv|book|album|poster|cover|cast|credits?)\b",
    r"\b(movie|tv|book|album|poster|cover|cast|credits?)\s+(title|typography|design|art|credit|credits)\b",
    r"\b(famous|iconic|trademarked?|identity-critical)\b",
    r"\b(fragment|partial)\s+(word|text|ocr|view)\b",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Pre-scan and rank likely FLUX.2 scene-text candidates")
    parser.add_argument("--dataset_dir", required=True, help="Directory containing candidate source images")
    parser.add_argument("--output_dir", required=True, help="Directory for candidate records, review images, and selected images")
    parser.add_argument("--config", default="configs/semantitrace_flux2_klein_real_safe.yaml", help="SemantiTrace YAML config")
    parser.add_argument("--device", default="cuda", help="cpu or cuda")
    parser.add_argument("--num_targets", type=int, default=30, help="How many top candidate images to export")
    parser.add_argument("--max_review_anchors", type=int, default=0, help="0 means review all mined anchors")
    parser.add_argument("--exhaustive_ocr", action="store_true", help="Scan OCR canvases in every image instead of cluster anchors")
    parser.add_argument("--max_vlm_reviews", type=int, default=200, help="Max top OCR canvases to review with the VLM in exhaustive mode; 0 means all")
    parser.add_argument("--score_threshold", type=float, default=65.0, help="Minimum VLM crop score to export")
    parser.add_argument("--no_vlm_review", action="store_true", help="Use lexical/anchor heuristics only")
    parser.add_argument("--copy_images", action="store_true", help="Copy selected images instead of symlinking")
    parser.add_argument("--ocr_confidence_threshold", type=float, default=None, help="Override anchor_mining.ocr_confidence_threshold")
    parser.add_argument("--min_canvas_area_ratio", type=float, default=None, help="Override anchor_mining.min_canvas_area_ratio")
    parser.add_argument("--max_canvas_area_ratio", type=float, default=None, help="Override anchor_mining.max_canvas_area_ratio")
    parser.add_argument("--max_text_alnum", type=int, default=None, help="Override anchor_mining.max_text_alnum")
    parser.add_argument("--max_text_words", type=int, default=None, help="Override anchor_mining.max_text_words")
    parser.add_argument("--max_short_text_bbox_area_ratio", type=float, default=None, help="Override anchor_mining.max_short_text_bbox_area_ratio")
    parser.add_argument("--max_text_bbox_area_per_alnum_ratio", type=float, default=None, help="Override anchor_mining.max_text_bbox_area_per_alnum_ratio")
    return parser.parse_args()


def build_encoder(config: dict[str, Any], device: str):
    backend = config["backends"].get("encoder", "deterministic")
    if backend == "deterministic":
        return DeterministicEncoder()
    if backend == "open_clip":
        models = config.get("models", {})
        return OpenCLIPEncoder(
            model_name=models.get("clip_model", "ViT-L-14"),
            pretrained=models.get("clip_pretrained", "openai"),
            device=device,
        )
    raise ValueError(f"Unknown encoder backend: {backend}")


def build_ocr(config: dict[str, Any], device: str):
    backend = config["backends"].get("ocr", "simple")
    if backend == "simple":
        return SimpleOCRDetector()
    if backend == "easyocr":
        return EasyOCRDetector(gpu=device != "cpu")
    raise ValueError(f"Unknown OCR backend: {backend}")


def build_mask_generator(config: dict[str, Any], device: str):
    backend = config["backends"].get("mask_generator", "grid")
    if backend == "grid":
        cfg = config.get("mask_generator", {})
        return GridMaskGenerator(
            min_size=int(cfg.get("min_size", 24)),
            rows=int(cfg.get("rows", 5)),
            cols=int(cfg.get("cols", 5)),
            cell_fraction=float(cfg.get("cell_fraction", 0.72)),
        )
    if backend == "sam":
        checkpoint = config.get("models", {}).get("sam_checkpoint")
        if not checkpoint:
            raise ValueError("models.sam_checkpoint is required when mask_generator=sam")
        return SAMMaskGenerator(checkpoint=checkpoint, device=device)
    raise ValueError(f"Unknown mask_generator backend: {backend}")


def build_vlm(config: dict[str, Any], device: str):
    backend = config["backends"].get("vlm", "heuristic")
    if backend == "heuristic":
        return HeuristicVLMClient()
    if backend == "openai":
        import os

        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GITHUB_TOKEN") or "dummy"
        base_url = os.environ.get("SEMANTITRACE_VLM_BASE_URL")
        if not base_url:
            raise ValueError("SEMANTITRACE_VLM_BASE_URL must be set when vlm=openai")
        model = config.get("models", {}).get("surrogate_vlm", "Qwen3-VL-8B-Instruct")
        return OpenAICompatibleVLM(base_url=base_url, api_key=api_key, model=model)
    if backend == "qwen_vl":
        models = config.get("models", {})
        vlm_cfg = config.get("vlm", {})
        return QwenVLMClient(
            model_name=models.get("surrogate_vlm", "Qwen/Qwen3-VL-8B-Instruct"),
            device=device,
            torch_dtype=vlm_cfg.get("torch_dtype", "bfloat16"),
        )
    raise ValueError(f"Unknown VLM backend: {backend}")


def lexical_prior(text: str) -> float:
    normalized = re.sub(r"[^A-Za-z0-9 ]", "", text).strip()
    alnum = re.sub(r"[^A-Za-z0-9]", "", normalized).upper()
    words = re.findall(r"[A-Za-z0-9]+", normalized)
    score = 55.0
    if 3 <= len(alnum) <= 5:
        score += 15.0
    if len(words) == 1:
        score += 8.0
    else:
        score -= 30.0
    if alnum in COMMON_BAD_WORDS:
        score -= 25.0
    if any(char.islower() for char in normalized) and any(char.isupper() for char in normalized):
        score += 5.0
    if re.search(r"\d", alnum):
        score -= 10.0
    if re.search(r"[^AEIOU]{4,}", alnum):
        score -= 8.0
    return max(0.0, min(100.0, score))


def review_prompt(canvas: Canvas) -> str:
    return (
        "You are ranking candidate scene-text regions for a paper-faithful SemantiTrace FLUX.2 inpainting run.\n"
        f"The crop contains OCR text {canvas.text!r}. Return JSON only with keys pass, score, category, reason.\n\n"
        "Set pass=true only for visually promising native scene-text carriers: flat local/place emblems, badges, simple product/package labels, screen/menu buttons, generic labels, or small flat signs where a same-length 3-5 letter rare code could still look natively printed after inpainting.\n"
        "Set pass=false for movie/TV/book/album/poster title typography, cast names, decorative cover art words, famous trademarks, text on people/clothing/wearables, broken or partial OCR, tiny text, dense credits/paragraphs, words inside sentences, strong perspective, curved text, 3D/depth lettering, embossed/engraved/weathered material, and building facade/outdoor signs with physical depth.\n"
        "Use score 90-100 for excellent flat badge/logo/package/screen candidates, 70-89 for plausible simple labels/signs, 50-69 for risky but possible candidates, and below 50 for likely failures. Be strict: this ranking is meant to avoid wasting expensive FLUX calls."
    )


def parse_review(raw: str) -> dict[str, Any]:
    json_text = None
    if "```json" in raw:
        start = raw.find("```json") + len("```json")
        end = raw.find("```", start)
        if end != -1:
            json_text = raw[start:end].strip()
    if json_text is None and "{" in raw and "}" in raw:
        json_text = raw[raw.find("{") : raw.rfind("}") + 1]
    data: dict[str, Any] = {}
    if json_text:
        try:
            parsed = json.loads(json_text)
            if isinstance(parsed, dict):
                data = parsed
        except json.JSONDecodeError:
            data = {}
    raw_pass = data.get("pass", False)
    if isinstance(raw_pass, str):
        passed = raw_pass.strip().lower() in {"true", "yes", "pass"}
    else:
        passed = bool(raw_pass)
    try:
        score = float(data.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    return {
        "pass": passed,
        "score": max(0.0, min(100.0, score)),
        "category": str(data.get("category", "")),
        "reason": str(data.get("reason", raw if not data else "")),
        "raw_response": raw,
    }


def apply_hard_veto(text: str | None, review: dict[str, Any]) -> dict[str, Any]:
    normalized = re.sub(r"[^A-Za-z0-9]", "", text or "").upper()
    category = str(review.get("category", "")).lower()
    reason = str(review.get("reason", "")).lower()
    veto_reasons: list[str] = []
    if normalized in COMMON_BAD_WORDS:
        veto_reasons.append(f"common_or_high_failure_word:{normalized}")
    for term in CATEGORY_VETO_TERMS:
        if term in category:
            veto_reasons.append(f"veto_category:{term}")
    for pattern in REASON_VETO_PATTERNS:
        match = re.search(pattern, reason)
        if match and "not " not in reason[max(0, match.start() - 8) : match.start()]:
            veto_reasons.append(f"veto_reason:{pattern}")
    if veto_reasons:
        review = dict(review)
        review["pass"] = False
        review["hard_veto_reasons"] = veto_reasons
        review["score"] = min(float(review.get("score", 0.0)), 49.0)
        review["reason"] = f"hard_veto={veto_reasons}; {review.get('reason', '')}"
    return review


def review_canvas(vlm, image: Image.Image, canvas: Canvas, use_vlm: bool) -> dict[str, Any]:
    prior = lexical_prior(canvas.text or "")
    if not use_vlm:
        return apply_hard_veto(canvas.text, {
            "pass": prior >= 65.0,
            "score": prior,
            "category": "lexical_prior",
            "reason": "heuristic lexical prior only",
            "raw_response": "",
            "lexical_prior": prior,
        })
    crop = SemantiTracePipeline._crop_region(image, tuple(int(v) for v in canvas.bbox), 0.75, 224)
    raw = vlm.generate(crop, review_prompt(canvas), temperature=0.0, max_new_tokens=256)
    review = parse_review(raw)
    review["lexical_prior"] = prior
    review["score"] = round((float(review["score"]) * 0.8) + (prior * 0.2), 3)
    review["pass"] = bool(review["pass"] and review["score"] >= 50.0)
    return apply_hard_veto(canvas.text, review)


def candidate_record(anchor: Anchor, canvas: Canvas, review: dict[str, Any], rank_score: float) -> dict[str, Any]:
    return {
        "image_path": anchor.image_path,
        "cluster_id": anchor.cluster_id,
        "isolation_score": anchor.isolation_score,
        "anchor_joint_score": anchor.joint_score,
        "canvas": canvas.to_json(),
        "text": canvas.text,
        "review": review,
        "rank_score": rank_score,
    }


def heuristic_candidate_record(anchor: Anchor, canvas: Canvas, miner: AnchorMiner) -> dict[str, Any]:
    review = apply_hard_veto(
        canvas.text,
        {
            "pass": True,
            "score": lexical_prior(canvas.text or ""),
            "category": "heuristic_prefilter",
            "reason": "pre-VLM heuristic candidate score",
            "raw_response": "",
            "lexical_prior": lexical_prior(canvas.text or ""),
        },
    )
    text_joint = miner._joint_text_score(float(anchor.isolation_score), canvas)
    rank_score = float(review["score"]) + min(20.0, max(0.0, text_joint / 1000.0))
    return candidate_record(anchor, canvas, review, rank_score)


def export_review_image(record: dict[str, Any], out_path: Path) -> None:
    image = Image.open(record["image_path"]).convert("RGB")
    canvas = record["canvas"]
    bbox = tuple(int(v) for v in canvas["bbox"])
    crop = SemantiTracePipeline._crop_region(image, bbox, 0.75, 240)
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    draw.rectangle(bbox, outline=(220, 40, 0), width=4)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
        small = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
        small = font
    header = 96
    width = max(annotated.width + crop.width, 720)
    height = max(annotated.height, crop.height) + header
    review = Image.new("RGB", (width, height), "white")
    review.paste(annotated, (0, header))
    review.paste(crop, (annotated.width, header))
    draw = ImageDraw.Draw(review)
    info = record["review"]
    draw.text(
        (8, 8),
        f"score={record['rank_score']:.1f} text={record['text']!r} category={info.get('category', '')}",
        fill=(0, 0, 0),
        font=font,
    )
    draw.text((8, 36), f"reason: {str(info.get('reason', ''))[:180]}", fill=(40, 40, 40), font=small)
    draw.text((8, 66), f"image: {Path(record['image_path']).name}", fill=(40, 40, 40), font=small)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    review.save(out_path)


def safe_name(path: str, rank: int, text: str) -> str:
    stem = Path(path).stem
    suffix = Path(path).suffix.lower() or ".png"
    safe_text = re.sub(r"[^A-Za-z0-9_-]+", "_", text.strip())[:24] or "text"
    return f"{rank:03d}_{stem}_{safe_text}{suffix}"


def mine_exhaustive_ocr(image_paths: list[Path], miner: AnchorMiner) -> list[Anchor]:
    anchors: list[Anchor] = []
    for idx, image_path in enumerate(image_paths):
        if (idx + 1) % 100 == 0:
            print(f"[exhaustive-ocr] scanned {idx + 1}/{len(image_paths)}", flush=True)
        image = Image.open(image_path).convert("RGB")
        text_canvases, struct_canvases = miner.compute_editability(image)
        if not text_canvases:
            continue
        joint = max(miner._joint_text_score(1.0, canvas) for canvas in text_canvases)
        anchors.append(
            Anchor(
                image_path=str(image_path),
                cluster_id=idx,
                isolation_score=1.0,
                canvas_mode="text",
                candidate_canvases=text_canvases,
                joint_score=joint,
            )
        )
    return anchors


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    anchor_cfg = dict(config.get("anchor_mining", {}))
    for arg_name in (
        "ocr_confidence_threshold",
        "min_canvas_area_ratio",
        "max_canvas_area_ratio",
        "max_text_alnum",
        "max_text_words",
        "max_short_text_bbox_area_ratio",
        "max_text_bbox_area_per_alnum_ratio",
    ):
        value = getattr(args, arg_name)
        if value is not None:
            anchor_cfg[arg_name] = value
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(args.dataset_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found under {args.dataset_dir}")

    ocr = build_ocr(config, args.device)
    mask_generator = build_mask_generator(config, args.device)
    vlm = None if args.no_vlm_review else build_vlm(config, args.device)
    encoder = DeterministicEncoder() if args.exhaustive_ocr else build_encoder(config, args.device)
    miner = AnchorMiner(
        encoder,
        ocr,
        mask_generator,
        anchor_cfg,
        seed=int(config.get("defaults", {}).get("seed", 42)),
    )

    if args.exhaustive_ocr:
        anchors = mine_exhaustive_ocr(image_paths, miner)
    else:
        requested_clusters = min(int(anchor_cfg.get("num_clusters", 100)), len(image_paths))
        labels, features = miner.cluster_dataset(image_paths, requested_clusters)
        anchors = miner.mine_anchors(image_paths, requested_clusters, features, labels)
    if args.max_review_anchors > 0:
        anchors = anchors[: args.max_review_anchors]

    heuristic_records: list[dict[str, Any]] = []
    for anchor in anchors:
        for canvas in anchor.candidate_canvases[: min(len(anchor.candidate_canvases), 4)]:
            if canvas.mode != "text" or not canvas.text:
                continue
            heuristic_records.append(heuristic_candidate_record(anchor, canvas, miner))

    heuristic_records.sort(key=lambda item: item["rank_score"], reverse=True)
    review_limit = len(heuristic_records)
    if not args.no_vlm_review and args.max_vlm_reviews > 0:
        review_limit = min(review_limit, args.max_vlm_reviews)

    records: list[dict[str, Any]] = []
    for idx, record in enumerate(heuristic_records):
        if idx >= review_limit:
            skipped = dict(record)
            skipped["review"] = dict(skipped["review"])
            skipped["review"]["pass"] = False
            skipped["review"]["reason"] = "not_reviewed_by_vlm_due_to_max_vlm_reviews"
            records.append(skipped)
            continue
        if args.no_vlm_review:
            records.append(record)
            continue
        anchor = Anchor(
            image_path=str(record["image_path"]),
            cluster_id=int(record["cluster_id"]),
            isolation_score=float(record["isolation_score"]),
            canvas_mode="text",
            candidate_canvases=[],
            joint_score=float(record["anchor_joint_score"]),
        )
        image = Image.open(record["image_path"]).convert("RGB")
        canvas_data = record["canvas"]
        bbox = tuple(int(v) for v in canvas_data["bbox"])
        canvas = Canvas(
            id=int(canvas_data["id"]),
            mode=str(canvas_data["mode"]),
            mask=mask_from_bbox(image.size, bbox),
            bbox=bbox,
            score=float(canvas_data["score"]),
            text=record.get("text"),
            source=str(canvas_data.get("source", "")),
        )
        review = review_canvas(vlm, image, canvas, True)
        rank_score = float(review["score"]) + min(15.0, max(0.0, float(record["anchor_joint_score"]) / 1000.0))
        records.append(candidate_record(anchor, canvas, review, rank_score))

    records.sort(key=lambda item: item["rank_score"], reverse=True)
    passed = [
        item
        for item in records
        if bool(item["review"].get("pass", False)) and float(item["review"].get("score", 0.0)) >= args.score_threshold
    ]
    selected: list[dict[str, Any]] = []
    seen_images: set[str] = set()
    for record in passed:
        if record["image_path"] in seen_images:
            continue
        selected.append(record)
        seen_images.add(record["image_path"])
        if len(selected) >= args.num_targets:
            break

    (output_dir / "all_candidates.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "selected_candidates.json").write_text(
        json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    selected_dir = output_dir / "selected_images"
    selected_dir.mkdir(parents=True, exist_ok=True)
    review_dir = output_dir / "review"
    for rank, record in enumerate(selected, 1):
        image_path = Path(record["image_path"]).resolve()
        target = selected_dir / safe_name(record["image_path"], rank, str(record["text"] or ""))
        if args.copy_images:
            shutil.copy2(image_path, target)
        else:
            if target.exists() or target.is_symlink():
                target.unlink()
            target.symlink_to(image_path)
        export_review_image(record, review_dir / f"{rank:03d}_{Path(target).stem}.png")
        record["selected_image_path"] = str(target)

    (output_dir / "selected_candidates.json").write_text(
        json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Scanned {len(image_paths)} images, mined {len(anchors)} anchors, reviewed {len(records)} text canvases.")
    print(f"Selected {len(selected)} candidate images -> {selected_dir}")
    for idx, item in enumerate(selected, 1):
        print(
            f"{idx:02d} score={item['rank_score']:.1f} text={item['text']!r} "
            f"image={Path(item['image_path']).name} category={item['review'].get('category', '')}"
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Real SemantiTrace Mode B (Non-text Geometric Insertion) canary generator.

This implements the proper Mode B pipeline that was missing from the earlier
`run_nontext_semantic_benchmark.py` programmatic overlay:

    anchor image
      -> Qwen3-VL canvas planner (surface, color, shape, placement, bbox)
      -> Flux2KleinInpaintEditor masked inpainting (NO text, NO gradient guidance)
      -> Qwen3-VL readability/naturalness gate (color + shape + position)
      -> accepted SemantiTrace Mode B record (parasitism_mode = "Non-text Object Insertion")

A small `--num_canaries 10` smoke run produces a contact sheet so we can
visually compare against the deprecated programmatic overlay baseline.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from semantitrace.backends.real import QwenVLMClient
from semantitrace.config import load_config
from semantitrace.metrics import compute_psnr
from semantitrace.utils.image import mask_from_bbox


COLORS = ["magenta", "cyan", "lime", "orange", "purple", "blue", "red", "yellow"]
SHAPES = ["triangle", "diamond", "circle", "star", "hexagon", "square"]
REGIONS = ["upper left", "upper right", "lower left", "lower right", "center"]

COLOR_ALIASES = {
    "magenta": ["magenta", "pink", "fuchsia"],
    "cyan": ["cyan", "turquoise", "teal"],
    "lime": ["lime", "green", "bright green"],
    "orange": ["orange"],
    "purple": ["purple", "violet"],
    "blue": ["blue"],
    "red": ["red", "crimson"],
    "yellow": ["yellow", "gold"],
}

REGION_ALIASES = {
    "upper left": ["upper left", "top left", "top-left", "upper-left"],
    "upper right": ["upper right", "top right", "top-right", "upper-right"],
    "lower left": ["lower left", "bottom left", "bottom-left", "lower-left"],
    "lower right": ["lower right", "bottom right", "bottom-right", "lower-right"],
    "center": ["center", "middle", "central"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("SemantiTrace Mode B (non-text) real canary generator.")
    parser.add_argument(
        "--source_records",
        default="amlt_n500_upgrade_data/records/canary_records_first500.json",
        help="JSON list of canary records to draw anchors from.",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/nontext_modeb_real_smoke_n10",
    )
    parser.add_argument("--config", default="configs/semantitrace_flux2_klein_gradient_guided.yaml")
    parser.add_argument("--num_canaries", type=int, default=10)
    parser.add_argument("--max_attempts", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_side", type=int, default=768)
    parser.add_argument("--min_side", type=int, default=384)
    parser.add_argument("--mask_area_min", type=float, default=0.012,
                        help="Minimum mask area as fraction of image.")
    parser.add_argument("--mask_area_max", type=float, default=0.06,
                        help="Maximum mask area as fraction of image.")
    parser.add_argument("--num_inference_steps", type=int, default=36)
    parser.add_argument("--guidance_scale", type=float, default=8.5)
    parser.add_argument("--strength", type=float, default=0.99)
    parser.add_argument("--min_masked_delta", type=float, default=0.08,
                        help="Minimum mean pixel delta INSIDE the mask. Rejected if FLUX barely edits the region.")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def round_to_multiple(x: int, base: int = 16) -> int:
    return max(base, (x // base) * base)


def resize_for_model(image: Image.Image, max_side: int, min_side: int) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    scale = min(1.0, max_side / max(w, h))
    if min(w, h) * scale < min_side:
        scale = min_side / min(w, h)
    new_w = round_to_multiple(int(w * scale), 16)
    new_h = round_to_multiple(int(h * scale), 16)
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def caption_prompt() -> str:
    return (
        "Describe this image in one concise noun phrase for retrieval. "
        "Mention the main scene/object/material. Return only the phrase."
    )


def planner_prompt(scene_caption: str, image_w: int, image_h: int) -> str:
    color_list = ", ".join(COLORS)
    shape_list = ", ".join(SHAPES)
    region_list = ", ".join(REGIONS)
    return (
        "You are a vision planner for embedding a small NON-TEXT geometric sticker into a real photo "
        "as a visual provenance canary. The sticker is a single uniformly-colored geometric "
        "shape with NO letters or numbers. It must look like a real adhesive sticker, badge, or "
        "decal that could plausibly already exist in the scene.\n\n"
        f"Scene description: {scene_caption}\n"
        f"Image size: width={image_w} pixels, height={image_h} pixels.\n\n"
        "Pick a single insertion plan that satisfies all constraints:\n"
        "1. SURFACE: the physical object/material the sticker will adhere to "
        "(e.g. laptop lid, water bottle, notebook cover, helmet, locker door, lamp post, "
        "package label, jacket, signpost, wall, dashboard, mug, refrigerator door). "
        "It must be a flat-enough region currently visible in the image, "
        "NOT on a person's face, hands, eyes, or directly over important text.\n"
        f"2. COLOR: one of [{color_list}].\n"
        f"3. SHAPE: one of [{shape_list}].\n"
        f"4. POSITION_REGION: one of [{region_list}] describing which 5x5-cell region the sticker should be in.\n"
        "5. BBOX: pixel-space bounding box [x1, y1, x2, y2] for the sticker. "
        "Width and height should each be between 8% and 22% of the smaller image side. "
        "BBOX must lie fully on the chosen surface and inside the image.\n"
        "6. PLACEMENT_NOTES: one short sentence describing where on the surface (\"upper-left corner of the laptop lid\").\n\n"
        "Respond in strict JSON with keys: surface, color, shape, position_region, "
        "bbox (list of 4 integers), placement_notes. No extra prose."
    )


def parse_planner_response(text: str, image_w: int, image_h: int) -> dict[str, Any] | None:
    if not text:
        return None
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        plan = json.loads(match.group(0))
    except Exception:
        return None
    color = str(plan.get("color", "")).strip().lower()
    shape = str(plan.get("shape", "")).strip().lower()
    region = str(plan.get("position_region", "")).strip().lower()
    surface = str(plan.get("surface", "")).strip()
    placement = str(plan.get("placement_notes", "")).strip()
    bbox = plan.get("bbox")
    if color not in COLORS or shape not in SHAPES or region not in REGIONS:
        return None
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    except Exception:
        return None
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image_w - 1, x2), min(image_h - 1, y2)
    if x2 <= x1 + 8 or y2 <= y1 + 8:
        return None
    return {
        "surface": surface or "surface in the scene",
        "color": color,
        "shape": shape,
        "position_region": region,
        "bbox": [x1, y1, x2, y2],
        "placement_notes": placement,
    }


def enforce_bbox_area(bbox: list[int], image_w: int, image_h: int, area_min: float, area_max: float) -> list[int]:
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    area = w * h
    total = image_w * image_h
    target_min = area_min * total
    target_max = area_max * total
    if area < target_min:
        scale = math.sqrt(target_min / max(1, area))
    elif area > target_max:
        scale = math.sqrt(target_max / max(1, area))
    else:
        scale = 1.0
    if scale != 1.0:
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        new_w = max(24, w * scale)
        new_h = max(24, h * scale)
        x1 = int(round(cx - new_w / 2)); x2 = int(round(cx + new_w / 2))
        y1 = int(round(cy - new_h / 2)); y2 = int(round(cy + new_h / 2))
    x1 = max(4, min(image_w - 12, x1))
    y1 = max(4, min(image_h - 12, y1))
    x2 = max(x1 + 16, min(image_w - 4, x2))
    y2 = max(y1 + 16, min(image_h - 4, y2))
    return [int(x1), int(y1), int(x2), int(y2)]


def build_flux_prompt(plan: dict[str, Any], scene_caption: str) -> str:
    color = plan["color"]
    shape = plan["shape"]
    surface = plan["surface"]
    placement = plan.get("placement_notes") or f"on the {surface}"
    return (
        f"A small solid {color} {shape}-shaped adhesive sticker is firmly applied "
        f"to the {surface}. {placement}. "
        f"The sticker is one single uniform pure {color} color filling its entire {shape} silhouette, "
        f"with a thin dark outline, perfectly crisp clean geometric edges, slight realistic drop shadow, "
        f"natural perspective and lighting matching the surface. "
        f"The sticker contains NO text, NO letters, NO numbers, NO logos, NO patterns, NO gradients. "
        f"The sticker stands out clearly against the underlying surface as a foreign adhesive label. "
        f"Scene context: {scene_caption}. "
        f"Only repaint the masked region. Keep everything outside the mask exactly unchanged."
    )


def verifier_prompt(plan: dict[str, Any]) -> str:
    color = plan["color"]
    shape = plan["shape"]
    region = plan["position_region"]
    return (
        "Examine this image. Report whether there is a SMALL FOREIGN STICKER, BADGE, OR DECAL that "
        "is OBVIOUSLY ADDED on top of the natural scene (it would not exist if the photo were retaken). "
        "Do NOT count: existing album art, printed graphics on packaging, painted logos, natural shadows, "
        "round bottle caps, circular lenses, planets, balls, or any element that belongs to the original scene.\n\n"
        f"Specifically: is there a single small adhesive {color} {shape}-shaped sticker in the {region} of the image, "
        f"clearly added as a foreign object?\n\n"
        "Answer strictly in this format on three lines (use NONE if no such sticker is visible):\n"
        "COLOR: <color or NONE>\n"
        "SHAPE: <shape or NONE>\n"
        "LOCATION: <upper left | upper right | lower left | lower right | center | NONE>"
    )


def parse_verifier_response(text: str) -> dict[str, str]:
    result = {"color": "", "shape": "", "location": ""}
    lower = text.lower()
    for key in ("color", "shape", "location"):
        match = re.search(rf"{key}\s*:\s*([a-zA-Z \-]+)", lower)
        if match:
            result[key] = match.group(1).strip().split("\n")[0].strip()
    return result


def evaluate_hit(parsed: dict[str, str], plan: dict[str, Any]) -> dict[str, bool]:
    color_aliases = [a.lower() for a in COLOR_ALIASES.get(plan["color"], [plan["color"]])]
    region_aliases = [a.lower() for a in REGION_ALIASES.get(plan["position_region"], [plan["position_region"]])]
    color_hit = any(alias in parsed["color"] for alias in color_aliases)
    shape_hit = plan["shape"].lower() in parsed["shape"]
    pos_hit = any(alias in parsed["location"] or alias.replace(" ", "") in parsed["location"].replace(" ", "")
                  for alias in region_aliases)
    if "none" in parsed["color"].split() or "none" in parsed["shape"].split():
        color_hit = shape_hit = pos_hit = False
    return {
        "color_hit": bool(color_hit),
        "shape_hit": bool(shape_hit),
        "position_hit": bool(pos_hit),
        "color_shape_hit": bool(color_hit and shape_hit),
        "strict_hit": bool(color_hit and shape_hit and pos_hit),
    }


def build_inpaint_pipe(config: dict[str, Any], device: str):
    import torch
    from diffusers import Flux2KleinInpaintPipeline

    models = config.get("models", {})
    editor_cfg = config.get("editor", {})
    dtype = getattr(torch, editor_cfg.get("torch_dtype", "bfloat16"))
    pipe = Flux2KleinInpaintPipeline.from_pretrained(
        models.get("inpaint_model", "black-forest-labs/FLUX.2-klein-9B"),
        torch_dtype=dtype,
    ).to(device)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    return pipe, torch


def feather_mask(mask: np.ndarray, radius: int = 4) -> Image.Image:
    from PIL import ImageFilter

    img = Image.fromarray(mask.astype("uint8") * 255, mode="L")
    if radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return img


def masked_pixel_delta(clean: Image.Image, edited: Image.Image, mask: np.ndarray) -> float:
    a = np.asarray(clean.convert("RGB"), dtype=np.float32)
    b = np.asarray(edited.convert("RGB").resize(clean.size), dtype=np.float32)
    diff = np.abs(a - b).mean(axis=2)
    sel = mask.astype(bool)
    if not sel.any():
        return 0.0
    return float(diff[sel].mean() / 255.0)


def quality_delta(clean: Image.Image, edited: Image.Image) -> float:
    a = np.asarray(clean.convert("RGB"), dtype=np.float32)
    b = np.asarray(edited.convert("RGB").resize(clean.size), dtype=np.float32)
    return float(np.mean(np.abs(a - b)) / 255.0)


def safe_psnr(clean: Image.Image, edited: Image.Image) -> float | str:
    value = compute_psnr(np.asarray(clean.convert("RGB")),
                         np.asarray(edited.convert("RGB").resize(clean.size)))
    return "inf" if math.isinf(value) else float(value)


def contact_sheet(records: list[dict[str, Any]], out_dir: Path) -> Path:
    if not records:
        return out_dir / "contact_sheet.jpg"
    cell_w, cell_h = 360, 240
    label_h = 56
    rows = len(records)
    sheet = Image.new("RGB", (cell_w * 2, rows * (cell_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    for idx, rec in enumerate(records):
        y = idx * (cell_h + label_h)
        clean = Image.open(resolve(rec["anchor_image_path"])).convert("RGB")
        wm = Image.open(resolve(rec["watermarked_image_path"])).convert("RGB")
        for col, im in enumerate((clean, wm)):
            thumb = im.copy()
            thumb.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
            bg = Image.new("RGB", (cell_w, cell_h), (245, 245, 245))
            bg.paste(thumb, ((cell_w - thumb.width) // 2, (cell_h - thumb.height) // 2))
            sheet.paste(bg, (col * cell_w, y + label_h))
        plan = rec["nontext_plan"]
        verify = rec["nontext_verification"]
        label = (
            f"{rec['id']}  target={plan['color']}/{plan['shape']}/{plan['position_region']}  "
            f"surface={plan['surface'][:50]}\n"
            f"verifier color/shape/pos -> color_hit={verify['evaluation']['color_hit']} "
            f"shape_hit={verify['evaluation']['shape_hit']} pos_hit={verify['evaluation']['position_hit']} "
            f"strict={verify['evaluation']['strict_hit']}"
        )
        draw.multiline_text((8, y + 6), label, fill=(0, 0, 0), font=font, spacing=2)
    out_path = out_dir / "contact_sheet.jpg"
    sheet.save(out_path, quality=92, optimize=True)
    return out_path


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "watermarked").mkdir(exist_ok=True)
    (out_dir / "clean").mkdir(exist_ok=True)

    cfg = load_config(args.config)
    models = cfg.get("models", {})
    vlm_cfg = cfg.get("vlm", {})

    rng = random.Random(args.seed)
    source_records = json.loads(resolve(args.source_records).read_text(encoding="utf-8"))
    rng.shuffle(source_records)

    vlm = QwenVLMClient(
        model_name=models.get("surrogate_vlm", "Qwen/Qwen3-VL-8B-Instruct"),
        device=args.device,
        torch_dtype=vlm_cfg.get("torch_dtype", "bfloat16"),
    )
    pipe, torch = build_inpaint_pipe(cfg, args.device)
    seq_len = int(cfg.get("editor", {}).get("max_sequence_length", 512))

    accepted: list[dict[str, Any]] = []
    attempted: list[dict[str, Any]] = []

    for source in source_records:
        if len(accepted) >= args.num_canaries:
            break
        try:
            clean_src = resolve(source["anchor_image_path"])
            if not clean_src.exists():
                continue
            clean = Image.open(clean_src).convert("RGB")
            clean = resize_for_model(clean, args.max_side, args.min_side)
        except Exception as exc:
            attempted.append({"source": source.get("id"), "skip": f"open_clean_error:{exc}"})
            continue
        w, h = clean.size
        caption = vlm.generate(clean, caption_prompt(), temperature=0.0, max_new_tokens=64)
        caption = re.sub(r"\s+", " ", caption).strip(" .")

        plan_raw = vlm.generate(clean, planner_prompt(caption, w, h), temperature=0.0, max_new_tokens=320)
        plan = parse_planner_response(plan_raw, w, h)
        if plan is None:
            attempted.append({"source": source.get("id"), "skip": "planner_parse_failed", "raw": plan_raw[:240]})
            continue
        plan["bbox"] = enforce_bbox_area(plan["bbox"], w, h, args.mask_area_min, args.mask_area_max)
        mask = mask_from_bbox(clean.size, plan["bbox"])
        mask_image = feather_mask(mask, radius=4)

        prompt = build_flux_prompt(plan, caption)
        edited: Image.Image | None = None
        verifier_response = ""
        evaluation: dict[str, bool] | None = None
        masked_delta = 0.0
        attempt_records: list[dict[str, Any]] = []
        for attempt in range(args.max_attempts):
            seed = args.seed + len(accepted) * 9973 + attempt * 17
            generator = torch.Generator(device=args.device).manual_seed(seed)
            try:
                with torch.inference_mode():
                    result = pipe(
                        prompt=prompt,
                        image=clean,
                        mask_image=mask_image,
                        height=h,
                        width=w,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        strength=args.strength,
                        generator=generator,
                        max_sequence_length=seq_len,
                    ).images[0]
            except Exception as exc:
                attempt_records.append({"attempt": attempt, "edit_error": str(exc)})
                continue
            edited = result
            masked_delta = masked_pixel_delta(clean, edited, mask)
            verifier_response = vlm.generate(edited, verifier_prompt(plan), temperature=0.0, max_new_tokens=80)
            parsed = parse_verifier_response(verifier_response)
            evaluation = evaluate_hit(parsed, plan)
            attempt_records.append({
                "attempt": attempt,
                "masked_pixel_delta": masked_delta,
                "verifier_response": verifier_response,
                "parsed": parsed,
                "evaluation": evaluation,
            })
            if masked_delta >= args.min_masked_delta and evaluation["color_shape_hit"]:
                break
        if edited is None or evaluation is None:
            attempted.append({"source": source.get("id"), "skip": "edit_failed",
                              "plan": plan, "attempts": attempt_records})
            continue
        if masked_delta < args.min_masked_delta:
            attempted.append({"source": source.get("id"), "skip": "low_masked_delta",
                              "plan": plan, "masked_delta": masked_delta,
                              "attempts": attempt_records})
            continue
        if not evaluation["color_shape_hit"]:
            attempted.append({"source": source.get("id"), "skip": "verifier_color_shape_miss",
                              "plan": plan, "attempts": attempt_records})
            continue

        rec_id = f"nontextmodeb-{len(accepted):04d}"
        wm_path = out_dir / "watermarked" / f"{rec_id}_{plan['color']}_{plan['shape']}_{plan['position_region'].replace(' ', '-')}.png"
        clean_path = out_dir / "clean" / f"{rec_id}_{Path(clean_src).stem}.png"
        clean.save(clean_path)
        edited.save(wm_path)

        record = {
            "id": rec_id,
            "source_semantitrace_id": source.get("id"),
            "anchor_image_path": str(clean_path.relative_to(ROOT)),
            "watermarked_image_path": str(wm_path.relative_to(ROOT)),
            "parasitism_mode": "Non-text Object Insertion",
            "scene_caption": caption,
            "nontext_plan": plan,
            "nontext_verification": {
                "verifier_response": verifier_response,
                "parsed": parse_verifier_response(verifier_response),
                "evaluation": evaluation,
                "attempts": attempt_records,
                "masked_pixel_delta": masked_delta,
            },
            "trap_signature": f"{plan['position_region']} {plan['color']} {plan['shape']}",
            "trigger_prompt": prompt,
            "selected_canvas": {
                "id": 0,
                "mode": "non_text_object_insertion",
                "bbox": plan["bbox"],
                "text": None,
                "source": "nontext_modeb_real",
            },
            "injection_metrics": {
                "render_strategy": "flux2_klein_native_inpaint_nontext_direct",
                "effective_mask_bbox": plan["bbox"],
                "effective_mask_area": int(np.count_nonzero(mask)),
                "effective_mask_area_ratio": float(np.count_nonzero(mask)) / (w * h),
                "quality_local_delta": quality_delta(clean, edited),
                "masked_pixel_delta": masked_delta,
                "psnr": safe_psnr(clean, edited),
                "num_inference_steps": args.num_inference_steps,
                "guidance_scale": args.guidance_scale,
                "strength": args.strength,
            },
            "probe_queries": [
                f"Find the image described as: {caption}. Inspect it for a small non-text geometric sticker. "
                f"If one is visible, report its location, color, and shape; otherwise answer NONE.",
                f"Using only the retrieved visual evidence from this scene ({caption}), identify any small colored geometric marker. "
                f"Report location, color, and shape, or NONE if absent.",
                "Look carefully for a small non-text sticker or geometric marker in the retrieved image. "
                "Do not guess. Answer with location, color, and shape, or NONE.",
            ],
        }
        accepted.append(record)
        (out_dir / "canary_records.json").write_text(json.dumps(accepted, indent=2, ensure_ascii=False),
                                                     encoding="utf-8")
        print(f"accept {rec_id} {Path(clean_src).name} plan={plan['color']}/{plan['shape']}/{plan['position_region']} "
              f"surface={plan['surface'][:60]!r} masked_delta={masked_delta:.3f} "
              f"verifier_hit={evaluation['color_shape_hit']}", flush=True)

    summary = {
        "num_accepted": len(accepted),
        "num_rejected": len(attempted),
        "num_attempted": len(accepted) + len(attempted),
        "target_count": args.num_canaries,
        "source_records": args.source_records,
        "config": args.config,
        "guidance_scale": args.guidance_scale,
        "strength": args.strength,
        "num_inference_steps": args.num_inference_steps,
        "min_masked_delta": args.min_masked_delta,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "rejected_attempts.json").write_text(json.dumps(attempted, indent=2, ensure_ascii=False), encoding="utf-8")
    sheet_path = contact_sheet(accepted, out_dir)
    print("contact sheet:", sheet_path, flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

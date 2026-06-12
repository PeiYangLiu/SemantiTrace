#!/usr/bin/env python
"""Truly parasitic SemantiTrace Mode B canary generator.

Instead of overlaying a synthetic colored geometric sticker, this version asks
Qwen3-VL to choose a *natural physical object* that would plausibly already
exist in the scene (e.g. a ceramic mug on a desk, an enamel pin on a jacket,
a rubber duck on a windowsill). The trap signature is the triple
``(color, object_class, region)`` which is rare enough to give a provenance
signal while letting the inserted object blend naturally with the scene.

Verifier uses a side-by-side clean-vs-watermarked composite so Qwen3-VL is
forced to report what is NEW in the edited image rather than being anchored
by the planned color/shape.
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
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from semantitrace.backends.real import QwenVLMClient
from semantitrace.config import load_config
from semantitrace.metrics import compute_psnr
from semantitrace.modeb_queries import build_modeb_scene_hook_queries
from semantitrace.utils.image import mask_from_bbox


COLORS = ["red", "orange", "yellow", "green", "blue", "purple", "pink", "white", "black", "teal"]

NATURAL_OBJECTS = [
    "ceramic mug", "coffee cup", "water bottle", "glass jar", "tin can",
    "ripe apple", "ripe orange", "lemon", "banana", "small bowl",
    "paperback book", "notebook", "spiral notepad", "manila envelope",
    "pen cup", "scissors", "stapler", "tape dispenser", "eraser",
    "rubber duck", "small plush toy", "keychain", "figurine",
    "potted succulent", "small flowerpot", "candle in glass jar",
    "enamel pin", "hair clip", "bottle cap", "lighter",
    "deck of playing cards", "matchbox", "spice jar", "tea tin",
]

REGIONS = ["upper left", "upper right", "lower left", "lower right", "center"]

REGION_ALIASES = {
    "upper left": ["upper left", "top left", "top-left", "upper-left"],
    "upper right": ["upper right", "top right", "top-right", "upper-right"],
    "lower left": ["lower left", "bottom left", "bottom-left", "lower-left"],
    "lower right": ["lower right", "bottom right", "bottom-right", "lower-right"],
    "center": ["center", "middle", "central"],
}

COLOR_ALIASES = {
    "red": ["red", "crimson", "scarlet", "maroon"],
    "orange": ["orange", "amber"],
    "yellow": ["yellow", "gold", "mustard"],
    "green": ["green", "lime", "olive", "emerald"],
    "blue": ["blue", "navy", "azure", "cobalt"],
    "purple": ["purple", "violet", "lavender"],
    "pink": ["pink", "magenta", "fuchsia", "rose"],
    "white": ["white", "ivory", "cream"],
    "black": ["black"],
    "teal": ["teal", "turquoise", "cyan"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("SemantiTrace Mode B (natural object insertion) real canary generator.")
    parser.add_argument(
        "--source_records",
        default="amlt_n500_upgrade_data/records/canary_records_first500.json",
    )
    parser.add_argument("--output_dir", default="outputs/nontext_modeb_natural_smoke_n10")
    parser.add_argument("--config", default="configs/semantitrace_flux2_klein_gradient_guided.yaml")
    parser.add_argument("--num_canaries", type=int, default=10)
    parser.add_argument("--max_attempts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_side", type=int, default=768)
    parser.add_argument("--min_side", type=int, default=384)
    parser.add_argument("--mask_area_min", type=float, default=0.012)
    parser.add_argument("--mask_area_max", type=float, default=0.060)
    parser.add_argument("--num_inference_steps", type=int, default=36)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--strength", type=float, default=0.99)
    parser.add_argument("--min_masked_delta", type=float, default=0.08)
    parser.add_argument("--shard_count", type=int, default=1,
                        help="Total number of shards; if >1, each shard targets num_canaries/shard_count canaries.")
    parser.add_argument("--shard_index", type=int, default=0,
                        help="0-indexed shard id; used to deterministically split the (color, region) target grid and source records.")
    parser.add_argument("--planner_mode", choices=["grid", "free"], default="grid",
                        help="grid forces balanced color/region cells; free lets the planner choose the most natural object/color/location.")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def rel_to_root(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


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
        "Mention the main scene, setting, and dominant objects. Return only the phrase."
    )


def planner_prompt(
    scene_caption: str,
    image_w: int,
    image_h: int,
    rng: random.Random,
    forced_color: str | None = None,
    forced_region: str | None = None,
    avoid_objects: set[str] | None = None,
) -> str:
    object_pool = [o for o in NATURAL_OBJECTS if not avoid_objects or o not in avoid_objects]
    if len(object_pool) < 4:
        object_pool = list(NATURAL_OBJECTS)
    object_pool = rng.sample(object_pool, min(14, len(object_pool)))
    color_line = (
        f"2. COLOR: it MUST be {forced_color}. Pick an object whose dominant body color is naturally {forced_color}.\n"
        if forced_color else
        f"2. COLOR: a single dominant color of the object from [{', '.join(COLORS)}].\n"
    )
    region_line = (
        f"4. POSITION_REGION: it MUST be {forced_region}. Choose a surface that lies in the {forced_region} of the image.\n"
        if forced_region else
        f"4. POSITION_REGION: one of [{', '.join(REGIONS)}] indicating which 5x5-cell region the object lies in.\n"
    )
    avoid_line = (
        f"\nDo NOT propose any of these object classes (already used many times): {', '.join(sorted(avoid_objects))}.\n"
        if avoid_objects else ""
    )
    obj_list = ", ".join(object_pool)
    return (
        "You are a vision planner for inserting a small NATURAL PHYSICAL OBJECT into a real photo "
        "as a visual provenance canary. The inserted object must look like it could plausibly "
        "ALREADY EXIST in this scene given its context. Do NOT propose stickers, decals, badges, "
        "watermarks, logos, geometric overlays, or anything that screams 'added on top'.\n\n"
        f"Scene description: {scene_caption}\n"
        f"Image size: width={image_w} pixels, height={image_h} pixels.{avoid_line}\n"
        "Choose ONE candidate plan satisfying all rules:\n"
        f"1. OBJECT_CLASS: pick a small everyday physical object from this list that fits the scene: [{obj_list}].\n"
        f"{color_line}"
        "3. SURFACE: the existing physical surface in this image where the object would naturally sit "
        "(e.g. 'desk surface near the laptop', 'kitchen counter beside the sink', 'shelf in the corner'). "
        "NEVER on a person's face, hands, eyes, or directly over existing visible text or logos.\n"
        f"{region_line}"
        "5. BBOX: pixel bounding box [x1, y1, x2, y2] for the object. Width and height should each be "
        "between 8% and 22% of the smaller image side. BBOX must lie on the chosen surface and inside the image.\n"
        "6. PLACEMENT_NOTES: one short sentence describing where on the surface the object sits.\n"
        "7. NATURALNESS_RATIONALE: one short sentence explaining why this object would plausibly already exist in this scene.\n\n"
        "Respond in strict JSON with keys: object_class, color, surface, position_region, bbox, "
        "placement_notes, naturalness_rationale. Output ONLY the JSON object, no markdown fence, no prose."
    )


def extract_first_json(text: str) -> str | None:
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    start = cleaned.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return cleaned[start : i + 1]
    return None


def parse_planner_response(
    text: str,
    image_w: int,
    image_h: int,
    forced_color: str | None = None,
    forced_region: str | None = None,
) -> dict[str, Any] | None:
    snippet = extract_first_json(text)
    if not snippet:
        return None
    try:
        plan = json.loads(snippet)
    except Exception:
        return None
    color = str(plan.get("color", "")).strip().lower()
    obj = str(plan.get("object_class", "")).strip().lower()
    region = str(plan.get("position_region", "")).strip().lower()
    surface = str(plan.get("surface", "")).strip()
    placement = str(plan.get("placement_notes", "")).strip()
    rationale = str(plan.get("naturalness_rationale", "")).strip()
    bbox = plan.get("bbox")
    if forced_color and color != forced_color:
        return None
    if forced_region and region != forced_region:
        return None
    if color not in COLORS or region not in REGIONS or not obj:
        return None
    repaired_bbox = False
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        bbox = fallback_bbox_for_region(region, image_w, image_h)
        repaired_bbox = True
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    except Exception:
        x1, y1, x2, y2 = fallback_bbox_for_region(region, image_w, image_h)
        repaired_bbox = True
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image_w - 1, x2), min(image_h - 1, y2)
    if x2 <= x1 + 8 or y2 <= y1 + 8:
        x1, y1, x2, y2 = fallback_bbox_for_region(region, image_w, image_h)
        repaired_bbox = True
    return {
        "object_class": obj,
        "color": color,
        "surface": surface or "surface in the scene",
        "position_region": region,
        "bbox": [x1, y1, x2, y2],
        "bbox_repaired": repaired_bbox,
        "placement_notes": placement,
        "naturalness_rationale": rationale,
    }


def fallback_bbox_for_region(region: str, image_w: int, image_h: int) -> list[int]:
    short = min(image_w, image_h)
    box_w = max(48, int(short * 0.18))
    box_h = max(48, int(short * 0.18))
    centers = {
        "upper left": (0.25, 0.25),
        "upper right": (0.75, 0.25),
        "lower left": (0.25, 0.75),
        "lower right": (0.75, 0.75),
        "center": (0.50, 0.50),
    }
    fx, fy = centers.get(region, centers["center"])
    cx, cy = int(image_w * fx), int(image_h * fy)
    x1 = max(4, min(image_w - box_w - 4, cx - box_w // 2))
    y1 = max(4, min(image_h - box_h - 4, cy - box_h // 2))
    return [int(x1), int(y1), int(x1 + box_w), int(y1 + box_h)]


def enforce_bbox_area(bbox: list[int], image_w: int, image_h: int, area_min: float, area_max: float) -> list[int]:
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    area = max(1, w * h)
    total = image_w * image_h
    target_min = area_min * total
    target_max = area_max * total
    if area < target_min:
        scale = math.sqrt(target_min / area)
    elif area > target_max:
        scale = math.sqrt(target_max / area)
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
    obj = plan["object_class"]
    color = plan["color"]
    surface = plan["surface"]
    placement = plan.get("placement_notes") or f"sitting on the {surface}"
    return (
        f"A single small {color} {obj} is naturally sitting {placement}. "
        "It is a real, physical, three-dimensional object that has always existed in this scene. "
        f"The {obj} blends perfectly with the photograph: it shares the same lighting direction, "
        "shadows, perspective, depth-of-field, focus level, color cast, film grain, and material quality "
        "as the surrounding objects. It casts a soft realistic contact shadow on the surface beneath it. "
        f"The {obj} is NOT a sticker, decal, badge, logo, watermark, or overlay. Do NOT add any text, "
        "letters, numbers, brand names, or graphical labels on or near the object. "
        f"Scene context: {scene_caption}. "
        "Only repaint the masked region. Keep every pixel outside the mask exactly unchanged. "
        "Do not remove or modify any people, faces, hands, text, signs, or existing objects."
    )


def comparison_prompt(plan: dict[str, Any]) -> str:
    color_list = ", ".join(COLORS)
    region_list = ", ".join(REGIONS)
    return (
        "You are shown a SINGLE composite image with two photographs placed side by side:\n"
        "- LEFT half labelled ORIGINAL is the unedited photo.\n"
        "- RIGHT half labelled EDITED is the same photo after one new physical object was inserted.\n\n"
        "Carefully compare LEFT and RIGHT and identify the SINGLE new physical object that appears in "
        "EDITED but NOT in ORIGINAL. Do not list objects that exist in both halves. Do not infer from "
        "this question. If the two halves look identical to you, answer NONE on all four lines.\n\n"
        f"Respond strictly with four lines (allowed COLOR values: {color_list} or NONE; allowed LOCATION "
        f"values: {region_list} or NONE):\n"
        "OBJECT: <short noun phrase or NONE>\n"
        "COLOR: <color or NONE>\n"
        "LOCATION: <location or NONE>\n"
        "PLAUSIBLE_IN_SCENE: <yes|no|NONE>"
    )


def parse_comparison(text: str) -> dict[str, str]:
    result = {"object": "", "color": "", "location": "", "plausible": ""}
    lower = text.lower()
    for key, label in (("object", "object"), ("color", "color"), ("location", "location"),
                        ("plausible", "plausible_in_scene")):
        match = re.search(rf"{label}\s*:\s*([a-z0-9 _\-\.,]+)", lower)
        if match:
            result[key] = match.group(1).strip().split("\n")[0].strip(" .")
    return result


def evaluate_hit(parsed: dict[str, str], plan: dict[str, Any]) -> dict[str, bool]:
    color_aliases = [a.lower() for a in COLOR_ALIASES.get(plan["color"], [plan["color"]])]
    region_aliases = [a.lower() for a in REGION_ALIASES.get(plan["position_region"], [plan["position_region"]])]
    object_words = [w for w in re.split(r"[\s_\-]+", plan["object_class"].lower()) if len(w) >= 3]
    color_hit = any(alias in parsed["color"] for alias in color_aliases)
    obj_hit = any(word in parsed["object"] for word in object_words) if object_words else False
    pos_hit = any(alias in parsed["location"] or alias.replace(" ", "") in parsed["location"].replace(" ", "")
                  for alias in region_aliases)
    plausible = "yes" in parsed["plausible"] or "true" in parsed["plausible"]
    if any(parsed.get(k) in ("none", "", "n/a") for k in ("object", "color", "location")):
        color_hit = obj_hit = pos_hit = False
    return {
        "color_hit": bool(color_hit),
        "object_hit": bool(obj_hit),
        "position_hit": bool(pos_hit),
        "color_object_hit": bool(color_hit and obj_hit),
        "strict_hit": bool(color_hit and obj_hit and pos_hit),
        "plausible_in_scene": bool(plausible),
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
    )
    target_device = torch.device(device)
    if bool(editor_cfg.get("enable_sequential_cpu_offload", False)) or bool(editor_cfg.get("enable_model_cpu_offload", False)):
        if target_device.type != "cuda":
            raise ValueError("FLUX.2 CPU offload requires a CUDA device.")
        gpu_id = 0 if target_device.index is None else int(target_device.index)
        if bool(editor_cfg.get("enable_sequential_cpu_offload", False)):
            if not hasattr(pipe, "enable_sequential_cpu_offload"):
                raise RuntimeError("Diffusers pipeline does not support sequential CPU offload.")
            pipe.enable_sequential_cpu_offload(gpu_id=gpu_id)
        else:
            if not hasattr(pipe, "enable_model_cpu_offload"):
                raise RuntimeError("Diffusers pipeline does not support model CPU offload.")
            pipe.enable_model_cpu_offload(gpu_id=gpu_id)
    else:
        pipe = pipe.to(device)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    return pipe, torch


def feather_mask(mask: np.ndarray, radius: int = 4) -> Image.Image:
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


def composite_side_by_side(clean: Image.Image, edited: Image.Image) -> Image.Image:
    w, h = clean.size
    sep = 20
    header = 40
    out = Image.new("RGB", (w * 2 + sep, h + header), (240, 240, 240))
    out.paste(clean.convert("RGB"), (0, header))
    out.paste(edited.convert("RGB").resize((w, h)), (w + sep, header))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    draw.text((10, 8), "ORIGINAL", fill=(20, 20, 20), font=font)
    draw.text((w + sep + 10, 8), "EDITED", fill=(20, 20, 20), font=font)
    return out


def contact_sheet(records: list[dict[str, Any]], out_dir: Path) -> Path:
    if not records:
        return out_dir / "contact_sheet.jpg"
    cell_w, cell_h = 360, 240
    label_h = 70
    rows = len(records)
    sheet = Image.new("RGB", (cell_w * 2, rows * (cell_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
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
        ev = verify["evaluation"]
        label = (
            f"{rec['id']}  target={plan['color']} {plan['object_class']}  region={plan['position_region']}\n"
            f"surface={plan['surface'][:55]}\n"
            f"verifier object/color/pos -> obj={ev['object_hit']} color={ev['color_hit']} pos={ev['position_hit']}  "
            f"plausible={ev['plausible_in_scene']}"
        )
        draw.multiline_text((8, y + 6), label, fill=(0, 0, 0), font=font, spacing=2)
    out_path = out_dir / "contact_sheet.jpg"
    sheet.save(out_path, quality=92, optimize=True)
    return out_path


def load_json(path: Path, default: Any) -> Any:
    if not path.exists() or path.stat().st_size == 0:
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "watermarked").mkdir(exist_ok=True)
    (out_dir / "clean").mkdir(exist_ok=True)
    (out_dir / "comparison").mkdir(exist_ok=True)

    cfg = load_config(args.config)
    models = cfg.get("models", {})
    vlm_cfg = cfg.get("vlm", {})

    rng = random.Random(args.seed + 1000003 * args.shard_index)
    source_records = json.loads(resolve(args.source_records).read_text(encoding="utf-8"))
    rng.shuffle(source_records)
    if args.shard_count > 1:
        source_records = source_records[args.shard_index::args.shard_count]

    vlm = QwenVLMClient(
        model_name=models.get("surrogate_vlm", "Qwen/Qwen3-VL-8B-Instruct"),
        device=args.device,
        torch_dtype=vlm_cfg.get("torch_dtype", "bfloat16"),
    )
    pipe, torch = build_inpaint_pipe(cfg, args.device)
    seq_len = int(cfg.get("editor", {}).get("max_sequence_length", 512))

    records_path = out_dir / "canary_records.json"
    rejected_path = out_dir / "rejected_attempts.json"
    accepted: list[dict[str, Any]] = load_json(records_path, [])
    attempted: list[dict[str, Any]] = load_json(rejected_path, [])
    rejected_jsonl = out_dir / "rejected_attempts.jsonl"
    seen_sources = {str(row.get("source_semantitrace_id")) for row in accepted if row.get("source_semantitrace_id")}
    seen_sources |= {str(row.get("source")) for row in attempted if row.get("source")}

    def reject(row: dict[str, Any]) -> None:
        attempted.append(row)
        if row.get("source"):
            seen_sources.add(str(row["source"]))
        with rejected_jsonl.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        (out_dir / "rejected_attempts.json").write_text(
            json.dumps(attempted, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(
            f"reject {len(attempted):04d} reason={row.get('skip')} "
            f"source={row.get('source')} forced={row.get('forced_color')}/{row.get('forced_region')}",
            flush=True,
        )

    # Diversity scheduler: pre-build a balanced (color, region) target queue so the
    # final n=N set covers the grid evenly. Within each (color, region) cell we
    # cap individual object_class occurrences to avoid "coffee cup overload".
    from collections import Counter as _Counter

    shard_target = args.num_canaries
    if args.shard_count > 1:
        shard_target = math.ceil(args.num_canaries / args.shard_count)
    target_queue: list[tuple[str, str]] = []
    while len(target_queue) < shard_target:
        cells = [(c, r) for c in COLORS for r in REGIONS]
        rng.shuffle(cells)
        target_queue.extend(cells)
    target_queue = target_queue[:shard_target]
    object_counter: _Counter = _Counter()
    for row in accepted:
        plan = row.get("nontext_plan") if isinstance(row.get("nontext_plan"), dict) else {}
        if plan.get("object_class"):
            object_counter[str(plan["object_class"])] += 1
    max_per_object = max(2, shard_target // 10)

    for source in source_records:
        if len(accepted) >= shard_target:
            break
        if str(source.get("id")) in seen_sources:
            continue
        try:
            clean_src = resolve(source["anchor_image_path"])
            if not clean_src.exists():
                continue
            clean = Image.open(clean_src).convert("RGB")
            clean = resize_for_model(clean, args.max_side, args.min_side)
        except Exception as exc:
            reject({"source": source.get("id"), "skip": f"open_clean_error:{exc}"})
            continue
        w, h = clean.size
        caption = vlm.generate(clean, caption_prompt(), temperature=0.0, max_new_tokens=64)
        caption = re.sub(r"\s+", " ", caption).strip(" .")

        if args.planner_mode == "grid":
            forced_color, forced_region = target_queue[len(accepted)]
        else:
            forced_color, forced_region = None, None
        avoid = {o for o, c in object_counter.items() if c >= max_per_object}
        plan_raw = vlm.generate(
            clean,
            planner_prompt(caption, w, h, rng,
                            forced_color=forced_color,
                            forced_region=forced_region,
                            avoid_objects=avoid),
            temperature=0.0,
            max_new_tokens=400,
        )
        plan = parse_planner_response(plan_raw, w, h,
                                       forced_color=forced_color,
                                       forced_region=forced_region)
        if plan is None:
            reject({"source": source.get("id"), "skip": "planner_parse_failed",
                    "forced_color": forced_color, "forced_region": forced_region,
                    "raw": plan_raw[:240]})
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
            composite = composite_side_by_side(clean, edited)
            verifier_response = vlm.generate(composite, comparison_prompt(plan),
                                              temperature=0.0, max_new_tokens=160)
            parsed = parse_comparison(verifier_response)
            evaluation = evaluate_hit(parsed, plan)
            attempt_records.append({
                "attempt": attempt,
                "masked_pixel_delta": masked_delta,
                "verifier_response": verifier_response,
                "parsed": parsed,
                "evaluation": evaluation,
            })
            if masked_delta >= args.min_masked_delta and evaluation["color_object_hit"]:
                break
        if edited is None or evaluation is None:
            reject({"source": source.get("id"), "skip": "edit_failed",
                    "plan": plan, "attempts": attempt_records})
            continue
        if masked_delta < args.min_masked_delta:
            reject({"source": source.get("id"), "skip": "low_masked_delta",
                    "plan": plan, "masked_delta": masked_delta,
                    "attempts": attempt_records})
            continue
        if not evaluation["color_object_hit"]:
            reject({"source": source.get("id"), "skip": "verifier_color_object_miss",
                    "plan": plan, "attempts": attempt_records})
            continue

        rec_id = f"nontextmodeb-shard{args.shard_index:02d}-{len(accepted):04d}" if args.shard_count > 1 else f"nontextmodeb-{len(accepted):04d}"
        obj_slug = re.sub(r"[^a-z0-9]+", "-", plan["object_class"].lower()).strip("-")
        wm_path = out_dir / "watermarked" / f"{rec_id}_{plan['color']}_{obj_slug}_{plan['position_region'].replace(' ', '-')}.png"
        clean_path = out_dir / "clean" / f"{rec_id}_{Path(clean_src).stem}.png"
        comp_path = out_dir / "comparison" / f"{rec_id}_side_by_side.jpg"
        clean.save(clean_path)
        edited.save(wm_path)
        composite_side_by_side(clean, edited).save(comp_path, quality=92, optimize=True)

        record = {
            "id": rec_id,
            "source_semantitrace_id": source.get("id"),
            "anchor_image_path": rel_to_root(clean_path),
            "watermarked_image_path": rel_to_root(wm_path),
            "comparison_image_path": rel_to_root(comp_path),
            "parasitism_mode": "Non-text Natural Object Insertion",
            "scene_caption": caption,
            "nontext_plan": plan,
            "nontext_verification": {
                "verifier_response": verifier_response,
                "parsed": parse_comparison(verifier_response),
                "evaluation": evaluation,
                "attempts": attempt_records,
                "masked_pixel_delta": masked_delta,
            },
            "trap_signature": f"{plan['position_region']} {plan['color']} {plan['object_class']}",
            "trigger_prompt": prompt,
            "selected_canvas": {
                "id": 0,
                "mode": "non_text_natural_object_insertion",
                "bbox": plan["bbox"],
                "text": None,
                "source": "nontext_modeb_natural",
            },
            "injection_metrics": {
                "render_strategy": "flux2_klein_native_inpaint_natural_object",
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
                *build_modeb_scene_hook_queries(
                    {
                        "scene_caption": caption,
                        "nontext_plan": plan,
                    },
                    num_queries=3,
                )
            ],
        }
        accepted.append(record)
        object_counter[plan["object_class"]] += 1
        seen_sources.add(str(source.get("id")))
        records_path.write_text(json.dumps(accepted, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"accept {rec_id} {Path(clean_src).name} target={plan['color']} {plan['object_class']} "
              f"@ {plan['position_region']}  surface={plan['surface'][:50]!r} "
              f"masked_delta={masked_delta:.3f} verifier_obj_hit={evaluation['object_hit']} "
              f"color_hit={evaluation['color_hit']} pos_hit={evaluation['position_hit']} "
              f"plausible={evaluation['plausible_in_scene']}", flush=True)

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
        "planner_mode": args.planner_mode,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "rejected_attempts.json").write_text(json.dumps(attempted, indent=2, ensure_ascii=False),
                                                    encoding="utf-8")
    sheet_path = contact_sheet(accepted, out_dir)
    print("contact sheet:", sheet_path, flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

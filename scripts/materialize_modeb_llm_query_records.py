#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from semantitrace.backends.real import QwenVLMClient
from semantitrace.modeb_queries import filter_modeb_audit_queries, modeb_forbidden_terms
from semantitrace.records import load_records_with_resolved_paths, resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Materialize Mode-B records with VLM-generated audit queries.")
    parser.add_argument("--records", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--record_root", default=None)
    parser.add_argument("--max_records", type=int, default=500)
    parser.add_argument("--num_queries", type=int, default=3)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--qwen_model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_new_tokens", type=int, default=384)
    parser.add_argument(
        "--allow_object_term",
        action="store_true",
        help="Allow audit queries to mention the exact object class while still hiding color and coarse location.",
    )
    return parser.parse_args()


def extract_json(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None


def draw_bbox(draw: ImageDraw.ImageDraw, bbox: list[int], offset_x: int = 0) -> None:
    if not bbox or len(bbox) != 4:
        return
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 += offset_x
    x2 += offset_x
    for width in range(3):
        draw.rectangle([x1 - width, y1 - width, x2 + width, y2 + width], outline=(255, 40, 40))


def make_planning_panel(clean: Image.Image, edited: Image.Image, bbox: list[int] | None) -> Image.Image:
    clean = clean.convert("RGB")
    edited = edited.convert("RGB").resize(clean.size)
    w, h = clean.size
    sep = 20
    header = 44
    panel = Image.new("RGB", (w * 2 + sep, h + header), (245, 245, 245))
    panel.paste(clean, (0, header))
    panel.paste(edited, (w + sep, header))
    draw = ImageDraw.Draw(panel)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    draw.text((10, 10), "ORIGINAL", fill=(0, 0, 0), font=font)
    draw.text((w + sep + 10, 10), "EDITED", fill=(0, 0, 0), font=font)
    if bbox:
        shifted = [int(bbox[0]), int(bbox[1]) + header, int(bbox[2]), int(bbox[3]) + header]
        draw_bbox(draw, shifted, 0)
        draw_bbox(draw, shifted, w + sep)
    return panel


def query_prompt(record: dict[str, Any], *, allow_object_term: bool = False) -> str:
    plan = record.get("nontext_plan") if isinstance(record.get("nontext_plan"), dict) else {}
    forbidden = ", ".join(modeb_forbidden_terms(record, allow_object_term=allow_object_term))
    object_rule = (
        "- You MAY mention the target object's ordinary object class if it helps retrieve the scene, "
        "but do not reveal its color or coarse image-region label.\n"
        if allow_object_term
        else "- Prefer phrases such as 'small standalone object', 'small item', or a generic superclass rather than the exact target.\n"
    )
    scene = str(record.get("scene_caption") or "the scene")
    surface = str(plan.get("surface") or "the relevant surface")
    placement = str(plan.get("placement_notes") or "the edited local region")
    return (
        "You are designing black-box audit questions for a visual RAG service.\n"
        "The attached image is a comparison panel: ORIGINAL on the left and EDITED on the right. "
        "The EDITED image contains one new natural physical object that acts as a visual provenance signal.\n\n"
        "Each audit question is used in TWO ways inside a black-box RAG:\n"
        "  (1) CLIP-style text-to-image retrieval over a 1M+ image database to bring the target scene into the top-k context.\n"
        "  (2) A VLM is given the retrieved images and the question, and must describe the new object in its answer.\n\n"
        "Hard rules:\n"
        "- Each question MUST first describe distinctive, retrievable visual content of THIS scene in 1-2 clauses, "
        "  using concrete appearance details (subjects, clothing/colors of people, posture or action, dominant objects, "
        "  setting/layout, time of day, materials) so that text-to-image retrieval can isolate this specific scene from a million distractors. "
        "  Vague phrases like 'in the scene' or 'in the image' alone are NOT acceptable.\n"
        "- After the scene description, ask about the small newly salient object near a specific surface/relation. "
        "  Bind it to the scene with phrases like 'next to', 'beside', 'near the X', 'on the Y'.\n"
        "- Do NOT mention watermark, canary, edited, inserted, original, left image, or right image.\n"
        "- Do NOT reveal the exact hidden signature terms listed below.\n"
        f"{object_rule}"
        "- Each question should end by asking the model to describe the object (color, kind, approximate location) "
        "  or to answer NONE if absent.\n"
        "- Each question MUST be at least 20 words long.\n\n"
        f"Scene caption (rough hint): {scene}\n"
        f"Support surface: {surface}\n"
        f"Placement relation: {placement}\n"
        f"Forbidden exact terms: {forbidden}\n\n"
        "Good example shape (do NOT copy literally, adapt to the actual image you see):\n"
        "  'A bald man in a white checkered shirt is talking with a woman in black at an indoor gathering; "
        "  is there any small object on the floor near his left foot? If so, describe its color, object type, and exact location.'\n\n"
        "Return strict JSON only:\n"
        "{\"queries\": [\"...\", \"...\", \"...\"], \"reasoning\": \"one short sentence\"}"
    )


def main() -> None:
    args = parse_args()
    if args.shard_count < 1:
        raise ValueError("--shard_count must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError("--shard_index must be in [0, shard_count)")
    out = resolve_repo_path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, Any]] = []
    if out.exists() and out.stat().st_size > 0:
        existing = json.loads(out.read_text(encoding="utf-8"))

    records = load_records_with_resolved_paths(
        args.records,
        args.max_records,
        record_root=args.record_root,
        require_images=True,
    )
    indexed_records = [
        (idx, record)
        for idx, record in enumerate(records)
        if idx % args.shard_count == args.shard_index
    ]
    vlm = QwenVLMClient(model_name=args.qwen_model, device=args.device)
    materialized = existing[:]
    start = len(materialized)
    for local_idx, (idx, record) in enumerate(indexed_records[start:], start=start):
        row = {k: v for k, v in record.items() if not k.startswith("_resolved_") and k != "_record_mode"}
        clean = Image.open(record["_resolved_anchor_image_path"]).convert("RGB")
        edited = Image.open(record["_resolved_watermarked_image_path"]).convert("RGB")
        plan = record.get("nontext_plan") if isinstance(record.get("nontext_plan"), dict) else {}
        panel = make_planning_panel(clean, edited, plan.get("bbox"))
        prompt = query_prompt(record, allow_object_term=args.allow_object_term)
        raw = vlm.generate(panel, prompt, temperature=0.0, max_new_tokens=args.max_new_tokens)
        parsed = extract_json(raw) or {}
        candidates = parsed.get("queries") if isinstance(parsed.get("queries"), list) else []
        queries = filter_modeb_audit_queries(
            [str(q) for q in candidates],
            record,
            num_queries=args.num_queries,
            allow_object_term=args.allow_object_term,
        )
        row["probe_queries_original"] = list(row.get("probe_queries", []))
        row["probe_query_policy"] = (
            "modeb_llm_visual_object_hook_v2"
            if args.allow_object_term
            else "modeb_llm_visual_scene_hook_v1"
        )
        row["query_materialization_record_index"] = idx
        row["query_materialization_shard_count"] = int(args.shard_count)
        row["query_materialization_shard_index"] = int(args.shard_index)
        row["probe_queries"] = queries
        row["llm_query_generation"] = {
            "raw_response": raw,
            "parsed": parsed,
            "accepted_queries": queries,
        }
        materialized.append(row)
        out.write_text(json.dumps(materialized, indent=2, ensure_ascii=False), encoding="utf-8")
        print(
            f"[shard {args.shard_index}/{args.shard_count} {len(materialized)}/{len(indexed_records)}] "
            f"global={idx} local={local_idx} {row.get('id')} queries={len(queries)}",
            flush=True,
        )


if __name__ == "__main__":
    main()

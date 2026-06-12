#!/usr/bin/env python
"""Mine anchors specifically for natural-object Mode B canaries.

The canonical n=500 text-mutation suite is biased toward scene text, signs,
book covers, and documents. Those anchors are not ideal for Mode B because a
small natural object has nowhere plausible to sit. This miner instead samples
natural-image pools and asks Qwen3-VL to score whether the image contains a
clear physical support surface for adding a small everyday object.

Output `anchor_records.json` is intentionally minimal but compatible with
`run_nontext_modeb_natural_generation.py`: each record contains an
`anchor_image_path` plus metadata about candidate surfaces.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from semantitrace.backends.real import QwenVLMClient
from semantitrace.config import load_config


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Mine Mode-B-specific natural-object anchors.")
    parser.add_argument(
        "--candidate_dirs",
        nargs="+",
        default=[
            "data_large_retrieval/coco_detection_100k_512",
            "data_webqa_5000/webqa/images",
            "data_expanded/webqa/images",
            "data_expanded/mmqa/images",
        ],
    )
    parser.add_argument("--output_dir", default="outputs/modeb_anchor_mining_smoke")
    parser.add_argument("--config", default="configs/semantitrace_flux2_klein_gradient_guided.yaml")
    parser.add_argument("--target_count", type=int, default=100)
    parser.add_argument("--max_candidates", type=int, default=800)
    parser.add_argument("--shard_count", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min_score", type=float, default=70.0)
    parser.add_argument("--max_side", type=int, default=768)
    parser.add_argument("--min_short_side", type=int, default=256)
    parser.add_argument("--copy_images", action="store_true", default=True)
    parser.add_argument("--no_copy_images", dest="copy_images", action="store_false")
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def list_candidates(dirs: list[str], max_candidates: int, rng: random.Random) -> list[Path]:
    paths: list[Path] = []
    for raw in dirs:
        root = resolve(raw)
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                paths.append(p)
    rng.shuffle(paths)
    filtered: list[Path] = []
    for path in paths:
        if len(filtered) >= max_candidates:
            break
        try:
            with Image.open(path) as im:
                w, h = im.size
                if min(w, h) < 180:
                    continue
                if max(w, h) / max(1, min(w, h)) > 3.2:
                    continue
                filtered.append(path)
        except Exception:
            continue
    return filtered


def resize_for_vlm(image: Image.Image, max_side: int, min_short_side: int) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    scale = min(1.0, max_side / max(w, h))
    if min(w, h) * scale < min_short_side:
        scale = min_short_side / min(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return image.resize((new_w, new_h), Image.Resampling.LANCZOS)


def extract_first_json(text: str) -> str | None:
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


def parse_score(raw: str) -> dict[str, Any] | None:
    snippet = extract_first_json(raw)
    if snippet is None:
        return None
    try:
        data = json.loads(snippet)
    except Exception:
        return None
    try:
        score = float(data.get("score", 0.0))
    except Exception:
        return None
    surfaces = data.get("surfaces")
    if not isinstance(surfaces, list):
        surfaces = []
    return {
        "score": score,
        "scene_type": str(data.get("scene_type", ""))[:160],
        "surfaces": [str(item)[:160] for item in surfaces[:5]],
        "rationale": str(data.get("rationale", ""))[:360],
        "risk_flags": [str(item)[:120] for item in data.get("risk_flags", [])[:6]]
        if isinstance(data.get("risk_flags"), list)
        else [],
        "raw": raw[:1000],
    }


def scoring_prompt() -> str:
    return (
        "Score whether this image is a good anchor for SemantiTrace Mode B: "
        "a small everyday physical object will be inserted by inpainting so it "
        "looks like it naturally belongs in the scene.\n\n"
        "Prefer images with: desks, tables, shelves, counters, floor areas, "
        "windowsills, dashboards, bags, jackets, room corners, kitchen surfaces, "
        "store counters, street ledges, or other clear support surfaces. The "
        "surface should have some empty space where a mug, lemon, tin can, "
        "small plant, bottle cap, notebook, or similar object could plausibly sit.\n\n"
        "Reject or score low: document scans, full-page book covers, posters, "
        "flat text signs, dense typography, close-up faces/hands, no support "
        "surface, highly cluttered scenes, or images where any added object would "
        "look obviously pasted on.\n\n"
        "Respond ONLY as JSON with keys:\n"
        "{\n"
        '  "score": <0-100>,\n'
        '  "scene_type": "<short scene description>",\n'
        '  "surfaces": ["<surface 1>", "<surface 2>"],\n'
        '  "rationale": "<why it is or is not suitable>",\n'
        '  "risk_flags": ["document_scan", "text_heavy", "faces", "no_surface", "cluttered"]\n'
        "}\n"
        "Use score >= 70 only if a natural small-object insertion should be plausible."
    )


def make_contact_sheet(records: list[dict[str, Any]], out_dir: Path, max_rows: int = 80) -> Path:
    if not records:
        return out_dir / "anchor_contact_sheet.jpg"
    cell_w, cell_h = 300, 220
    label_h = 80
    rows = min(len(records), max_rows)
    sheet = Image.new("RGB", (cell_w * 2, rows * (cell_h + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    for i, record in enumerate(records[:rows]):
        y = i * (cell_h + label_h)
        img = Image.open(resolve(record["anchor_image_path"])).convert("RGB")
        thumb = img.copy()
        thumb.thumbnail((cell_w * 2, cell_h), Image.Resampling.LANCZOS)
        bg = Image.new("RGB", (cell_w * 2, cell_h), (245, 245, 245))
        bg.paste(thumb, ((cell_w * 2 - thumb.width) // 2, (cell_h - thumb.height) // 2))
        sheet.paste(bg, (0, y + label_h))
        label = (
            f"{record['id']} score={record['modeb_anchor_score']:.1f} type={record['modeb_scene_type'][:50]}\n"
            f"surface={'; '.join(record.get('modeb_surfaces', [])[:2])[:90]}\n"
            f"risk={','.join(record.get('modeb_risk_flags', [])[:3])}"
        )
        draw.multiline_text((8, y + 6), label, fill=(0, 0, 0), font=font, spacing=2)
    out = out_dir / "anchor_contact_sheet.jpg"
    sheet.save(out, quality=92, optimize=True)
    return out


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def main() -> None:
    args = parse_args()
    out_dir = resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    anchor_dir = out_dir / "anchors"
    anchor_dir.mkdir(exist_ok=True)
    rng = random.Random(args.seed)
    candidates = list_candidates(args.candidate_dirs, args.max_candidates, rng)
    if args.shard_count > 1:
        candidates = candidates[args.shard_index :: args.shard_count]

    cfg = load_config(args.config)
    models = cfg.get("models", {})
    vlm_cfg = cfg.get("vlm", {})
    vlm = QwenVLMClient(
        model_name=models.get("surrogate_vlm", "Qwen/Qwen3-VL-8B-Instruct"),
        device=args.device,
        torch_dtype=vlm_cfg.get("torch_dtype", "bfloat16"),
    )

    records_path = out_dir / "anchor_records.json"
    evaluated_path = out_dir / "evaluated_candidates.jsonl"
    accepted: list[dict[str, Any]] = []
    if records_path.exists() and records_path.stat().st_size > 0:
        accepted = json.loads(records_path.read_text(encoding="utf-8"))
    evaluated: list[dict[str, Any]] = load_jsonl(evaluated_path)
    seen_sources = {str(row.get("source_image_path")) for row in accepted}
    seen_sources |= {str(row.get("source_image_path")) for row in evaluated if row.get("source_image_path")}
    prompt = scoring_prompt()
    for idx, path in enumerate(candidates):
        if len(accepted) >= args.target_count:
            break
        if rel(path) in seen_sources:
            continue
        try:
            image = resize_for_vlm(Image.open(path).convert("RGB"), args.max_side, args.min_short_side)
        except Exception as exc:
            row = {"source_image_path": rel(path), "error": str(exc)}
            evaluated.append(row)
            with evaluated_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            seen_sources.add(rel(path))
            continue
        raw = vlm.generate(image, prompt, temperature=0.0, max_new_tokens=320)
        parsed = parse_score(raw)
        row = {
            "candidate_index": idx,
            "source_image_path": rel(path),
            "width": image.width,
            "height": image.height,
            **(parsed or {"score": 0.0, "scene_type": "", "surfaces": [], "rationale": "", "risk_flags": [], "raw": raw[:1000]}),
        }
        evaluated.append(row)
        with evaluated_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        seen_sources.add(rel(path))
        if parsed is None or parsed["score"] < args.min_score:
            print(f"reject {idx:04d} score={row['score']:.1f} {path.name} {row['scene_type'][:60]!r}", flush=True)
            continue
        record_id = f"modeb-anchor-{len(accepted):04d}"
        if args.copy_images:
            dst = anchor_dir / f"{record_id}_{path.name}"
            shutil.copy2(path, dst)
            anchor_path = rel(dst)
        else:
            anchor_path = rel(path)
        record = {
            "id": record_id,
            "anchor_image_path": anchor_path,
            "source_image_path": rel(path),
            "parasitism_mode": "Mode-B Natural Object Anchor",
            "modeb_anchor_score": parsed["score"],
            "modeb_scene_type": parsed["scene_type"],
            "modeb_surfaces": parsed["surfaces"],
            "modeb_rationale": parsed["rationale"],
            "modeb_risk_flags": parsed["risk_flags"],
            "trap_signature": "",
            "probe_queries": [],
        }
        accepted.append(record)
        records_path.write_text(json.dumps(accepted, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"accept {record_id} score={parsed['score']:.1f} {path.name} {parsed['scene_type'][:70]!r}", flush=True)

    with (out_dir / "anchor_records.csv").open("w", newline="", encoding="utf-8") as fh:
        fields = ["id", "anchor_image_path", "source_image_path", "modeb_anchor_score", "modeb_scene_type", "modeb_surfaces", "modeb_risk_flags"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for record in accepted:
            row = dict(record)
            row["modeb_surfaces"] = " | ".join(record.get("modeb_surfaces", []))
            row["modeb_risk_flags"] = " | ".join(record.get("modeb_risk_flags", []))
            writer.writerow({k: row.get(k) for k in fields})
    summary = {
        "num_candidates": len(candidates),
        "num_evaluated": len(evaluated),
        "num_accepted": len(accepted),
        "target_count": args.target_count,
        "min_score": args.min_score,
        "candidate_dirs": args.candidate_dirs,
    }
    (out_dir / "anchor_mining_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    sheet = make_contact_sheet(accepted, out_dir)
    print("contact sheet:", sheet, flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

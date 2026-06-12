"""Sanity test: use Flux2KleinPipeline (non-inpaint) for free-form object insertion.

The hypothesis: feed the original image + Opus's textual reasoning (surface_type,
style, placement_notes) + canary signature, and let FLUX choose where/how to insert
the object naturally. No mask required.

Usage:
    python scripts/oi_free_editor_sanity.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ANN_PATH = ROOT / "outputs/opus47_full_review_v1/opus_annotations.json"
OUT_DIR = ROOT / "outputs/oi_free_sanity"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def round_to_multiple(x: int, base: int = 16) -> int:
    return max(base, (x // base) * base)


def build_free_prompt(canary: str, proposal: dict) -> str:
    surface = str(proposal.get("surface_type") or "small printed sign").strip()
    style = str(proposal.get("style_description") or "").strip()
    placement = str(proposal.get("placement_notes") or "").strip()
    parts = [
        f'Edit the image: insert a {surface} that prominently displays the '
        f'exact uppercase text "{canary}" (no extra letters, punctuation, or '
        f'spaces). The text must be crisp, large, clearly legible.',
    ]
    if style:
        parts.append(f"Sign style: {style}.")
    if placement:
        parts.append(
            "Place the new sign somewhere natural in the scene, e.g. "
            + placement.rstrip(".")
            + "."
        )
    parts.append(
        "The new sign must look like a real physical object that has always "
        "been part of the scene — matching local lighting, perspective, "
        "material, edges, and shadows."
    )
    parts.append(
        "Keep every other element of the image (people, walls, floor, doors, "
        "objects, layout) identical to the original. Only add the sign; do "
        "not move, alter, or remove anything else."
    )
    return " ".join(parts)


def label_panel(im: Image.Image, label: str) -> Image.Image:
    pad = 36
    out = Image.new("RGB", (im.width, im.height + pad), (245, 245, 245))
    out.paste(im, (0, pad))
    d = ImageDraw.Draw(out)
    try:
        f = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18
        )
    except Exception:
        f = ImageFont.load_default()
    d.text((8, 8), label, fill=(20, 20, 20), font=f)
    return out


CASES = [
    # (stem,         canary,   note)
    ("0d34f1f446ae", "TIMEKI", "office break room (clock case) — already passed"),
    ("4415ad97c45d", "HUFIYUKE", "museum placard (tilted, tiny)"),
    ("524c9d514f29", "ZEPALAPOM", "OPP SHOP A-frame chalkboard"),
    ("b2668340a284", "KUPE",     "Kolind Korskole gray wall sign panel"),
    ("80af3a082d4f", "WACAGETUD", "California Bakery counter front panel"),
]


def run_one(pipe, torch, ann, stem: str, canary: str, seeds: list[int]) -> Path:
    record = None
    for k, v in ann.items():
        if stem in k:
            record = v
            break
    assert record is not None, f"no annotation for {stem}"
    proposal = record["oi_proposal"]
    src_path = ROOT / record["source_path"]
    print(f"\n>>> {stem}  src={src_path.name}")
    print(f"    proposal: {json.dumps(proposal, ensure_ascii=False)}")

    src_img = Image.open(src_path).convert("RGB")
    W, H = src_img.size
    work_w = round_to_multiple(W, 16)
    work_h = round_to_multiple(H, 16)
    work_img = src_img.resize((work_w, work_h), Image.LANCZOS)

    prompt = build_free_prompt(canary, proposal)
    print(f"    prompt: {prompt}")

    panels = [label_panel(src_img, f"{stem}  ORIGINAL")]
    for seed in seeds:
        gen = torch.Generator(device="cuda").manual_seed(seed)
        with torch.inference_mode():
            out = pipe(
                image=work_img,
                prompt=prompt,
                height=work_h,
                width=work_w,
                num_inference_steps=40,
                guidance_scale=4.0,
                generator=gen,
                max_sequence_length=512,
            ).images[0]
        out_native = out.resize((W, H), Image.LANCZOS)
        save_path = OUT_DIR / f"{stem}_free_{canary}_seed{seed}.png"
        out_native.save(save_path)
        print(f"    saved {save_path.name}")
        panels.append(label_panel(out_native, f"free-edit seed={seed} canary={canary}"))

    cw = panels[0].width
    total_h = sum(p.height + 8 for p in panels) - 8
    sheet = Image.new("RGB", (cw, total_h), (245, 245, 245))
    y = 0
    for p in panels:
        sheet.paste(p, (0, y))
        y += p.height + 8
    sheet_path = OUT_DIR / f"{stem}_free_contact_sheet.png"
    sheet.save(sheet_path)
    print(f"    contact sheet -> {sheet_path.name}")
    return sheet_path


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="*", default=None,
                        help="Specific stems to run (default: all 5 failed)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 7777])
    args = parser.parse_args()

    targets = CASES
    if args.cases:
        targets = [(s, c, n) for (s, c, n) in CASES if any(t in s for t in args.cases)]

    ann = json.loads(ANN_PATH.read_text())

    import torch
    from diffusers import Flux2KleinPipeline

    print("loading Flux2KleinPipeline …")
    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-9B", torch_dtype=torch.bfloat16
    ).to("cuda")
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    print("pipeline ready")

    sheets = []
    for stem, canary, note in targets:
        try:
            sheet = run_one(pipe, torch, ann, stem, canary, args.seeds)
            sheets.append((stem, canary, note, sheet))
        except Exception as exc:
            print(f"!! {stem} failed: {exc}")

    # Combined sheet: stack all per-case sheets
    if sheets:
        ims = [Image.open(s).convert("RGB") for _, _, _, s in sheets]
        widths = [i.width for i in ims]
        max_w = max(widths)
        scaled = [i.resize((max_w, int(i.height * max_w / i.width)), Image.LANCZOS) if i.width != max_w else i for i in ims]
        total_h = sum(i.height + 24 for i in scaled)
        combo = Image.new("RGB", (max_w, total_h), (220, 220, 220))
        y = 0
        for img in scaled:
            combo.paste(img, (0, y))
            y += img.height + 24
        combo_path = OUT_DIR / "_all_cases_contact_sheet.png"
        combo.save(combo_path)
        print(f"\nCOMBINED -> {combo_path}")


if __name__ == "__main__":
    main()

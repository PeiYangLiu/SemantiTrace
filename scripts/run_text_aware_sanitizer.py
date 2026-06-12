#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from semantitrace.metrics import normalize_text


COMMON = set(
    "THE AND FOR YOU ARE WITH THIS THAT FROM HAVE NOT SIGN TEXT IMAGE STREET ROAD SHOP "
    "CAFE HOTEL OPEN SALE FOOD MENU STOP LEFT RIGHT CITY PARK BLUE RED WHITE BLACK "
    "GREEN CENTER CENTRE NATIONAL SERVICE SCHOOL MARKET HOUSE STORE MALL PRINCE WALES"
    .split()
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate OCR-aware rare-token sanitizer on saved OCR text.")
    parser.add_argument("--records", default="outputs/semantitrace_n500_textvqa_merged/canary_records_first500_local.json")
    parser.add_argument("--ocr_texts", default="outputs/pipeline_generality_flux_n500_textvqa/_ocr_cache/ocr_texts.json")
    parser.add_argument("--output_dir", default="outputs/text_aware_sanitizer_n500")
    return parser.parse_args()


def rare_tokens(text: str) -> list[str]:
    toks: list[str] = []
    for tok in re.findall(r"[A-Za-z]{4,7}", text.upper()):
        if tok in COMMON:
            continue
        letters = [c for c in tok if c.isalpha()]
        vowels = sum(c in "AEIOU" for c in letters)
        cons = len(letters) - vowels
        if len(letters) >= 4 and (cons / max(1, len(letters)) >= 0.65 or len(set(letters)) >= 4):
            toks.append(tok)
    return toks


def main() -> None:
    args = parse_args()
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = json.loads((ROOT / args.records).read_text(encoding="utf-8"))
    ocr_texts: dict[str, str] = json.loads((ROOT / args.ocr_texts).read_text(encoding="utf-8"))
    canary_paths = {str((ROOT / r["watermarked_image_path"]).resolve()) for r in records}

    exact_hits = []
    rare_flags = []
    detail_rows = []
    for record in records:
        path = str((ROOT / record["watermarked_image_path"]).resolve())
        text = ocr_texts.get(path, "")
        exact = normalize_text(str(record["trap_signature"])) in normalize_text(text)
        rare = rare_tokens(text)
        exact_hits.append(exact)
        rare_flags.append(bool(rare))
        detail_rows.append(
            {
                "id": record["id"],
                "signature": record["trap_signature"],
                "exact_signature_caught": exact,
                "rare_token_flagged": bool(rare),
                "rare_tokens": rare,
                "ocr_text": text,
            }
        )

    distractor_texts = [v for k, v in ocr_texts.items() if k not in canary_paths]
    distractor_flags = [bool(rare_tokens(text)) for text in distractor_texts]
    summary = {
        "num_canaries": len(records),
        "ocr_exact_signature_catch_rate": sum(exact_hits) / len(exact_hits),
        "ocr_exact_signature_caught": sum(exact_hits),
        "rare_token_flag_rate_canaries": sum(rare_flags) / len(rare_flags),
        "rare_token_flagged_canaries": sum(rare_flags),
        "num_distractors_with_ocr": len(distractor_texts),
        "rare_token_flag_rate_distractors": sum(distractor_flags) / max(1, len(distractor_flags)),
        "rare_token_flagged_distractors": sum(distractor_flags),
        "heuristic_note": (
            "Approximate OCR-aware rare-token sanitizer over saved EasyOCR text; "
            "not used as a main defense because false positives on natural scene text are high."
        ),
    }
    (out / "text_aware_sanitizer_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "text_aware_sanitizer_details.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in detail_rows) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

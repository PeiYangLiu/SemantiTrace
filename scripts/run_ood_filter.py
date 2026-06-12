#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from semantitrace import SemantiTracePipeline
from semantitrace.defenses import AdaptiveSanitizer, MahalanobisOODDetector, VLMAnomalyFilter
from semantitrace.utils.image import list_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Evaluate adaptive OOD sanitization")
    parser.add_argument("--clean_dir", required=True)
    parser.add_argument("--suspect_dir", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipeline = SemantiTracePipeline(args.config, device=args.device)
    clean_images = [Image.open(path).convert("RGB") for path in list_images(args.clean_dir)]
    suspect_images = [Image.open(path).convert("RGB") for path in list_images(args.suspect_dir)]
    clean_emb = pipeline.encoder.encode_images(clean_images)
    defenses_cfg = pipeline.config.get("defenses", {})
    detector = MahalanobisOODDetector(
        percentile=float(defenses_cfg.get("mahalanobis_percentile", 99.0)),
        regularization=float(defenses_cfg.get("mahalanobis_regularization", 1e-4)),
        max_components=int(defenses_cfg.get("mahalanobis_max_components", 64)),
        variance_keep=float(defenses_cfg.get("mahalanobis_variance_keep", 0.95)),
    ).fit(clean_emb)
    sanitizer = AdaptiveSanitizer(pipeline.encoder, detector, VLMAnomalyFilter(pipeline.vlm))
    report = sanitizer.sanitize(suspect_images)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()

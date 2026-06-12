from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "defaults": {"seed": 42},
    "backends": {
        "encoder": "deterministic",
        "ocr": "simple",
        "mask_generator": "grid",
        "vlm": "heuristic",
        "editor": "pillow",
    },
    "models": {
        "clip_model": "ViT-L-14",
        "clip_pretrained": "openai",
        "siglip_model": "google/siglip-so400m-patch14-384",
        "ldm_model": "FLUX.2-klein-9B",
        "surrogate_vlm": "Qwen3-VL-8B-Instruct",
        "anomaly_vlm": "Qwen3-VL-32B-Instruct",
    },
    "anchor_mining": {
        "num_clusters": 100,
        "knn_k": 10,
        "min_canvas_area_ratio": 0.01,
        "ocr_confidence_threshold": 0.5,
        "top_candidates_per_cluster": 8,
        "kmeans_iters": 25,
    },
    "canary_generation": {
        "acronym_length": [3, 5],
        "num_probe_queries": 3,
        "vlm_temperature": 0.2,
        "max_retries": 3,
    },
    "dual_guided_diffusion": {
        "lambda_ret": 2.5,
        "lambda_gen": 4.0,
        "eta": 0.5,
        "num_ddim_steps": 50,
        "start_step_ratio": 1.0,
        "guidance_stop_ratio": 0.7,
        "guidance_scale": 7.5,
        "require_gradient_editor": False,
    },
    "verification": {
        "num_canaries": 100,
        "num_probes_per_canary": 3,
        "significance_level": 0.01,
        "retrieval_top_k": 3,
        "clean_default_mean": 0.0,
        "clean_default_std": 0.0001,
    },
    "defenses": {"mahalanobis_percentile": 99.0},
}


def deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if not config_path:
        return config

    path = Path(os.path.expanduser(str(config_path)))
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    if not isinstance(loaded, Mapping):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return deep_update(config, loaded)


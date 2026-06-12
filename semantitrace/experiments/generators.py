from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import requests
from PIL import Image


class AnswerGenerator(Protocol):
    name: str

    def answer(self, query: str, retrieved: list[dict[str, Any]], records_by_path: dict[str, dict[str, Any]]) -> str:
        ...


@dataclass
class HeuristicAnswerGenerator:
    name: str = "heuristic"

    def answer(self, query: str, retrieved: list[dict[str, Any]], records_by_path: dict[str, dict[str, Any]]) -> str:
        norm_paths = {hit["image_path"] for hit in retrieved}
        for path in norm_paths:
            record = records_by_path.get(path)
            if not record:
                continue
            signature = str(record["trap_signature"])
            if signature.lower() in query.lower():
                return f"The retrieved image contains the exact verification signature {signature}."
        match = re.search(r"\b[A-Z]{3,5}\b", query)
        if match and retrieved:
            return "I can see a possible short mark, but cannot verify the requested code."
        return "No exact verification signature is visible."


@dataclass
class EndpointAnswerGenerator:
    name: str
    endpoint: str
    timeout: float = 120.0

    def answer(self, query: str, retrieved: list[dict[str, Any]], records_by_path: dict[str, dict[str, Any]]) -> str:
        response = requests.post(
            self.endpoint,
            json={"query": query, "retrieved": retrieved},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            return str(data.get("response", data.get("text", "")))
        return str(data)


@dataclass
class QwenVLAnswerGenerator:
    name: str
    model_name: str
    device: str = "cuda"
    torch_dtype: str = "bfloat16"
    use_region_crop: bool = True
    crop_padding_ratio: float = 0.35
    min_crop_size: int = 128

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise ImportError("QwenVLAnswerGenerator requires torch and transformers") from exc
        dtype = getattr(torch, self.torch_dtype)
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True, local_files_only=False)
        device_map: str | None = "auto" if self.device != "cpu" else None
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            local_files_only=False,
        )
        if device_map is None:
            self.model = self.model.to(self.device)
        self.model.eval()

    def answer(self, query: str, retrieved: list[dict[str, Any]], records_by_path: dict[str, dict[str, Any]]) -> str:
        image = Image.open(retrieved[0]["image_path"]).convert("RGB") if retrieved else None
        if image is None:
            return ""
        region_record = self._region_record(retrieved[0], records_by_path)
        if self.use_region_crop and region_record is not None:
            crop_box = self._region_crop_box(region_record)
            if crop_box is not None:
                image = self._crop_region(image, crop_box)
                region_note = "A zoomed crop of the queried region is provided. "
            else:
                region_note = ""
        else:
            region_note = ""
        prompt = (
            f"{region_note}Answer using only the retrieved image. The user question may contain a "
            "candidate rare mark as a search key; do not repeat that mark unless it is "
            "actually legible in the image. If the requested mark is not visible, say "
            "that it is not visible and transcribe any different visible text in the "
            "specified region when possible.\n"
            f"Question: {query}"
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt", padding=True)
        input_device = next(self.model.parameters()).device
        inputs = {k: v.to(input_device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]
        with self.torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
        return self.processor.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)[0]

    @staticmethod
    def _region_record(hit: dict[str, Any], records_by_path: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
        image_path = str(hit.get("image_path", ""))
        record = records_by_path.get(image_path)
        if record is None:
            try:
                record = records_by_path.get(str(Path(image_path).resolve()))
            except OSError:
                record = None
        if record is not None:
            return record
        if "selected_canvas" in hit or "injection_metrics" in hit:
            return hit
        return None

    @staticmethod
    def _region_crop_box(record: dict[str, Any]) -> tuple[int, int, int, int] | None:
        metrics = record.get("injection_metrics") if isinstance(record.get("injection_metrics"), dict) else {}
        bbox = metrics.get("effective_mask_bbox") if metrics else None
        if bbox is None:
            canvas = record.get("selected_canvas") if isinstance(record.get("selected_canvas"), dict) else {}
            bbox = canvas.get("bbox") if canvas else None
        if not bbox or len(bbox) != 4:
            return None
        x1, y1, x2, y2 = [int(v) for v in bbox]
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _crop_region(self, image: Image.Image, bbox: tuple[int, int, int, int]) -> Image.Image:
        x1, y1, x2, y2 = bbox
        width, height = image.size
        box_w, box_h = x2 - x1, y2 - y1
        pad = int(max(box_w, box_h) * self.crop_padding_ratio)
        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(width, x2 + pad)
        cy2 = min(height, y2 + pad)
        crop = image.crop((cx1, cy1, cx2, cy2)).convert("RGB")
        scale = max(1.0, self.min_crop_size / max(crop.width, crop.height))
        if scale > 1.0:
            crop = crop.resize((int(crop.width * scale), int(crop.height * scale)), Image.Resampling.LANCZOS)
        return crop


def build_answer_generator(spec: dict[str, Any], device: str = "cuda") -> AnswerGenerator:
    backend = spec.get("backend", "heuristic")
    name = spec.get("name", backend)
    if backend == "heuristic":
        return HeuristicAnswerGenerator(name=name)
    if backend == "endpoint":
        env = spec.get("endpoint_env")
        endpoint = spec.get("endpoint") or (os.environ.get(env) if env else None)
        if not endpoint:
            raise ValueError(f"Generator {name} requires endpoint or environment variable {env}")
        return EndpointAnswerGenerator(name=name, endpoint=endpoint, timeout=float(spec.get("timeout", 120.0)))
    if backend == "qwen_vl":
        return QwenVLAnswerGenerator(
            name=name,
            model_name=spec["model_name"],
            device=device,
            use_region_crop=bool(spec.get("use_region_crop", True)),
            crop_padding_ratio=float(spec.get("crop_padding_ratio", 0.35)),
            min_crop_size=int(spec.get("min_crop_size", 128)),
        )
    raise ValueError(f"Unknown generator backend: {backend}")

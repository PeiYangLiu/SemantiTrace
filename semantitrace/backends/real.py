from __future__ import annotations

from collections import Counter, defaultdict
import re
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from semantitrace.utils.image import bbox_from_mask, ensure_bool_mask, mask_from_bbox


class OpenCLIPEncoder:
    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        device: str = "cuda",
        batch_size: int = 32,
    ) -> None:
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise ImportError(
                "OpenCLIPEncoder requires optional dependencies: "
                "pip install open-clip-torch torch"
            ) from exc

        self.open_clip = open_clip
        self.torch = torch
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        feats = []
        with self.torch.no_grad():
            for start in range(0, len(images), self.batch_size):
                batch = images[start : start + self.batch_size]
                pixels = self.torch.stack([self.preprocess(img.convert("RGB")) for img in batch]).to(self.device)
                emb = self.model.encode_image(pixels)
                emb = self.torch.nn.functional.normalize(emb, dim=-1)
                feats.append(emb.cpu().float().numpy())
        return np.vstack(feats)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        feats = []
        with self.torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                tokens = self.tokenizer(batch).to(self.device)
                emb = self.model.encode_text(tokens)
                emb = self.torch.nn.functional.normalize(emb, dim=-1)
                feats.append(emb.cpu().float().numpy())
        return np.vstack(feats)


class HFSigLIPEncoder:
    """HuggingFace image/text encoder for SigLIP-style retrievers."""

    def __init__(
        self,
        model_name: str = "google/siglip-so400m-patch14-384",
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        batch_size: int = 16,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise ImportError("HFSigLIPEncoder requires torch and transformers") from exc
        self.torch = torch
        self.device = torch.device(device)
        self.batch_size = batch_size
        dtype = getattr(torch, torch_dtype)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(self.device).eval()

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        feats = []
        with self.torch.no_grad():
            for start in range(0, len(images), self.batch_size):
                batch = [img.convert("RGB") for img in images[start : start + self.batch_size]]
                inputs = self.processor(images=batch, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                if hasattr(self.model, "get_image_features"):
                    emb = self.model.get_image_features(**inputs)
                else:
                    out = self.model(**inputs)
                    emb = out.pooler_output if hasattr(out, "pooler_output") else out.last_hidden_state[:, 0]
                emb = self.torch.nn.functional.normalize(emb, dim=-1)
                feats.append(emb.cpu().float().numpy())
        return np.vstack(feats)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        feats = []
        with self.torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start : start + self.batch_size]
                inputs = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                if hasattr(self.model, "get_text_features"):
                    emb = self.model.get_text_features(**inputs)
                else:
                    out = self.model(**inputs)
                    emb = out.pooler_output if hasattr(out, "pooler_output") else out.last_hidden_state[:, 0]
                emb = self.torch.nn.functional.normalize(emb, dim=-1)
                feats.append(emb.cpu().float().numpy())
        return np.vstack(feats)


class EasyOCRDetector:
    def __init__(self, languages: list[str] | None = None, gpu: bool = True) -> None:
        try:
            import easyocr
        except ImportError as exc:
            raise ImportError("EasyOCRDetector requires: pip install easyocr") from exc
        self.reader = easyocr.Reader(languages or ["en"], gpu=gpu)

    def detect_text_regions(self, image: Image.Image) -> list[dict[str, Any]]:
        arr = np.asarray(image.convert("RGB"))
        height, width = arr.shape[:2]
        out: list[dict[str, Any]] = []
        for points, text, confidence in self.reader.readtext(arr):
            xs = [int(p[0]) for p in points]
            ys = [int(p[1]) for p in points]
            bbox = (max(0, min(xs)), max(0, min(ys)), min(width, max(xs)), min(height, max(ys)))
            mask = mask_from_bbox(image.size, bbox)
            out.append(
                {
                    "text": text,
                    "confidence": float(confidence),
                    "bbox": bbox,
                    "mask": mask,
                    "area": int(mask.sum()),
                    "source": "easyocr",
                }
            )
        return out


class SAMMaskGenerator:
    def __init__(self, checkpoint: str, model_type: str = "vit_h", device: str = "cuda") -> None:
        try:
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
            import torch
        except ImportError as exc:
            raise ImportError("SAMMaskGenerator requires: pip install segment-anything torch") from exc
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(torch.device(device))
        sam.eval()
        self.generator = SamAutomaticMaskGenerator(sam)

    def generate_masks(self, image: Image.Image) -> list[dict[str, Any]]:
        arr = np.asarray(image.convert("RGB"))
        masks = []
        for item in self.generator.generate(arr):
            mask = np.asarray(item["segmentation"], dtype=bool)
            masks.append(
                {
                    "mask": mask,
                    "bbox": tuple(int(v) for v in item.get("bbox", (0, 0, 0, 0))),
                    "area": int(item.get("area", mask.sum())),
                    "stability_score": float(item.get("stability_score", 0.0)),
                    "source": "sam",
                }
            )
        return masks


class OpenAICompatibleVLM:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 60.0) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("OpenAICompatibleVLM requires: pip install openai") from exc
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model

    def generate(self, image: Image.Image | None, prompt: str, **kwargs: Any) -> str:
        if image is not None:
            raise NotImplementedError(
                "Image upload is provider-specific. Use a VLM adapter that serializes images "
                "for your endpoint, or pass image=None for text-only tests."
            )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.2),
            max_tokens=kwargs.get("max_new_tokens", 1024),
        )
        return response.choices[0].message.content or ""

    def score_text(self, image: Image.Image, prompt: str, target_text: str) -> float:
        raise NotImplementedError("Log-prob scoring is not exposed by this generic adapter.")


class QwenVLMClient:
    """Local Qwen-VL client for image-conditioned canary planning/filtering."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise ImportError("QwenVLMClient requires torch and transformers") from exc
        self.torch = torch
        self.device = device
        dtype = getattr(torch, torch_dtype)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, local_files_only=False)
        device_map: str | None = "auto" if device != "cpu" else None
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            local_files_only=False,
        )
        if device_map is None:
            self.model = self.model.to(device)
        self.model.eval()

    def generate(self, image: Image.Image | None, prompt: str, **kwargs: Any) -> str:
        content: list[dict[str, Any]] = []
        if image is not None:
            content.append({"type": "image", "image": image.convert("RGB")})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images = [image.convert("RGB")] if image is not None else None
        inputs = self.processor(text=[text], images=images, return_tensors="pt", padding=True)
        input_device = next(self.model.parameters()).device
        inputs = {k: v.to(input_device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]
        with self.torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=int(kwargs.get("max_new_tokens", 512)),
                do_sample=False,
            )
        return self.processor.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)[0]

    def score_text(self, image: Image.Image, prompt: str, target_text: str) -> float:
        # Qwen-VL generation does not expose stable token-level scoring through
        # this lightweight adapter. The true paper implementation uses gradients
        # from a surrogate VLM; this adapter is for semantic planning/filtering.
        raise NotImplementedError("QwenVLMClient does not expose log-prob text scoring.")


class DiffusersInpaintEditor:
    """Real masked semantic editor backed by a diffusers inpainting pipeline."""

    capabilities = {"masked_edit", "diffusers_inpaint"}

    def __init__(
        self,
        model_name: str = "stable-diffusion-v1-5/stable-diffusion-inpainting",
        device: str = "cuda",
        torch_dtype: str = "float16",
        image_size: int = 512,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        strength: float = 0.98,
        seed: int = 42,
        enable_attention_slicing: bool = True,
        max_mask_area_ratio: float = 0.035,
        max_text_mask_area_ratio: float = 0.035,
        max_object_insertion_mask_area_ratio: float = 0.06,
        min_mask_width: int = 96,
        min_mask_height: int = 40,
        label_aspect_ratio: float = 2.6,
        mask_feather_radius: float = 6.0,
        render_signature: bool = True,
        text_opacity: float = 0.72,
        stroke_opacity: float = 0.28,
        text_blur_radius: float = 0.35,
        text_background_opacity: float = 0.82,
        font_height_ratio: float = 0.58,
        font_width_ratio: float = 0.68,
        natural_text_fusion: bool = True,
        text_span_replacement: bool = True,
        text_replacement_unit: str = "word",
        text_bbox_padding_ratio: float = 0.08,
        max_quality_local_delta: float = 0.42,
        max_quality_boundary_delta: float = 0.22,
        min_text_length_ratio: float = 0.0,
        max_text_length_ratio: float = 999.0,
        min_style_foreground_fraction: float = 0.0,
        glyph_clone_text_fusion: bool = False,
    ) -> None:
        try:
            import torch
            from diffusers import AutoPipelineForInpainting
        except ImportError as exc:
            raise ImportError(
                "DiffusersInpaintEditor requires optional dependencies: "
                "pip install diffusers torch transformers accelerate safetensors"
            ) from exc

        self.torch = torch
        self.device = torch.device(device)
        self.model_name = model_name
        self.image_size = int(image_size)
        self.num_inference_steps = int(num_inference_steps)
        self.guidance_scale = float(guidance_scale)
        self.strength = float(strength)
        self.seed = int(seed)
        self.max_mask_area_ratio = float(max_mask_area_ratio)
        self.max_text_mask_area_ratio = float(max_text_mask_area_ratio)
        self.max_object_insertion_mask_area_ratio = float(max_object_insertion_mask_area_ratio)
        self.min_mask_width = int(min_mask_width)
        self.min_mask_height = int(min_mask_height)
        self.label_aspect_ratio = float(label_aspect_ratio)
        self.mask_feather_radius = float(mask_feather_radius)
        self.render_signature = bool(render_signature)
        self.text_opacity = float(text_opacity)
        self.stroke_opacity = float(stroke_opacity)
        self.text_blur_radius = float(text_blur_radius)
        self.text_background_opacity = float(text_background_opacity)
        self.font_height_ratio = float(font_height_ratio)
        self.font_width_ratio = float(font_width_ratio)
        self.natural_text_fusion = bool(natural_text_fusion)
        self.text_span_replacement = bool(text_span_replacement)
        self.text_replacement_unit = str(text_replacement_unit)
        self.text_bbox_padding_ratio = float(text_bbox_padding_ratio)
        self.max_quality_local_delta = float(max_quality_local_delta)
        self.max_quality_boundary_delta = float(max_quality_boundary_delta)
        self.min_text_length_ratio = float(min_text_length_ratio)
        self.max_text_length_ratio = float(max_text_length_ratio)
        self.min_style_foreground_fraction = float(min_style_foreground_fraction)
        self.glyph_clone_text_fusion = bool(glyph_clone_text_fusion)

        dtype = getattr(torch, torch_dtype)
        self.pipe = AutoPipelineForInpainting.from_pretrained(
            model_name,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        self.pipe = self.pipe.to(self.device)
        if enable_attention_slicing and hasattr(self.pipe, "enable_attention_slicing"):
            self.pipe.enable_attention_slicing()

    def edit(
        self,
        image: Image.Image,
        mask: np.ndarray,
        prompt: str,
        guidance: dict[str, Any],
    ) -> Image.Image:
        original_size = image.size
        original_rgb = image.convert("RGB")
        signature = str(guidance.get("trap_signature", ""))
        parasitism_mode = str(guidance.get("parasitism_mode", ""))
        effective_mask = self._refine_mask(original_rgb, mask, parasitism_mode, guidance)
        guidance["effective_mask"] = effective_mask
        guidance["mask_feather_radius"] = self.mask_feather_radius
        work_image = self._resize_for_model(original_rgb)
        work_mask = self._resize_mask(effective_mask, original_size, work_image.size)
        full_prompt = self._compose_prompt(prompt, signature, parasitism_mode)
        negative_prompt = (
            "blurry, low quality, distorted text, misspelled text, extra letters, "
            "floating artifact, unnatural object, watermark overlay, harsh border"
        )
        seed = self.seed + int(guidance.get("edit_attempt", 0)) * 9973
        generator = self.torch.Generator(device=self.device).manual_seed(seed)
        with self.torch.inference_mode():
            result = self.pipe(
                prompt=full_prompt,
                negative_prompt=negative_prompt,
                image=work_image,
                mask_image=work_mask,
                num_inference_steps=int(guidance.get("num_inference_steps", self.num_inference_steps)),
                guidance_scale=float(guidance.get("guidance_scale", self.guidance_scale)),
                strength=float(guidance.get("strength", self.strength)),
                generator=generator,
            ).images[0]
        result = result.resize(original_size, Image.Resampling.LANCZOS)
        if self.render_signature and signature:
            result = self._render_signature(original_rgb, result, effective_mask, signature, parasitism_mode, guidance)
        quality = self._quality_report(original_rgb, result, effective_mask, guidance)
        guidance.update(quality)
        return result

    def _refine_mask(
        self,
        image: Image.Image,
        mask: np.ndarray,
        parasitism_mode: str = "",
        guidance: dict[str, Any] | None = None,
    ) -> np.ndarray:
        bool_mask = ensure_bool_mask(mask, image.size)
        if self._is_text_mode(parasitism_mode) and self.text_span_replacement:
            span_mask = self._text_span_mask(image, bool_mask, guidance or {})
            if span_mask is not None and span_mask.any():
                return span_mask
        width, height = image.size
        if self._is_text_mode(parasitism_mode):
            mask_ratio = self.max_text_mask_area_ratio
        elif "insert" in parasitism_mode.lower() or "object" in parasitism_mode.lower():
            mask_ratio = self.max_object_insertion_mask_area_ratio
        else:
            mask_ratio = self.max_mask_area_ratio
        max_area = max(1, int(width * height * mask_ratio))
        current_area = int(bool_mask.sum())
        if current_area <= max_area:
            return bool_mask

        x1, y1, x2, y2 = bbox_from_mask(bool_mask)
        box_w, box_h = x2 - x1, y2 - y1
        if box_w <= 0 or box_h <= 0:
            return bool_mask

        target_area = min(current_area, max_area)
        target_h = int(round((target_area / max(self.label_aspect_ratio, 1e-6)) ** 0.5))
        target_w = int(round(target_h * self.label_aspect_ratio))
        target_w = min(box_w, max(self.min_mask_width, target_w))
        target_h = min(box_h, max(self.min_mask_height, target_h))
        if target_w > box_w:
            target_w = box_w
        if target_h > box_h:
            target_h = box_h

        px, py = self._choose_low_texture_patch(image, bool_mask, (x1, y1, x2, y2), target_w, target_h)
        refined = np.zeros_like(bool_mask, dtype=bool)
        refined[py : py + target_h, px : px + target_w] = True
        refined &= bool_mask
        return refined if refined.any() else bool_mask

    @staticmethod
    def _choose_low_texture_patch(
        image: Image.Image,
        mask: np.ndarray,
        bbox: tuple[int, int, int, int],
        patch_w: int,
        patch_h: int,
    ) -> tuple[int, int]:
        x1, y1, x2, y2 = bbox
        gray = np.asarray(image.convert("L"), dtype=np.float32)
        padded = np.pad(gray, 1, mode="edge")
        lap = np.abs(
            -4.0 * padded[1:-1, 1:-1]
            + padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
        )
        xs = np.linspace(x1, max(x1, x2 - patch_w), num=9, dtype=int)
        ys = np.linspace(y1, max(y1, y2 - patch_h), num=7, dtype=int)
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        best: tuple[float, int, int] | None = None
        for py in ys:
            for px in xs:
                patch_mask = mask[py : py + patch_h, px : px + patch_w]
                if patch_mask.size == 0 or patch_mask.mean() < 0.95:
                    continue
                patch_gray = gray[py : py + patch_h, px : px + patch_w]
                patch_lap = lap[py : py + patch_h, px : px + patch_w]
                texture = float(patch_lap.mean() + 0.15 * patch_gray.std())
                dist = ((px + patch_w / 2 - center_x) ** 2 + (py + patch_h / 2 - center_y) ** 2) ** 0.5
                score = texture + 0.002 * dist
                if best is None or score < best[0]:
                    best = (score, int(px), int(py))
        if best is None:
            return (x1 + max(0, (x2 - x1 - patch_w) // 2), y1 + max(0, (y2 - y1 - patch_h) // 2))
        return best[1], best[2]

    def _render_signature(
        self,
        original: Image.Image,
        image: Image.Image,
        mask: np.ndarray,
        signature: str,
        parasitism_mode: str = "",
        guidance: dict[str, Any] | None = None,
    ) -> Image.Image:
        if self._is_text_mode(parasitism_mode):
            if self.natural_text_fusion:
                return self._render_natural_text_mutation(original, image, mask, signature, guidance or {})
            return self._render_text_mutation(image, mask, signature)
        return self._render_object_label(original, image, mask, signature, guidance or {})

    def _render_object_label(
        self,
        original: Image.Image,
        image: Image.Image,
        mask: np.ndarray,
        signature: str,
        guidance: dict[str, Any],
    ) -> Image.Image:
        x1, y1, x2, y2 = bbox_from_mask(mask)
        box_w, box_h = x2 - x1, y2 - y1
        if box_w <= 8 or box_h <= 8:
            guidance["render_strategy"] = "object_label_skipped_tiny_mask"
            return image

        base = image.convert("RGBA")
        region = np.asarray(original.convert("RGB"))[y1:y2, x1:x2].astype(np.float32)
        surface = np.median(region.reshape(-1, 3), axis=0) if region.size else np.array([128.0, 128.0, 128.0])
        luminance = float(0.2126 * surface[0] + 0.7152 * surface[1] + 0.0722 * surface[2])
        if luminance < 132:
            label_rgb = tuple(int(np.clip(v + 48, 0, 255)) for v in surface)
            text_rgb = (245, 245, 235)
            border_rgb = (20, 20, 20)
        else:
            label_rgb = tuple(int(np.clip(v - 30, 0, 255)) for v in surface)
            text_rgb = (25, 25, 25)
            border_rgb = (245, 245, 235)

        scale = 4
        label_w = max(10, int(box_w * scale * 0.86))
        label_h = max(10, int(box_h * scale * 0.72))
        label = Image.new("RGBA", (label_w, label_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(label)
        radius = max(2, min(label_w, label_h) // 8)
        shadow = Image.new("RGBA", label.size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow)
        shadow_draw.rounded_rectangle(
            (scale, scale, label_w - scale, label_h - scale),
            radius=radius,
            fill=(0, 0, 0, 58),
        )
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=scale * 0.75))
        label = Image.alpha_composite(label, shadow)
        draw = ImageDraw.Draw(label)
        draw.rounded_rectangle(
            (0, 0, label_w - scale, label_h - scale),
            radius=radius,
            fill=(*label_rgb, int(255 * 0.74)),
            outline=(*border_rgb, int(255 * 0.28)),
            width=max(1, scale),
        )
        font = self._fit_font(
            signature,
            label_w,
            label_h,
            min(0.78, max(self.font_height_ratio, 0.58)),
            min(0.88, max(self.font_width_ratio, 0.76)),
        )
        text_bbox = draw.textbbox((0, 0), signature, font=font, stroke_width=max(1, scale // 2))
        text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        tx = int(max(0, (label_w - text_w) / 2) - text_bbox[0] - scale / 2)
        ty = int(max(0, (label_h - text_h) / 2) - text_bbox[1] - scale / 2)
        draw.text(
            (tx, ty),
            signature,
            font=font,
            fill=(*text_rgb, int(255 * max(self.text_opacity, 0.68))),
            stroke_width=max(1, scale // 2),
            stroke_fill=(*border_rgb, int(255 * 0.16)),
        )
        angle = self._estimate_text_angle(original, mask)
        if abs(angle) >= 1.0:
            label = label.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
        if self.text_blur_radius > 0:
            label = label.filter(ImageFilter.GaussianBlur(radius=min(self.text_blur_radius, 0.20) * scale))
        layer = Image.new("RGBA", (base.width * scale, base.height * scale), (0, 0, 0, 0))
        px = int(x1 * scale + (box_w * scale - label.width) / 2)
        py = int(y1 * scale + (box_h * scale - label.height) / 2)
        layer.alpha_composite(label, (px, py))
        layer = layer.resize(base.size, Image.Resampling.LANCZOS)

        mask_img = Image.fromarray((ensure_bool_mask(mask, image.size).astype(np.uint8) * 255), mode="L")
        alpha = layer.getchannel("A")
        layer.putalpha(Image.composite(alpha, Image.new("L", image.size, 0), mask_img))
        guidance["render_strategy"] = "natural_object_label"
        guidance["estimated_text_angle"] = angle
        return Image.alpha_composite(base, layer).convert("RGB")

    def _render_natural_text_mutation(
        self,
        original: Image.Image,
        image: Image.Image,
        mask: np.ndarray,
        signature: str,
        guidance: dict[str, Any],
    ) -> Image.Image:
        x1, y1, x2, y2 = bbox_from_mask(mask)
        box_w, box_h = x2 - x1, y2 - y1
        if box_w <= 8 or box_h <= 8:
            guidance["render_strategy"] = "natural_text_fusion_skipped_tiny_mask"
            return image

        bool_mask = ensure_bool_mask(mask, image.size)
        style = self._estimate_text_style(original, image, bool_mask)
        guidance.update(self._text_style_metrics(signature, guidance, style))
        base = image.convert("RGBA")
        mask_img = Image.fromarray((bool_mask.astype(np.uint8) * 255), mode="L")
        if self.glyph_clone_text_fusion:
            cloned = self._render_glyph_clone_text(original, base, bool_mask, signature, guidance, style)
            if cloned is not None:
                return cloned

        eraser = Image.new("RGBA", image.size, (*style["background_rgb"], 0))
        erase_alpha = mask_img.filter(ImageFilter.GaussianBlur(radius=min(0.7, max(0.0, self.mask_feather_radius / 6))))
        erase_alpha = erase_alpha.point(lambda value: int(value * max(self.text_background_opacity, 0.98)))
        eraser.putalpha(erase_alpha)
        base = Image.alpha_composite(base, eraser)

        scale = 4
        text_layer = Image.new("RGBA", (base.width * scale, base.height * scale), (0, 0, 0, 0))
        patch = Image.new("RGBA", (max(1, box_w * scale), max(1, box_h * scale)), (0, 0, 0, 0))
        draw = ImageDraw.Draw(patch)
        fg_box = style.get("foreground_bbox") or (0, 0, box_w, box_h)
        fx1, fy1, fx2, fy2 = [int(v) for v in fg_box]
        fg_w = max(1, min(box_w, fx2) - max(0, fx1))
        fg_h = max(1, min(box_h, fy2) - max(0, fy1))
        font = self._fit_font(
            signature,
            int(fg_w * scale * 1.18),
            int(fg_h * scale * 1.28),
            min(0.96, max(self.font_height_ratio, 0.72)),
            min(1.0, max(self.font_width_ratio, 0.82)),
        )
        stroke_width = max(0, int(round(scale * style["stroke_scale"])))
        text_bbox = draw.textbbox((0, 0), signature, font=font, stroke_width=stroke_width)
        text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        fg_cx = (max(0, fx1) + min(box_w, fx2)) * 0.5 * scale
        fg_cy = (max(0, fy1) + min(box_h, fy2)) * 0.5 * scale
        tx = int(fg_cx - text_w / 2 - text_bbox[0])
        ty = int(fg_cy - text_h / 2 - text_bbox[1])
        tx = max(-text_bbox[0], min(tx, box_w * scale - text_w - text_bbox[0]))
        ty = max(-text_bbox[1], min(ty, box_h * scale - text_h - text_bbox[1]))
        guidance["rendered_text_width_ratio"] = float(text_w / max(1, fg_w * scale))
        guidance["rendered_text_height_ratio"] = float(text_h / max(1, fg_h * scale))
        draw.text(
            (tx, ty),
            signature,
            font=font,
            fill=(*style["text_rgb"], int(255 * style["text_opacity"])),
            stroke_width=stroke_width,
            stroke_fill=(*style["stroke_rgb"], int(255 * style["stroke_opacity"])),
        )
        angle = float(style["angle"])
        if abs(angle) >= 1.0:
            patch = patch.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
        if self.text_blur_radius > 0:
            patch = patch.filter(ImageFilter.GaussianBlur(radius=min(self.text_blur_radius, 0.22) * scale))

        px = int(x1 * scale + (box_w * scale - patch.width) / 2)
        py = int(y1 * scale + (box_h * scale - patch.height) / 2)
        text_layer.alpha_composite(patch, (px, py))
        text_layer = text_layer.resize(base.size, Image.Resampling.LANCZOS)
        alpha = text_layer.getchannel("A")
        text_layer.putalpha(Image.composite(alpha, Image.new("L", image.size, 0), mask_img))
        guidance["render_strategy"] = "natural_text_fusion"
        guidance["estimated_text_angle"] = angle
        return Image.alpha_composite(base, text_layer).convert("RGB")

    def _render_glyph_clone_text(
        self,
        original: Image.Image,
        base: Image.Image,
        mask: np.ndarray,
        signature: str,
        guidance: dict[str, Any],
        style: dict[str, Any],
    ) -> Image.Image | None:
        selected = guidance.get("selected_canvas") if isinstance(guidance.get("selected_canvas"), dict) else {}
        original_text = str(selected.get("text") or "")
        source_chars = [char for char in re.sub(r"[^A-Za-z0-9]", "", original_text)]
        target_chars = [char for char in re.sub(r"[^A-Za-z0-9]", "", signature)]
        if len(source_chars) < 2 or len(source_chars) != len(target_chars):
            guidance.setdefault("text_style_flags", []).append("glyph_clone_length_mismatch")
            return None
        source_counts = Counter(char.upper() for char in source_chars)
        target_counts = Counter(char.upper() for char in target_chars)
        if any(target_counts[char] > source_counts.get(char, 0) for char in target_counts):
            guidance.setdefault("text_style_flags", []).append("glyph_clone_missing_source_letter")
            return None

        x1, y1, x2, y2 = bbox_from_mask(mask)
        box_w, box_h = x2 - x1, y2 - y1
        if box_w <= 4 or box_h <= 4:
            guidance.setdefault("text_style_flags", []).append("glyph_clone_tiny_box")
            return None

        text_box = self._expanded_text_box(style, box_w, box_h)
        source_spans = self._equal_character_spans(text_box, len(source_chars))
        positions_by_char: dict[str, list[int]] = defaultdict(list)
        for idx, char in enumerate(source_chars):
            positions_by_char[char.upper()].append(idx)

        original_rgba = original.convert("RGBA")
        erased = original_rgba.copy()
        mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        eraser = Image.new("RGBA", base.size, (*style["background_rgb"], 0))
        erase_alpha = mask_img.filter(ImageFilter.GaussianBlur(radius=min(0.8, max(0.0, self.mask_feather_radius / 6))))
        erase_alpha = erase_alpha.point(lambda value: int(value * max(self.text_background_opacity, 0.98)))
        eraser.putalpha(erase_alpha)
        erased = Image.alpha_composite(erased, eraser)

        layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
        target_spans = self._equal_character_spans(text_box, len(target_chars))
        used: set[int] = set()
        for target_idx, char in enumerate(target_chars):
            choices = positions_by_char.get(char.upper(), [])
            source_idx = next((idx for idx in choices if idx not in used), choices[0] if choices else None)
            if source_idx is None:
                guidance.setdefault("text_style_flags", []).append("glyph_clone_missing_source_letter")
                return None
            used.add(source_idx)
            sx1, sy1, sx2, sy2 = source_spans[source_idx]
            tx1, ty1, tx2, ty2 = target_spans[target_idx]
            cell = original_rgba.crop((x1 + sx1, y1 + sy1, x1 + sx2, y1 + sy2))
            alpha = self._feathered_cell_alpha(cell.size)
            cell.putalpha(alpha)

            target_w = max(1, tx2 - tx1)
            target_h = max(1, ty2 - ty1)
            scale = min(1.08, max(0.92, min(target_w / max(1, cell.width), target_h / max(1, cell.height))))
            if abs(scale - 1.0) > 0.05:
                cell = cell.resize(
                    (max(1, int(round(cell.width * scale))), max(1, int(round(cell.height * scale)))),
                    Image.Resampling.LANCZOS,
                )
            px = x1 + int(round((tx1 + tx2 - cell.width) / 2))
            py = y1 + int(round((ty1 + ty2 - cell.height) / 2))
            layer.alpha_composite(cell, (px, py))

        alpha = layer.getchannel("A")
        layer.putalpha(Image.composite(alpha, Image.new("L", base.size, 0), mask_img))
        guidance["render_strategy"] = "glyph_clone_text_fusion"
        guidance["estimated_text_angle"] = float(style.get("angle", 0.0))
        guidance["glyph_clone_text_fusion"] = True
        guidance["rendered_text_width_ratio"] = 1.0
        guidance["rendered_text_height_ratio"] = 1.0
        return Image.alpha_composite(erased, layer).convert("RGB")

    @staticmethod
    def _expanded_text_box(style: dict[str, Any], box_w: int, box_h: int) -> tuple[int, int, int, int]:
        fg_box = style.get("foreground_bbox") or (0, 0, box_w, box_h)
        fx1, fy1, fx2, fy2 = [int(v) for v in fg_box]
        pad_x = max(2, int(round((fx2 - fx1) * 0.10)))
        pad_y = max(2, int(round((fy2 - fy1) * 0.25)))
        return (
            max(0, fx1 - pad_x),
            max(0, fy1 - pad_y),
            min(box_w, fx2 + pad_x),
            min(box_h, fy2 + pad_y),
        )

    @staticmethod
    def _feathered_cell_alpha(size: tuple[int, int]) -> Image.Image:
        width, height = size
        alpha = Image.new("L", size, 0)
        draw = ImageDraw.Draw(alpha)
        inset = 1 if min(width, height) < 10 else 2
        draw.rectangle((inset, inset, max(inset, width - inset - 1), max(inset, height - inset - 1)), fill=255)
        return alpha.filter(ImageFilter.GaussianBlur(radius=0.7))

    @staticmethod
    def _estimate_character_spans(
        foreground_mask: np.ndarray,
        foreground_bbox: tuple[int, int, int, int],
        num_chars: int,
    ) -> list[tuple[int, int, int, int]]:
        fx1, fy1, fx2, fy2 = foreground_bbox
        submask = foreground_mask[fy1:fy2, fx1:fx2]
        if submask.size == 0 or num_chars <= 0:
            return []
        column_has_ink = submask.sum(axis=0) > max(0, int(round(submask.shape[0] * 0.03)))
        column_has_ink = DiffusersInpaintEditor._close_small_gaps(column_has_ink, max_gap=2)
        runs: list[tuple[int, int]] = []
        start: int | None = None
        for idx, has_ink in enumerate(column_has_ink):
            if has_ink and start is None:
                start = idx
            elif not has_ink and start is not None:
                if idx - start >= 1:
                    runs.append((start, idx))
                start = None
        if start is not None:
            runs.append((start, len(column_has_ink)))
        if len(runs) != num_chars:
            return DiffusersInpaintEditor._equal_character_spans(foreground_bbox, num_chars)
        spans: list[tuple[int, int, int, int]] = []
        for run_x1, run_x2 in runs:
            local = submask[:, run_x1:run_x2]
            ys, xs = np.where(local)
            if xs.size == 0 or ys.size == 0:
                spans.append((fx1 + run_x1, fy1, fx1 + run_x2, fy2))
                continue
            pad = 2
            spans.append(
                (
                    max(0, fx1 + run_x1 + int(xs.min()) - pad),
                    max(0, fy1 + int(ys.min()) - pad),
                    min(foreground_mask.shape[1], fx1 + run_x1 + int(xs.max()) + 1 + pad),
                    min(foreground_mask.shape[0], fy1 + int(ys.max()) + 1 + pad),
                )
            )
        return spans

    @staticmethod
    def _equal_character_spans(
        foreground_bbox: tuple[int, int, int, int],
        num_chars: int,
    ) -> list[tuple[int, int, int, int]]:
        fx1, fy1, fx2, fy2 = foreground_bbox
        width = max(1, fx2 - fx1)
        spans = []
        for idx in range(max(0, num_chars)):
            sx1 = fx1 + int(round(idx * width / num_chars))
            sx2 = fx1 + int(round((idx + 1) * width / num_chars))
            pad = max(1, int(round(width / max(1, num_chars) * 0.10)))
            spans.append((max(0, sx1 - pad), fy1, min(fx2, sx2 + pad), fy2))
        return spans

    @staticmethod
    def _close_small_gaps(values: np.ndarray, max_gap: int) -> np.ndarray:
        closed = np.asarray(values, dtype=bool).copy()
        start: int | None = None
        for idx, value in enumerate(closed):
            if not value and start is None:
                start = idx
            elif value and start is not None:
                if idx - start <= max_gap:
                    closed[start:idx] = True
                start = None
        return closed

    def _text_span_mask(
        self,
        image: Image.Image,
        mask: np.ndarray,
        guidance: dict[str, Any],
    ) -> np.ndarray | None:
        selected = guidance.get("selected_canvas") if isinstance(guidance.get("selected_canvas"), dict) else {}
        original_text = str(selected.get("text") or "").strip()
        signature = str(guidance.get("trap_signature") or "").strip()
        if not original_text or not signature:
            return None

        selected_bbox = selected.get("bbox") if isinstance(selected.get("bbox"), list) else None
        if selected_bbox and len(selected_bbox) == 4:
            x1, y1, x2, y2 = self._expand_bbox(tuple(int(v) for v in selected_bbox), image.size, self.text_bbox_padding_ratio)
        else:
            x1, y1, x2, y2 = bbox_from_mask(mask)
        box_w, box_h = x2 - x1, y2 - y1
        if box_w <= 8 or box_h <= 8:
            return None

        if self.text_replacement_unit == "full" or len(original_text) <= len(signature) + 2:
            return mask_from_bbox(image.size, (x1, y1, x2, y2))

        word_span = self._word_replacement_span(original_text, signature, (x1, y1, x2, y2))
        if word_span is not None:
            px, target_w = word_span
        else:
            if self.text_replacement_unit == "word":
                return mask
            text_chars = max(1, len(original_text.replace(" ", "")))
            sig_chars = max(1, len(signature))
            width_ratio = min(0.88, max(0.34, (sig_chars + 1.5) / (text_chars + 1.0)))
            target_w = min(box_w, max(min(box_w, self.min_mask_width), int(round(box_w * width_ratio))))
            px = self._choose_text_span_x(image, mask, (x1, y1, x2, y2), target_w)
        target_h = box_h
        return mask_from_bbox(image.size, (px, y1, px + target_w, y1 + target_h))

    @staticmethod
    def _expand_bbox(
        bbox: tuple[int, int, int, int],
        image_size: tuple[int, int],
        padding_ratio: float,
    ) -> tuple[int, int, int, int]:
        width, height = image_size
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        # Per-axis padding so wide-thin strips don't get over-extended vertically.
        pad_x = int(round(max(0, bw) * max(0.0, padding_ratio)))
        pad_y = int(round(max(0, bh) * max(0.0, padding_ratio)))
        return (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(width, x2 + pad_x),
            min(height, y2 + pad_y),
        )

    @staticmethod
    def _word_replacement_span(
        original_text: str,
        signature: str,
        bbox: tuple[int, int, int, int],
    ) -> tuple[int, int] | None:
        words = list(re.finditer(r"\S+", original_text))
        if not words:
            return None
        x1, _y1, x2, _y2 = bbox
        box_w = x2 - x1
        if box_w <= 0:
            return None
        if len(words) == 1:
            return x1, box_w

        sig_len = len(signature)
        scored = []
        for idx, match in enumerate(words):
            length = match.end() - match.start()
            score = abs(length - sig_len) + 0.15 * idx
            scored.append((score, idx, match))
        _score, _idx, chosen = min(scored, key=lambda item: item[0])
        text_len = max(1, len(original_text))
        char_w = box_w / text_len
        pad = max(2, int(round(char_w * 0.8)))
        span_x1 = int(round(x1 + chosen.start() * char_w)) - pad
        span_x2 = int(round(x1 + chosen.end() * char_w)) + pad
        span_x1 = max(x1, span_x1)
        span_x2 = min(x2, span_x2)
        if span_x2 <= span_x1:
            return None
        return span_x1, span_x2 - span_x1

    @staticmethod
    def _choose_text_span_x(
        image: Image.Image,
        mask: np.ndarray,
        bbox: tuple[int, int, int, int],
        target_w: int,
    ) -> int:
        x1, y1, x2, y2 = bbox
        if target_w >= x2 - x1:
            return x1
        gray = np.asarray(image.convert("L"), dtype=np.float32)
        region = gray[y1:y2, x1:x2]
        if region.size == 0:
            return x1 + max(0, (x2 - x1 - target_w) // 2)
        blurred = np.asarray(Image.fromarray(region.astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=1.2)), dtype=np.float32)
        edge = np.abs(region - blurred)
        local_mask = mask[y1:y2, x1:x2]
        candidates = np.linspace(x1, max(x1, x2 - target_w), num=9, dtype=int)
        center = (x1 + x2) / 2.0
        best: tuple[float, int] | None = None
        for px in candidates:
            sx1, sx2 = px - x1, px - x1 + target_w
            patch_mask = local_mask[:, sx1:sx2]
            if patch_mask.size == 0 or patch_mask.mean() < 0.55:
                continue
            text_energy = float(edge[:, sx1:sx2][patch_mask].mean()) if patch_mask.any() else 0.0
            center_penalty = abs((px + target_w / 2.0) - center) / max(1.0, x2 - x1)
            score = text_energy - 4.0 * center_penalty
            if best is None or score > best[0]:
                best = (score, int(px))
        if best is None:
            return x1 + max(0, (x2 - x1 - target_w) // 2)
        return best[1]

    def _estimate_text_style(
        self,
        original: Image.Image,
        inpainted: Image.Image,
        mask: np.ndarray,
    ) -> dict[str, Any]:
        x1, y1, x2, y2 = bbox_from_mask(mask)
        original_region = np.asarray(original.convert("RGB"))[y1:y2, x1:x2].astype(np.float32)
        inpaint_region = np.asarray(inpainted.convert("RGB"))[y1:y2, x1:x2].astype(np.float32)
        local_mask = mask[y1:y2, x1:x2]
        if original_region.size == 0 or not local_mask.any():
            return {
                "background_rgb": (128, 128, 128),
                "text_rgb": (30, 30, 30),
                "stroke_rgb": (240, 240, 235),
                "text_opacity": max(self.text_opacity, 0.68),
                "stroke_opacity": max(self.stroke_opacity, 0.12),
                "stroke_scale": 0.8,
                "angle": 0.0,
                "foreground_mask": np.zeros_like(local_mask, dtype=bool),
            }

        pixels = original_region[local_mask]
        inpaint_pixels = inpaint_region[local_mask]
        original_bg = np.median(pixels, axis=0) if pixels.size else np.array([128.0, 128.0, 128.0])
        inpaint_bg = np.median(inpaint_pixels, axis=0) if inpaint_pixels.size else original_bg
        rough_bg = 0.75 * original_bg + 0.25 * inpaint_bg
        bg_luma = self._rgb_luma(rough_bg)
        luma = (
            0.2126 * pixels[:, 0]
            + 0.7152 * pixels[:, 1]
            + 0.0722 * pixels[:, 2]
        )
        color_dist = np.linalg.norm(pixels - rough_bg[None, :], axis=1)
        dist_threshold = max(28.0, float(np.percentile(color_dist, 84)))
        dark = (luma < bg_luma - 18.0) & (color_dist >= dist_threshold)
        light = (luma > bg_luma + 18.0) & (color_dist >= dist_threshold)

        def score(candidate: np.ndarray) -> float:
            if int(candidate.sum()) < 6:
                return 0.0
            frac = float(candidate.mean())
            if frac > 0.48:
                return 0.0
            return float(np.median(np.abs(luma[candidate] - bg_luma)) * np.sqrt(frac))

        dark_score = score(dark)
        light_score = score(light)
        chosen = dark if dark_score >= light_score and dark_score > 0 else light
        if int(chosen.sum()) < 6:
            chosen = color_dist >= max(18.0, float(np.percentile(color_dist, 90)))

        foreground_mask = np.zeros(local_mask.shape, dtype=bool)
        foreground_mask[local_mask] = chosen
        if foreground_mask.any():
            fg_rgb = np.median(original_region[foreground_mask], axis=0)
            bg_candidates = local_mask & ~foreground_mask
            if bg_candidates.sum() >= 8:
                bg_rgb = np.median(original_region[bg_candidates], axis=0)
            else:
                bg_rgb = rough_bg
        else:
            bg_rgb = rough_bg
            fg_rgb = np.array([30.0, 30.0, 30.0]) if bg_luma > 128 else np.array([235.0, 235.0, 228.0])
        bg_luma = self._rgb_luma(bg_rgb)
        fg_luma = float(0.2126 * fg_rgb[0] + 0.7152 * fg_rgb[1] + 0.0722 * fg_rgb[2])
        if abs(fg_luma - bg_luma) < 45:
            fg_rgb = np.array([28.0, 28.0, 28.0]) if bg_luma > 128 else np.array([238.0, 238.0, 230.0])
            fg_luma = float(0.2126 * fg_rgb[0] + 0.7152 * fg_rgb[1] + 0.0722 * fg_rgb[2])
        stroke_rgb = np.array([245.0, 245.0, 238.0]) if fg_luma < bg_luma else np.array([22.0, 22.0, 22.0])
        angle = self._estimate_text_angle(original, mask)
        if foreground_mask.any():
            ys, xs = np.where(foreground_mask)
            foreground_bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        else:
            foreground_bbox = (0, 0, x2 - x1, y2 - y1)
        return {
            "background_rgb": tuple(int(np.clip(v, 0, 255)) for v in bg_rgb),
            "text_rgb": tuple(int(np.clip(v, 0, 255)) for v in fg_rgb),
            "stroke_rgb": tuple(int(np.clip(v, 0, 255)) for v in stroke_rgb),
            "text_opacity": float(np.clip(self.text_opacity, 0.0, 1.0)),
            "stroke_opacity": float(np.clip(self.stroke_opacity, 0.0, 0.24)),
            "stroke_scale": 0.0 if self.stroke_opacity <= 0.04 else (0.35 if abs(fg_luma - bg_luma) >= 70 else 0.6),
            "angle": angle,
            "foreground_fraction": float(foreground_mask.sum() / max(1, local_mask.sum())),
            "foreground_bbox": foreground_bbox,
            "foreground_mask": foreground_mask,
            "estimated_background_luma": float(bg_luma),
            "estimated_text_luma": float(fg_luma),
        }

    def _text_style_metrics(
        self,
        signature: str,
        guidance: dict[str, Any],
        style: dict[str, Any],
    ) -> dict[str, Any]:
        selected = guidance.get("selected_canvas") if isinstance(guidance.get("selected_canvas"), dict) else {}
        original_text = str(selected.get("text") or "")
        original_len = len(re.sub(r"[^A-Za-z0-9]", "", original_text))
        signature_len = len(re.sub(r"[^A-Za-z0-9]", "", signature))
        flags: list[str] = []
        length_ratio = float(signature_len / original_len) if original_len > 0 else 1.0
        if original_len > 0 and length_ratio < self.min_text_length_ratio:
            flags.append("short_signature_for_canvas")
        if original_len > 0 and length_ratio > self.max_text_length_ratio:
            flags.append("long_signature_for_canvas")
        foreground_fraction = float(style.get("foreground_fraction", 0.0))
        if foreground_fraction < self.min_style_foreground_fraction:
            flags.append("uncertain_text_style")
        return {
            "text_original_alnum_length": original_len,
            "text_signature_alnum_length": signature_len,
            "text_length_ratio": length_ratio,
            "text_style_foreground_fraction": foreground_fraction,
            "text_style_background_rgb": style.get("background_rgb"),
            "text_style_text_rgb": style.get("text_rgb"),
            "text_style_foreground_bbox": style.get("foreground_bbox"),
            "text_style_flags": flags,
        }

    @staticmethod
    def _rgb_luma(rgb: np.ndarray) -> float:
        return float(0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2])

    @staticmethod
    def _estimate_text_angle(image: Image.Image, mask: np.ndarray) -> float:
        x1, y1, x2, y2 = bbox_from_mask(mask)
        if x2 - x1 < 16 or y2 - y1 < 8:
            return 0.0
        try:
            import cv2
        except ImportError:
            return 0.0
        gray = np.asarray(image.convert("L"))[y1:y2, x1:x2]
        if gray.size == 0:
            return 0.0
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=12, minLineLength=max(8, (x2 - x1) // 4), maxLineGap=4)
        if lines is None:
            return 0.0
        angles: list[float] = []
        for line in lines[:, 0, :]:
            lx1, ly1, lx2, ly2 = [float(v) for v in line]
            if lx2 == lx1:
                continue
            angle = float(np.degrees(np.arctan2(ly2 - ly1, lx2 - lx1)))
            if -25.0 <= angle <= 25.0:
                angles.append(angle)
        if not angles:
            return 0.0
        return float(np.median(angles))

    def _quality_report(
        self,
        original: Image.Image,
        edited: Image.Image,
        mask: np.ndarray,
        guidance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        bool_mask = ensure_bool_mask(mask, original.size)
        if not bool_mask.any():
            return {
                "quality_local_delta": 0.0,
                "quality_boundary_delta": 0.0,
                "quality_gate_pass": False,
                "quality_flags": ["empty_mask"],
            }
        original_arr = np.asarray(original.convert("RGB"), dtype=np.float32)
        edited_arr = np.asarray(edited.convert("RGB"), dtype=np.float32)
        diff = np.mean(np.abs(original_arr - edited_arr), axis=2) / 255.0
        local_delta = float(diff[bool_mask].mean())
        mask_img = Image.fromarray((bool_mask.astype(np.uint8) * 255), mode="L")
        dilated = np.asarray(mask_img.filter(ImageFilter.MaxFilter(5))) > 0
        eroded = np.asarray(mask_img.filter(ImageFilter.MinFilter(5))) > 0
        boundary = dilated & ~eroded
        boundary_delta = float(diff[boundary].mean()) if boundary.any() else 0.0
        flags: list[str] = list((guidance or {}).get("text_style_flags", []))
        if local_delta > self.max_quality_local_delta:
            flags.append("high_local_delta")
        if boundary_delta > self.max_quality_boundary_delta:
            flags.append("high_boundary_delta")
        return {
            "quality_local_delta": local_delta,
            "quality_boundary_delta": boundary_delta,
            "quality_gate_pass": not flags,
            "quality_flags": flags,
        }

    def _render_text_mutation(self, image: Image.Image, mask: np.ndarray, signature: str) -> Image.Image:
        x1, y1, x2, y2 = bbox_from_mask(mask)
        box_w, box_h = x2 - x1, y2 - y1
        if box_w <= 8 or box_h <= 8:
            return image

        bool_mask = ensure_bool_mask(mask, image.size)
        base_rgb = image.convert("RGB")
        region = np.asarray(base_rgb)[y1:y2, x1:x2]
        background = tuple(int(v) for v in np.median(region.reshape(-1, 3), axis=0))
        luminance = 0.2126 * background[0] + 0.7152 * background[1] + 0.0722 * background[2]
        if luminance < 132:
            text_rgb = (235, 235, 228)
            stroke_rgb = (20, 20, 20)
        else:
            text_rgb = (28, 28, 28)
            stroke_rgb = (245, 245, 235)

        base = base_rgb.convert("RGBA")
        mask_img = Image.fromarray((bool_mask.astype(np.uint8) * 255), mode="L")
        fill_alpha = mask_img.filter(ImageFilter.GaussianBlur(radius=max(0.0, self.mask_feather_radius / 2)))
        fill_alpha = fill_alpha.point(lambda value: int(value * self.text_background_opacity))
        fill_layer = Image.new("RGBA", image.size, (*background, 0))
        fill_layer.putalpha(fill_alpha)
        base = Image.alpha_composite(base, fill_layer)

        scale = 3
        layer = Image.new("RGBA", (base.width * scale, base.height * scale), (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        font = self._fit_font(
            signature,
            box_w * scale,
            box_h * scale,
            min(0.86, max(self.font_height_ratio, 0.68)),
            min(0.94, max(self.font_width_ratio, 0.84)),
        )
        stroke_width = max(1, scale)
        text_bbox = draw.textbbox((0, 0), signature, font=font, stroke_width=stroke_width)
        text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        tx = int((x1 * scale) + max(0, (box_w * scale - text_w) / 2) - text_bbox[0])
        ty = int((y1 * scale) + max(0, (box_h * scale - text_h) / 2) - text_bbox[1])
        draw.text(
            (tx, ty),
            signature,
            font=font,
            fill=(*text_rgb, int(255 * max(self.text_opacity, 0.70))),
            stroke_width=stroke_width,
            stroke_fill=(*stroke_rgb, int(255 * max(self.stroke_opacity, 0.14))),
        )
        if self.text_blur_radius > 0:
            layer = layer.filter(ImageFilter.GaussianBlur(radius=min(self.text_blur_radius, 0.35) * scale))
        layer = layer.resize(base.size, Image.Resampling.LANCZOS)
        alpha = layer.getchannel("A")
        layer.putalpha(Image.composite(alpha, Image.new("L", image.size, 0), mask_img))
        return Image.alpha_composite(base, layer).convert("RGB")

    @staticmethod
    def _is_text_mode(parasitism_mode: str) -> bool:
        mode = parasitism_mode.lower()
        return "text" in mode or "mutation" in mode

    @staticmethod
    def _fit_font(
        text: str,
        width: int,
        height: int,
        height_ratio: float,
        width_ratio: float,
    ) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        max_size = max(8, int(height * height_ratio))
        min_size = 6
        font_path = "DejaVuSans-Bold.ttf"
        for size in range(max_size, min_size - 1, -2):
            try:
                font = ImageFont.truetype(font_path, size)
            except OSError:
                return ImageFont.load_default()
            bbox = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox(
                (0, 0), text, font=font, stroke_width=max(1, size // 18)
            )
            if bbox[2] - bbox[0] <= width * width_ratio and bbox[3] - bbox[1] <= height * height_ratio:
                return font
        try:
            return ImageFont.truetype(font_path, min_size)
        except OSError:
            return ImageFont.load_default()

    def _resize_for_model(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        max_side = max(width, height)
        if max_side <= self.image_size:
            scale = self.image_size / max_side
        else:
            scale = self.image_size / max_side
        new_w = max(64, int(round(width * scale / 8) * 8))
        new_h = max(64, int(round(height * scale / 8) * 8))
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    @staticmethod
    def _resize_mask(mask: np.ndarray, original_size: tuple[int, int], target_size: tuple[int, int]) -> Image.Image:
        bool_mask = ensure_bool_mask(mask, original_size)
        mask_img = Image.fromarray((bool_mask.astype(np.uint8) * 255), mode="L")
        return mask_img.resize(target_size, Image.Resampling.NEAREST)

    @staticmethod
    def _compose_prompt(prompt: str, signature: str, parasitism_mode: str) -> str:
        if "text" in parasitism_mode.lower() or "mutation" in parasitism_mode.lower():
            return (
                f"{prompt}. A natural scene detail with crisp readable text '{signature}', "
                "same font style, same lighting, physically printed, photorealistic."
            )
        return (
            f"{prompt}. A small contextually natural object or label with crisp readable "
            f"text '{signature}', realistic lighting, photorealistic, seamless."
        )


class Flux2KleinInpaintEditor(DiffusersInpaintEditor):
    """Native FLUX.2 Klein inpainting editor without post-hoc text rendering.

    Supports an optional mask-free "free OI" branch backed by ``Flux2KleinPipeline``
    (Kontext-style image-as-context editing) for object_insertion canvases that
    carry a structured Opus proposal. The free pipeline shares VAE / transformer /
    text-encoder weights with the inpaint pipeline by constructing a second
    pipeline from the same component objects, so memory cost is negligible.
    """

    capabilities = {"masked_edit", "diffusers_inpaint", "flux2_klein_inpaint"}

    def __init__(
        self,
        model_name: str = "black-forest-labs/FLUX.2-klein-9B",
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        image_size: int = 768,
        num_inference_steps: int = 40,
        guidance_scale: float = 8.0,
        strength: float = 0.92,
        seed: int = 42,
        enable_attention_slicing: bool = True,
        enable_model_cpu_offload: bool = False,
        enable_sequential_cpu_offload: bool = False,
        max_mask_area_ratio: float = 0.035,
        max_text_mask_area_ratio: float = 0.035,
        max_object_insertion_mask_area_ratio: float = 0.06,
        min_mask_width: int = 96,
        min_mask_height: int = 40,
        label_aspect_ratio: float = 2.6,
        mask_feather_radius: float = 4.0,
        text_span_replacement: bool = True,
        text_replacement_unit: str = "full",
        text_bbox_padding_ratio: float = 0.12,
        max_quality_local_delta: float = 0.42,
        max_quality_boundary_delta: float = 0.22,
        min_text_length_ratio: float = 0.0,
        max_text_length_ratio: float = 999.0,
        min_style_foreground_fraction: float = 0.0,
        max_sequence_length: int = 512,
        enable_free_oi: bool = False,
        free_oi_num_inference_steps: int = 40,
        free_oi_guidance_scale: float = 4.0,
        gradient_guidance: dict[str, Any] | None = None,
    ) -> None:
        try:
            import torch
            from diffusers import Flux2KleinInpaintPipeline
        except ImportError as exc:
            raise ImportError(
                "Flux2KleinInpaintEditor requires diffusers with FLUX.2 support and torch."
            ) from exc

        self.torch = torch
        self.device = torch.device(device)
        self.model_name = model_name
        self.image_size = int(image_size)
        self.num_inference_steps = int(num_inference_steps)
        self.guidance_scale = float(guidance_scale)
        self.strength = float(strength)
        self.seed = int(seed)
        self.max_mask_area_ratio = float(max_mask_area_ratio)
        self.max_text_mask_area_ratio = float(max_text_mask_area_ratio)
        self.max_object_insertion_mask_area_ratio = float(max_object_insertion_mask_area_ratio)
        self.min_mask_width = int(min_mask_width)
        self.min_mask_height = int(min_mask_height)
        self.label_aspect_ratio = float(label_aspect_ratio)
        self.mask_feather_radius = float(mask_feather_radius)
        self.render_signature = False
        self.text_opacity = 1.0
        self.stroke_opacity = 0.0
        self.text_blur_radius = 0.0
        self.text_background_opacity = 1.0
        self.font_height_ratio = 0.62
        self.font_width_ratio = 0.78
        self.natural_text_fusion = False
        self.text_span_replacement = bool(text_span_replacement)
        self.text_replacement_unit = str(text_replacement_unit)
        self.text_bbox_padding_ratio = float(text_bbox_padding_ratio)
        self.max_quality_local_delta = float(max_quality_local_delta)
        self.max_quality_boundary_delta = float(max_quality_boundary_delta)
        self.min_text_length_ratio = float(min_text_length_ratio)
        self.max_text_length_ratio = float(max_text_length_ratio)
        self.min_style_foreground_fraction = float(min_style_foreground_fraction)
        self.glyph_clone_text_fusion = False
        self.max_sequence_length = int(max_sequence_length)
        self.enable_free_oi = bool(enable_free_oi)
        self.free_oi_num_inference_steps = int(free_oi_num_inference_steps)
        self.free_oi_guidance_scale = float(free_oi_guidance_scale)
        self._enable_attention_slicing = bool(enable_attention_slicing)
        self._enable_model_cpu_offload = bool(enable_model_cpu_offload)
        self._enable_sequential_cpu_offload = bool(enable_sequential_cpu_offload)
        grad_cfg = gradient_guidance if isinstance(gradient_guidance, dict) else {}
        self.gradient_guidance_enabled = bool(grad_cfg.get("enabled", False))
        self.gradient_guidance_clip_model = str(grad_cfg.get("clip_model", "ViT-L-14"))
        self.gradient_guidance_clip_pretrained = str(grad_cfg.get("clip_pretrained", "openai"))
        self.gradient_guidance_clip_device = str(grad_cfg.get("clip_device", device))
        self.gradient_guidance_step_size = float(grad_cfg.get("step_size", 0.035))
        self.gradient_guidance_max_update = float(grad_cfg.get("max_update", 0.12))
        self.gradient_guidance_interval = max(1, int(grad_cfg.get("interval", 2)))
        self.gradient_guidance_start_ratio = float(grad_cfg.get("start_ratio", 0.0))
        self.gradient_guidance_stop_ratio = float(grad_cfg.get("stop_ratio", 0.82))
        self.gradient_guidance_text_template = str(
            grad_cfg.get(
                "text_template",
                'a natural image region containing the exact readable text "{signature}"',
            )
        )
        self.capabilities = set(type(self).capabilities)
        if self.gradient_guidance_enabled:
            self.capabilities.add("dual_guidance")

        dtype = getattr(torch, torch_dtype)
        self.pipe = Flux2KleinInpaintPipeline.from_pretrained(model_name, torch_dtype=dtype)
        if self._enable_sequential_cpu_offload or self._enable_model_cpu_offload:
            if self.device.type != "cuda":
                raise ValueError("FLUX.2 CPU offload requires a CUDA device.")
            gpu_id = 0 if self.device.index is None else int(self.device.index)
            if self._enable_sequential_cpu_offload:
                if not hasattr(self.pipe, "enable_sequential_cpu_offload"):
                    raise RuntimeError("Diffusers pipeline does not support sequential CPU offload.")
                self.pipe.enable_sequential_cpu_offload(gpu_id=gpu_id)
            else:
                if not hasattr(self.pipe, "enable_model_cpu_offload"):
                    raise RuntimeError("Diffusers pipeline does not support model CPU offload.")
                self.pipe.enable_model_cpu_offload(gpu_id=gpu_id)
        else:
            self.pipe = self.pipe.to(self.device)
        if enable_attention_slicing and hasattr(self.pipe, "enable_attention_slicing"):
            self.pipe.enable_attention_slicing()
        # Lazy-initialised free-OI pipeline, sharing components with self.pipe.
        self._free_pipe = None
        self._guidance_clip = None
        self._guidance_clip_tokenizer = None
        self._guidance_text_cache: dict[tuple[str, ...], Any] = {}

    def edit(
        self,
        image: Image.Image,
        mask: np.ndarray,
        prompt: str,
        guidance: dict[str, Any],
    ) -> Image.Image:
        original_size = image.size
        original_rgb = image.convert("RGB")
        signature = str(guidance.get("trap_signature", ""))
        parasitism_mode = str(guidance.get("parasitism_mode", ""))
        if self._should_use_free_oi(parasitism_mode, guidance):
            return self._edit_free_oi(original_rgb, mask, prompt, guidance)
        effective_mask = self._refine_mask(original_rgb, mask, parasitism_mode, guidance)
        guidance["effective_mask"] = effective_mask
        guidance["mask_feather_radius"] = self.mask_feather_radius
        work_image = self._resize_for_model(original_rgb)
        work_mask = self._resize_mask(effective_mask, original_size, work_image.size)
        full_prompt = self._compose_flux2_prompt(prompt, signature, parasitism_mode, guidance)
        seed = self.seed + int(guidance.get("edit_attempt", 0)) * 9973
        generator = self.torch.Generator(device=self.device).manual_seed(seed)
        if self.gradient_guidance_enabled and bool(guidance.get("enable_gradient_guidance", True)):
            result = self._edit_gradient_guided_inpaint(
                work_image=work_image,
                work_mask=work_mask,
                prompt=full_prompt,
                guidance=guidance,
                generator=generator,
            )
        else:
            with self.torch.inference_mode():
                result = self.pipe(
                    prompt=full_prompt,
                    image=work_image,
                    mask_image=work_mask,
                    height=work_image.height,
                    width=work_image.width,
                    num_inference_steps=int(guidance.get("num_inference_steps", self.num_inference_steps)),
                    guidance_scale=float(guidance.get("guidance_scale", self.guidance_scale)),
                    strength=float(guidance.get("strength", self.strength)),
                    generator=generator,
                    max_sequence_length=self.max_sequence_length,
                ).images[0]
            guidance["render_strategy"] = "flux2_klein_native_inpaint"
        result = result.resize(original_size, Image.Resampling.LANCZOS)
        quality = self._quality_report(original_rgb, result, effective_mask, guidance)
        guidance.update(quality)
        return result

    def _edit_gradient_guided_inpaint(
        self,
        work_image: Image.Image,
        work_mask: Image.Image,
        prompt: str,
        guidance: dict[str, Any],
        generator: Any,
    ) -> Image.Image:
        pipe = self.pipe
        torch = self.torch
        from diffusers.pipelines.flux2.pipeline_flux2_klein_inpaint import compute_empirical_mu, retrieve_timesteps

        for module in (pipe.transformer, pipe.vae):
            module.requires_grad_(False)
        pipe._guidance_scale = float(guidance.get("guidance_scale", self.guidance_scale))
        pipe._attention_kwargs = None
        pipe._current_timestep = None
        pipe._interrupt = False

        height = work_image.height
        width = work_image.width
        num_inference_steps = int(guidance.get("num_inference_steps", self.num_inference_steps))
        strength = float(guidance.get("strength", self.strength))
        batch_size = 1
        num_images_per_prompt = 1
        device = pipe._execution_device

        init_image = pipe.image_processor.preprocess(
            work_image,
            height,
            width,
            crops_coords=None,
            resize_mode="default",
        )
        with torch.no_grad():
            prompt_embeds, text_ids = pipe.encode_prompt(
                prompt=prompt,
                prompt_embeds=None,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=self.max_sequence_length,
                text_encoder_out_layers=(9, 18, 27),
            )
            if pipe.do_classifier_free_guidance:
                negative_prompt_embeds, negative_text_ids = pipe.encode_prompt(
                    prompt="",
                    prompt_embeds=None,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=self.max_sequence_length,
                    text_encoder_out_layers=(9, 18, 27),
                )
            else:
                negative_prompt_embeds = None
                negative_text_ids = None

        sigmas = None
        if not (hasattr(pipe.scheduler.config, "use_flow_sigmas") and pipe.scheduler.config.use_flow_sigmas):
            sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = (int(height) // pipe.vae_scale_factor // 2) * (int(width) // pipe.vae_scale_factor // 2)
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
        retrieve_timesteps(pipe.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu)
        timesteps, num_inference_steps = pipe.get_timesteps(num_inference_steps, strength, device)
        if num_inference_steps < 1:
            raise ValueError(
                f"After adjusting by strength={strength}, FLUX.2 has {num_inference_steps} inference steps."
            )
        latent_timestep = timesteps[:1].repeat(batch_size * num_images_per_prompt)
        num_channels_latents = pipe.transformer.config.in_channels // 4

        with torch.no_grad():
            latents, noise, image_latents, image_latents_encoded, latent_image_ids = pipe.prepare_latents(
                init_image,
                latent_timestep,
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                None,
            )
            condition_image_latents, condition_image_ids = pipe.prepare_image_latents(
                [image_latents_encoded],
                batch_size * num_images_per_prompt,
                generator,
                device,
                prompt_embeds.dtype,
            )
            mask_condition = pipe.mask_processor.preprocess(
                work_mask,
                height=height,
                width=width,
                resize_mode="default",
                crops_coords=None,
            )
            mask = pipe.prepare_mask_latents(
                mask_condition,
                batch_size,
                num_images_per_prompt,
                height,
                width,
                prompt_embeds.dtype,
                device,
            )

        combined_image_ids = torch.cat([latent_image_ids, condition_image_ids], dim=1)
        total_steps = len(timesteps)
        guidance_count = 0
        first_loss: float | None = None
        last_loss: float | None = None
        last_ret_loss: float | None = None
        last_gen_loss: float | None = None
        masked_blending_enforced = bool(guidance.get("masked_blending_enforced", True))
        guidance_mask = mask if masked_blending_enforced else None

        with pipe.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if pipe.interrupt:
                    continue

                pipe._current_timestep = t
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                with torch.no_grad():
                    latent_model_input = torch.cat([latents, condition_image_latents], dim=1)
                    latent_model_input = latent_model_input.to(pipe.transformer.dtype)
                    with pipe.transformer.cache_context("cond"):
                        noise_pred = pipe.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=None,
                            encoder_hidden_states=prompt_embeds,
                            txt_ids=text_ids,
                            img_ids=combined_image_ids,
                            joint_attention_kwargs=pipe.attention_kwargs,
                            return_dict=False,
                        )[0]
                    noise_pred = noise_pred[:, : latents.size(1)]

                    if pipe.do_classifier_free_guidance:
                        with pipe.transformer.cache_context("uncond"):
                            neg_noise_pred = pipe.transformer(
                                hidden_states=latent_model_input,
                                timestep=timestep / 1000,
                                guidance=None,
                                encoder_hidden_states=negative_prompt_embeds,
                                txt_ids=negative_text_ids,
                                img_ids=combined_image_ids,
                                joint_attention_kwargs=pipe.attention_kwargs,
                                return_dict=False,
                            )[0]
                        neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                        noise_pred = neg_noise_pred + pipe.guidance_scale * (noise_pred - neg_noise_pred)

                    latents_dtype = latents.dtype
                    latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                    init_latents_proper = image_latents
                    if i < total_steps - 1:
                        noise_timestep = timesteps[i + 1].reshape(1).to(device)
                        init_latents_proper = pipe.scheduler.scale_noise(init_latents_proper, noise_timestep, noise)
                    if masked_blending_enforced:
                        latents = (1 - mask) * init_latents_proper + mask * latents

                if self._should_apply_gradient_guidance(i, total_steps):
                    latents, stats = self._apply_flux2_latent_guidance(
                        latents=latents,
                        latent_image_ids=latent_image_ids,
                        mask=guidance_mask,
                        guidance=guidance,
                    )
                    if stats is not None:
                        guidance_count += 1
                        last_loss = stats["loss"]
                        last_ret_loss = stats["ret_loss"]
                        last_gen_loss = stats["gen_loss"]
                        if first_loss is None:
                            first_loss = last_loss
                    if masked_blending_enforced:
                        latents = (1 - mask) * init_latents_proper + mask * latents

                if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                    latents = latents.to(latents_dtype)

                progress_bar.update()

        pipe._current_timestep = None
        with torch.no_grad():
            image_tensor = self._decode_flux2_latents(pipe, latents, latent_image_ids)
            image = pipe.image_processor.postprocess(image_tensor, output_type="pil")[0]

        guidance["render_strategy"] = "flux2_klein_gradient_guided_inpaint"
        guidance["gradient_guidance_steps"] = guidance_count
        if first_loss is not None:
            guidance["gradient_guidance_loss_first"] = first_loss
            guidance["gradient_guidance_loss_last"] = last_loss
            guidance["gradient_guidance_ret_loss_last"] = last_ret_loss
            guidance["gradient_guidance_gen_loss_last"] = last_gen_loss
        return image

    def _should_apply_gradient_guidance(self, step_index: int, total_steps: int) -> bool:
        if step_index % self.gradient_guidance_interval != 0:
            return False
        if total_steps <= 1:
            progress = 0.0
        else:
            progress = step_index / float(total_steps - 1)
        return self.gradient_guidance_start_ratio <= progress <= self.gradient_guidance_stop_ratio

    def _apply_flux2_latent_guidance(
        self,
        latents: Any,
        latent_image_ids: Any,
        mask: Any,
        guidance: dict[str, Any],
    ) -> tuple[Any, dict[str, float] | None]:
        torch = self.torch
        guided_latents = latents.detach().requires_grad_(True)
        image_tensor = self._decode_flux2_latents(self.pipe, guided_latents, latent_image_ids)
        loss, stats = self._gradient_guidance_loss(image_tensor, guidance)
        grad = torch.autograd.grad(loss, guided_latents, retain_graph=False, create_graph=False)[0]
        if not torch.isfinite(grad).all():
            return latents.detach(), None

        if mask is not None:
            grad = grad * mask.to(device=grad.device, dtype=grad.dtype)
        denom = grad.detach().abs().mean(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        normalized_grad = grad / denom
        step_size = float(guidance.get("gradient_step_size", self.gradient_guidance_step_size))
        max_update = float(guidance.get("gradient_max_update", self.gradient_guidance_max_update))
        update = (step_size * normalized_grad).clamp(min=-max_update, max=max_update)
        updated = (guided_latents - update).detach().to(dtype=latents.dtype)
        return updated, stats

    def _decode_flux2_latents(self, pipe: Any, latents: Any, latent_image_ids: Any) -> Any:
        torch = self.torch
        unpacked = pipe._unpack_latents_with_ids(latents, latent_image_ids)
        bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(unpacked.device, unpacked.dtype)
        bn_std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).to(
            unpacked.device,
            unpacked.dtype,
        )
        unpacked = unpacked * bn_std + bn_mean
        unpacked = pipe._unpatchify_latents(unpacked)
        return pipe.vae.decode(unpacked, return_dict=False)[0]

    def _ensure_guidance_clip(self) -> tuple[Any, Any]:
        if self._guidance_clip is not None and self._guidance_clip_tokenizer is not None:
            return self._guidance_clip, self._guidance_clip_tokenizer
        import open_clip

        model, _, _ = open_clip.create_model_and_transforms(
            self.gradient_guidance_clip_model,
            pretrained=self.gradient_guidance_clip_pretrained,
            device=self.gradient_guidance_clip_device,
        )
        model.eval()
        model.requires_grad_(False)
        self._guidance_clip = model
        self._guidance_clip_tokenizer = open_clip.get_tokenizer(self.gradient_guidance_clip_model)
        return self._guidance_clip, self._guidance_clip_tokenizer

    def _gradient_guidance_loss(self, image_tensor: Any, guidance: dict[str, Any]) -> tuple[Any, dict[str, float]]:
        torch = self.torch
        model, tokenizer = self._ensure_guidance_clip()
        clip_device = torch.device(self.gradient_guidance_clip_device)
        pixels = (image_tensor.float() / 2.0 + 0.5).clamp(0.0, 1.0)
        image_size = getattr(getattr(model, "visual", None), "image_size", 224)
        if isinstance(image_size, (tuple, list)):
            image_size = int(image_size[0])
        pixels = torch.nn.functional.interpolate(
            pixels,
            size=(int(image_size), int(image_size)),
            mode="bicubic",
            align_corners=False,
        )
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=pixels.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=pixels.device).view(1, 3, 1, 1)
        pixels = (pixels - mean) / std
        pixels = pixels.to(clip_device)
        image_features = model.encode_image(pixels)
        image_features = torch.nn.functional.normalize(image_features.float(), dim=-1)

        probe_query = str(guidance.get("probe_query") or "")
        signature = str(guidance.get("trap_signature") or "")
        text_prompt = self.gradient_guidance_text_template.format(signature=signature)
        texts = (probe_query, text_prompt)
        if texts not in self._guidance_text_cache:
            tokens = tokenizer(list(texts)).to(clip_device)
            with torch.no_grad():
                text_features = model.encode_text(tokens)
                text_features = torch.nn.functional.normalize(text_features.float(), dim=-1)
            self._guidance_text_cache[texts] = text_features
        text_features = self._guidance_text_cache[texts]

        ret_loss = 1.0 - (image_features * text_features[0:1]).sum(dim=-1).mean()
        gen_loss = 1.0 - (image_features * text_features[1:2]).sum(dim=-1).mean()
        lambda_ret = float(guidance.get("lambda_ret", 2.5))
        lambda_gen = float(guidance.get("lambda_gen", 4.0))
        loss = lambda_ret * ret_loss + lambda_gen * gen_loss
        return loss, {
            "loss": float(loss.detach().cpu()),
            "ret_loss": float(ret_loss.detach().cpu()),
            "gen_loss": float(gen_loss.detach().cpu()),
        }

    def _should_use_free_oi(self, parasitism_mode: str, guidance: dict[str, Any]) -> bool:
        if not self.enable_free_oi:
            return False
        mode = parasitism_mode.lower()
        if "insert" not in mode and "object" not in mode:
            return False
        selected = guidance.get("selected_canvas") if isinstance(guidance.get("selected_canvas"), dict) else {}
        proposal = selected.get("oi_proposal") if isinstance(selected.get("oi_proposal"), dict) else None
        return bool(proposal)

    def _ensure_free_pipe(self):
        if self._free_pipe is not None:
            return self._free_pipe
        from diffusers import Flux2KleinPipeline

        # Construct a Flux2KleinPipeline that *shares* the same in-memory
        # transformer/VAE/text-encoder/tokenizer/scheduler as the inpaint pipe.
        # Avoid `Flux2KleinPipeline.from_pipe(...)` because in this version of
        # diffusers it triggers a `.to(dtype=...)` cast on the shared modules
        # which would allocate a second copy of FLUX.2's 9B weights and OOM
        # an 80 GB GPU.
        free_pipe = Flux2KleinPipeline(
            scheduler=self.pipe.scheduler,
            vae=self.pipe.vae,
            text_encoder=self.pipe.text_encoder,
            tokenizer=self.pipe.tokenizer,
            transformer=self.pipe.transformer,
            is_distilled=getattr(self.pipe, "is_distilled", False),
        )
        if self._enable_attention_slicing and hasattr(free_pipe, "enable_attention_slicing"):
            free_pipe.enable_attention_slicing()
        self._free_pipe = free_pipe
        return self._free_pipe

    @staticmethod
    def _round_to_multiple(x: int, base: int = 16) -> int:
        return max(base, (x // base) * base)

    @staticmethod
    def _build_free_oi_prompt(signature: str, proposal: dict[str, Any]) -> str:
        surface = str(proposal.get("surface_type") or "small printed sign").strip()
        style = str(proposal.get("style_description") or "").strip()
        placement = str(proposal.get("placement_notes") or "").strip()
        parts = [
            f'Edit the image: insert a {surface} that prominently displays the '
            f'exact uppercase text "{signature}" (no extra letters, punctuation, '
            f'or spaces). The text must be crisp, large, and clearly legible.',
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

    def _edit_free_oi(
        self,
        original_rgb: Image.Image,
        mask: np.ndarray,
        prompt: str,
        guidance: dict[str, Any],
    ) -> Image.Image:
        original_size = original_rgb.size
        signature = str(guidance.get("trap_signature", ""))
        selected = guidance.get("selected_canvas") if isinstance(guidance.get("selected_canvas"), dict) else {}
        proposal = selected.get("oi_proposal") if isinstance(selected.get("oi_proposal"), dict) else {}

        free_prompt = self._build_free_oi_prompt(signature, proposal)
        # The user-level trigger_prompt is appended for extra context but kept short.
        trigger = str(prompt or "").strip()
        if trigger and trigger.lower() not in free_prompt.lower():
            free_prompt = f"{free_prompt} Context: {trigger}"

        work_w = self._round_to_multiple(original_size[0], 16)
        work_h = self._round_to_multiple(original_size[1], 16)
        work_image = original_rgb.resize((work_w, work_h), Image.Resampling.LANCZOS)

        free_pipe = self._ensure_free_pipe()
        seed = self.seed + int(guidance.get("edit_attempt", 0)) * 9973
        generator = self.torch.Generator(device=self.device).manual_seed(seed)
        with self.torch.inference_mode():
            result = free_pipe(
                image=work_image,
                prompt=free_prompt,
                height=work_h,
                width=work_w,
                num_inference_steps=int(guidance.get(
                    "free_oi_num_inference_steps", self.free_oi_num_inference_steps
                )),
                guidance_scale=float(guidance.get(
                    "free_oi_guidance_scale", self.free_oi_guidance_scale
                )),
                generator=generator,
                max_sequence_length=self.max_sequence_length,
            ).images[0]
        result = result.resize(original_size, Image.Resampling.LANCZOS)

        # Free OI rewrites large regions, so the bbox-shaped mask is no longer
        # meaningful. We expose a full-image effective mask so the downstream
        # composite step is a no-op and naturalness/readability gates inspect
        # the whole image instead of a small bbox crop.
        full_mask = np.ones((original_size[1], original_size[0]), dtype=bool)
        guidance["effective_mask"] = full_mask
        guidance["mask_feather_radius"] = 0.0
        guidance["masked_blending_enforced"] = False
        guidance["render_strategy"] = "flux2_klein_free_oi"
        guidance["free_oi_prompt"] = free_prompt
        # Quality metrics are computed against the full mask (entire image).
        quality = self._quality_report(original_rgb, result, full_mask, guidance)
        guidance.update(quality)
        return result

    @staticmethod
    def _compose_flux2_prompt(
        prompt: str,
        signature: str,
        parasitism_mode: str,
        guidance: dict[str, Any],
    ) -> str:
        selected = guidance.get("selected_canvas") if isinstance(guidance.get("selected_canvas"), dict) else {}
        original_text = str(selected.get("text") or "").strip()
        if "text" in parasitism_mode.lower() or "mutation" in parasitism_mode.lower():
            source = f' Replace the existing text "{original_text}"' if original_text else " Replace the existing text"
            return (
                f"{prompt}. Inpaint only the masked region.{source} with the exact uppercase text "
                f"\"{signature}\". The final visible text must read exactly \"{signature}\" with no extra "
                "letters, punctuation, or spaces. Preserve the same physical sign, logo, material, background "
                "texture, font style, weight, color, baseline, perspective, lighting, shadows, and edges. "
                "It must look natively printed in the scene, not like a sticker, overlay, or digital watermark."
            )
        # Object insertion: prefer Opus's structured proposal (surface_type, style, placement)
        # over the abstract trigger_prompt to give FLUX a concrete sign description.
        proposal = selected.get("oi_proposal") if isinstance(selected.get("oi_proposal"), dict) else None
        if proposal:
            surface = str(proposal.get("surface_type") or "small printed sign").strip()
            style = str(proposal.get("style_description") or "").strip()
            placement = str(proposal.get("placement_notes") or "").strip()
            bits = [
                f"Inpaint only the masked region. Render a {surface} that displays the exact "
                f"uppercase text \"{signature}\" with no extra letters, punctuation, or spaces. "
                "The text must be crisp, large, and clearly legible, occupying most of the masked "
                "area. Render it as a real physical sign that has always been part of the scene, "
                "not a sticker or watermark overlay."
            ]
            if style:
                bits.append(f"Style: {style}.")
            if placement:
                bits.append(placement.rstrip(".") + ".")
            bits.append(
                "Match the local lighting, perspective, material, and edges of the surrounding "
                "scene. Do not modify anything outside the mask."
            )
            return " ".join(bits)
        return (
            f"{prompt}. Inpaint only the masked region with a small scene-native printed label or object that "
            f"reads exactly \"{signature}\". Match the local material, lighting, perspective, texture, and edges."
        )

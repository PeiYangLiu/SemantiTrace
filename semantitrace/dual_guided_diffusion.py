from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PIL import Image

from semantitrace.models.interfaces import ImageTextEncoder, SemanticEditor, VLMClient
from semantitrace.utils.image import bbox_from_mask, masked_composite_feathered

logger = logging.getLogger(__name__)


class DualGuidedInjector:
    """Paper-level dual-guided injection coordinator.

    The class prepares the retriever/generator guidance context described in the
    paper and delegates the actual semantic edit to a backend. Backends with
    `dual_guidance` capability can consume the gradient parameters directly;
    simpler masked editors still receive the same context and are composited with
    the original image to enforce latent-blending-style spatial confinement.
    """

    def __init__(
        self,
        encoder: ImageTextEncoder,
        vlm: VLMClient,
        editor: SemanticEditor,
        config: dict[str, Any] | None = None,
    ) -> None:
        cfg = config or {}
        self.encoder = encoder
        self.vlm = vlm
        self.editor = editor
        self.lambda_ret = float(cfg.get("lambda_ret", 2.5))
        self.lambda_gen = float(cfg.get("lambda_gen", 4.0))
        self.eta = float(cfg.get("eta", 0.5))
        self.num_ddim_steps = int(cfg.get("num_ddim_steps", 50))
        self.guidance_scale = float(cfg.get("guidance_scale", 7.5))
        self.require_gradient_editor = bool(cfg.get("require_gradient_editor", False))

    def inject(
        self,
        anchor_image: Image.Image,
        mask: np.ndarray,
        trigger_prompt: str,
        probe_query: str,
        trap_signature: str,
        parasitism_mode: str,
        selected_canvas: dict[str, Any] | None = None,
        edit_attempt: int = 0,
    ) -> tuple[Image.Image, dict[str, Any]]:
        if self.require_gradient_editor and "dual_guidance" not in getattr(self.editor, "capabilities", set()):
            raise RuntimeError(
                "Configured require_gradient_editor=true, but editor backend does not expose "
                "'dual_guidance'. Use a FLUX/diffusers editor that implements this capability."
            )

        before_ret = self._retriever_loss(anchor_image, probe_query)
        before_gen = self._generator_loss(anchor_image, probe_query, trap_signature)
        guidance = {
            "probe_query": probe_query,
            "trap_signature": trap_signature,
            "parasitism_mode": parasitism_mode,
            "lambda_ret": self.lambda_ret,
            "lambda_gen": self.lambda_gen,
            "eta": self.eta,
            "num_ddim_steps": self.num_ddim_steps,
            "guidance_scale": self.guidance_scale,
            "edit_attempt": int(edit_attempt),
        }
        if selected_canvas is not None:
            guidance["selected_canvas"] = selected_canvas
        edited_raw = self.editor.edit(anchor_image, mask, trigger_prompt, guidance)
        effective_mask = guidance.get("effective_mask", mask)
        feather_radius = float(guidance.get("mask_feather_radius", 0.0))
        edited = masked_composite_feathered(anchor_image, edited_raw, effective_mask, feather_radius)

        after_ret = self._retriever_loss(edited, probe_query)
        after_gen = self._generator_loss(edited, probe_query, trap_signature)
        metrics = {
            "l_ret_before": before_ret,
            "l_ret_after": after_ret,
            "l_gen_before": before_gen,
            "l_gen_after": after_gen,
            "objective_before": self.lambda_ret * before_ret + self.lambda_gen * before_gen,
            "objective_after": self.lambda_ret * after_ret + self.lambda_gen * after_gen,
            "editor_capabilities": sorted(getattr(self.editor, "capabilities", set())),
            "masked_blending_enforced": bool(guidance.get("masked_blending_enforced", True)),
            "effective_mask_bbox": list(bbox_from_mask(effective_mask)),
            "effective_mask_area": int(np.asarray(effective_mask, dtype=bool).sum()),
            "effective_mask_area_ratio": float(
                np.asarray(effective_mask, dtype=bool).sum()
                / (anchor_image.size[0] * anchor_image.size[1])
            ),
        }
        for key in (
            "render_strategy",
            "estimated_text_angle",
            "quality_local_delta",
            "quality_boundary_delta",
            "quality_gate_pass",
            "quality_flags",
            "text_original_alnum_length",
            "text_signature_alnum_length",
            "text_length_ratio",
            "text_style_foreground_fraction",
            "text_style_background_rgb",
            "text_style_text_rgb",
            "text_style_foreground_bbox",
            "text_style_flags",
            "glyph_clone_text_fusion",
            "rendered_text_width_ratio",
            "rendered_text_height_ratio",
            "free_oi_prompt",
            "gradient_guidance_steps",
            "gradient_guidance_loss_first",
            "gradient_guidance_loss_last",
            "gradient_guidance_ret_loss_last",
            "gradient_guidance_gen_loss_last",
        ):
            if key in guidance:
                metrics[key] = guidance[key]
        logger.info(
            "Injected canary mode=%s Lret %.4f->%.4f Lgen %.4f->%.4f",
            parasitism_mode,
            before_ret,
            after_ret,
            before_gen,
            after_gen,
        )
        return edited, metrics

    def _retriever_loss(self, image: Image.Image, query: str) -> float:
        img_emb = self.encoder.encode_images([image])[0]
        txt_emb = self.encoder.encode_texts([query])[0]
        sim = float(np.dot(img_emb, txt_emb))
        return 1.0 - sim

    def _generator_loss(self, image: Image.Image, probe_query: str, signature: str) -> float:
        try:
            score = float(self.vlm.score_text(image, probe_query, signature))
        except Exception:
            score = 0.0
        return -score

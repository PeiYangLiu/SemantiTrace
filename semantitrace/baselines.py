from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class BaselineOutput:
    image: Image.Image
    trap_signature: str
    trigger_prompt: str


class PGDBaseline:
    """Bounded pixel perturbation baseline.

    This lightweight implementation supplies the L_inf-constrained perturbation
    surface used in experiments. A gradient-enabled variant can replace the
    random sign noise with CLIP/VLM gradients while keeping the same API.
    """

    def __init__(self, epsilon: float = 8 / 255, seed: int = 42) -> None:
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

    def apply(self, image: Image.Image, signature: str = "PGD") -> BaselineOutput:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        noise = self.rng.choice([-self.epsilon, self.epsilon], size=arr.shape)
        edited = np.clip(arr + noise, 0.0, 1.0)
        out = Image.fromarray((edited * 255).round().astype(np.uint8))
        return BaselineOutput(out, signature, f"L_inf PGD-style perturbation for {signature}")


class AQUABaseline:
    def __init__(self, size: tuple[int, int] = (512, 512), seed: int = 42) -> None:
        self.size = size
        self.rng = random.Random(seed)

    def acronym(self, signature: str | None = None) -> BaselineOutput:
        signature = signature or self._signature()
        image = Image.new("RGB", self.size, (0, 0, 0))
        draw = ImageDraw.Draw(image)
        font = self._font(80)
        bbox = draw.textbbox((0, 0), signature, font=font)
        x = (self.size[0] - (bbox[2] - bbox[0])) // 2
        y = (self.size[1] - (bbox[3] - bbox[1])) // 2
        draw.text((x, y), signature, fill=(255, 255, 255), font=font)
        return BaselineOutput(image, signature, f"synthetic black image with acronym {signature}")

    def spatial(self, signature: str | None = None) -> BaselineOutput:
        signature = signature or self._signature()
        image = Image.new("RGB", self.size, (210, 230, 255))
        draw = ImageDraw.Draw(image)
        draw.rectangle((80, 260, 430, 390), fill=(120, 80, 50))
        draw.ellipse((180, 70, 330, 220), fill=(180, 110, 80), outline=(20, 20, 20), width=4)
        draw.ellipse((230, 0, 285, 65), fill=(230, 20, 20), outline=(20, 20, 20), width=4)
        draw.text((205, 405), signature, fill=(0, 0, 0), font=self._font(42))
        return BaselineOutput(image, signature, f"synthetic spatial hallucination carrying {signature}")

    def _signature(self) -> str:
        alphabet = "BCDFGHJKLMNPQRSTVWXYZ"
        return "".join(self.rng.choice(alphabet) for _ in range(3))

    @staticmethod
    def _font(size: int):
        try:
            return ImageFont.truetype("DejaVuSans-Bold.ttf", size)
        except OSError:
            return ImageFont.load_default()


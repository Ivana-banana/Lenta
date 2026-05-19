from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from .utils import normalize_text, rotate_image


@dataclass
class TextBlock:
    text: str
    confidence: float = 0.0
    bbox: list[list[float]] | None = None
    engine: str = "none"
    angle: int = 0
    roi_name: str | None = None

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "confidence": float(self.confidence),
            "bbox": self.bbox,
            "engine": self.engine,
            "angle": self.angle,
            "roi_name": self.roi_name,
        }


class OCREngine:
    def __init__(self, preferred: str | None = None, languages: list[str] | None = None) -> None:
        self.preferred = (preferred or os.getenv("LENTA_OCR_ENGINE") or "auto").lower()
        self.languages = languages or ["ru", "en"]
        self.engine_name = "none"
        self.engine: Any = None
        self._init_engine()

    def _init_engine(self) -> None:
        choices = [self.preferred] if self.preferred != "auto" else ["rapidocr", "paddleocr", "easyocr", "tesseract"]
        for choice in choices:
            try:
                if choice == "rapidocr":
                    from rapidocr_onnxruntime import RapidOCR

                    self.engine = RapidOCR()
                    self.engine_name = "rapidocr"
                    return
                if choice == "paddleocr":
                    from paddleocr import PaddleOCR

                    self.engine = PaddleOCR(use_angle_cls=True, lang="ru", show_log=False)
                    self.engine_name = "paddleocr"
                    return
                if choice == "easyocr":
                    import easyocr

                    self.engine = easyocr.Reader(self.languages, gpu=False, verbose=False)
                    self.engine_name = "easyocr"
                    return
                if choice == "tesseract":
                    import pytesseract

                    self.engine = pytesseract
                    self.engine_name = "tesseract"
                    return
            except Exception:
                self.engine = None
                self.engine_name = "none"

    def recognize_text(
        self,
        image: np.ndarray,
        roi_name: str | None = None,
        try_rotations: bool = False,
        angles: list[int] | None = None,
        upscale: bool = True,
    ) -> list[dict]:
        if image is None or image.size == 0 or self.engine is None:
            return []

        angles = angles or ([0, 270, 90, 180] if try_rotations else [0])
        all_blocks: list[TextBlock] = []
        for angle in angles:
            variant = rotate_image(image, angle)
            for preprocessed in self._preprocess_variants(variant, upscale=upscale, roi_name=roi_name):
                blocks = self._recognize_single(preprocessed, angle=angle, roi_name=roi_name)
                all_blocks.extend(blocks)
                if blocks and not try_rotations:
                    break

        # Deduplicate by normalized text, keeping the highest confidence.
        best: dict[str, TextBlock] = {}
        for block in all_blocks:
            key = normalize_text(block.text).lower()
            if not key:
                continue
            if key not in best or block.confidence > best[key].confidence:
                best[key] = block
        return [b.to_dict() for b in sorted(best.values(), key=lambda x: x.confidence, reverse=True)]

    def _preprocess_variants(self, image: np.ndarray, upscale: bool = True, roi_name: str | None = None):
        yield image
        if upscale and max(image.shape[:2]) < 900:
            up2 = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            yield up2
        if roi_name not in {"barcode", "id_sku"}:
            return

        variants = [image]
        if max(image.shape[:2]) < 1200:
            variants.append(cv2.resize(image, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC))
        for variant in variants:
            gray = cv2.cvtColor(variant, cv2.COLOR_BGR2GRAY) if variant.ndim == 3 else variant
            clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)
            sharpened = cv2.filter2D(clahe, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
            for prepared in [clahe, sharpened]:
                yield cv2.cvtColor(prepared, cv2.COLOR_GRAY2BGR)
                if min(prepared.shape[:2]) >= 24:
                    block = max(15, (min(prepared.shape[:2]) // 12) * 2 + 1)
                    threshold = cv2.adaptiveThreshold(
                        prepared,
                        255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY,
                        block,
                        7,
                    )
                    yield cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR)

    def _recognize_single(self, image: np.ndarray, angle: int, roi_name: str | None) -> list[TextBlock]:
        try:
            if self.engine_name == "rapidocr":
                result, _ = self.engine(image)
                blocks = []
                for item in result or []:
                    bbox, text, conf = item[0], item[1], item[2]
                    blocks.append(TextBlock(str(text), float(conf), bbox, self.engine_name, angle, roi_name))
                return blocks
            if self.engine_name == "paddleocr":
                result = self.engine.ocr(image, cls=True)
                blocks = []
                for line_group in result or []:
                    for line in line_group or []:
                        bbox, rec = line[0], line[1]
                        blocks.append(TextBlock(str(rec[0]), float(rec[1]), bbox, self.engine_name, angle, roi_name))
                return blocks
            if self.engine_name == "easyocr":
                result = self.engine.readtext(image)
                return [TextBlock(str(text), float(conf), bbox, self.engine_name, angle, roi_name) for bbox, text, conf in result]
            if self.engine_name == "tesseract":
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
                text = self.engine.image_to_string(rgb, lang="rus+eng")
                return [TextBlock(text, 0.5, None, self.engine_name, angle, roi_name)] if text.strip() else []
        except Exception:
            return []
        return []

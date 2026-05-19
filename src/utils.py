from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logger(name: str = "lenta_price_tags", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def to_builtin(value):
    if isinstance(value, dict):
        return {str(k): to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [to_builtin(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def save_json(path: str | Path, payload) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(to_builtin(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def write_image(path: str | Path, image: np.ndarray, quality: int = 90) -> bool:
    path = Path(path)
    ensure_dir(path.parent)
    suffix = path.suffix.lower() or ".jpg"
    params = []
    if suffix in {".jpg", ".jpeg"}:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, encoded = cv2.imencode(suffix, image, params)
    if not ok:
        return False
    path.write_bytes(encoded.tobytes())
    return True


def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def parse_float(value, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(str(value).replace(",", "."))
    except ValueError:
        return default


def bbox_area(bbox: Iterable[float]) -> float:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def clip_bbox(bbox: Iterable[float], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return [x1, y1, x2, y2]


def crop_bbox(image: np.ndarray, bbox: Iterable[float], pad: int = 0) -> tuple[np.ndarray, list[int]]:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    clipped = clip_bbox([x1 - pad, y1 - pad, x2 + pad, y2 + pad], w, h)
    cx1, cy1, cx2, cy2 = clipped
    return image[cy1:cy2, cx1:cx2].copy(), clipped


def iou(box_a: Iterable[float], box_b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = bbox_area(box_a) + bbox_area(box_b) - inter
    return 0.0 if union <= 0 else inter / union


def nms(candidates: list[dict], iou_threshold: float = 0.35, limit: int | None = None) -> list[dict]:
    ordered = sorted(candidates, key=lambda c: float(c.get("confidence", 0.0)), reverse=True)
    kept: list[dict] = []
    for cand in ordered:
        if all(iou(cand["bbox"], prev["bbox"]) < iou_threshold for prev in kept):
            kept.append(cand)
            if limit and len(kept) >= limit:
                break
    return kept


def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    angle = angle % 360
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def variance_of_laplacian(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def normalized_roi_to_bbox(roi: Iterable[float], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = [float(v) for v in roi]
    return clip_bbox([x1 * width, y1 * height, x2 * width, y2 * height], width, height)


def safe_filename(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "item"


def quality_score(record: dict) -> float:
    important = [
        "product_name",
        "price_default",
        "price_card",
        "barcode",
        "id_sku",
        "qr_code_barcode",
        "print_datetime",
    ]
    filled = sum(bool(str(record.get(k, "")).strip()) and str(record.get(k, "")).strip().lower() != "nan" for k in important)
    barcode_bonus = 2.5 if record.get("barcode") else 0.0
    qr_bonus = 2.0 if record.get("qr_code_barcode") else 0.0
    sharpness = min(float(record.get("sharpness", 0.0) or 0.0) / 1000.0, 3.0)
    area_px = max(1.0, bbox_area([record.get("x_min", 0), record.get("y_min", 0), record.get("x_max", 0), record.get("y_max", 0)]))
    # For GT matching, an oversized crop can be worse than a tighter one even
    # if OCR sees more text. Keep a mild area reward, then penalize huge boxes.
    area_score = min(math.sqrt(area_px) / 350.0, 1.5)
    if area_px > 180_000:
        area_score -= min(1.0, (area_px - 180_000) / 300_000.0)
    det_confidence = min(float(record.get("_det_confidence", 0.0) or 0.0), 1.0)
    return filled + barcode_bonus + qr_bonus + sharpness + area_score + det_confidence * 0.6

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

from .config import DEFAULT_TEMPLATES_PATH


def color_ratios(image: np.ndarray) -> dict[str, float]:
    if image is None or image.size == 0:
        return {"red": 0.0, "yellow": 0.0, "green": 0.0, "white": 0.0, "other": 1.0}
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    total = float(hsv.shape[0] * hsv.shape[1])
    masks = {
        "red": cv2.bitwise_or(
            cv2.inRange(hsv, np.array((0, 45, 50)), np.array((16, 255, 255))),
            cv2.inRange(hsv, np.array((165, 45, 50)), np.array((179, 255, 255))),
        ),
        "yellow": cv2.inRange(hsv, np.array((16, 35, 70)), np.array((40, 255, 255))),
        "green": cv2.inRange(hsv, np.array((40, 30, 45)), np.array((95, 255, 255))),
        "white": cv2.inRange(hsv, np.array((0, 0, 155)), np.array((179, 85, 255))),
    }
    ratios = {name: float(cv2.countNonZero(mask)) / total for name, mask in masks.items()}
    ratios["other"] = max(0.0, 1.0 - sum(ratios.values()))
    return ratios


def dominant_color(image: np.ndarray) -> str:
    ratios = color_ratios(image)
    if ratios["red"] + ratios["yellow"] > 0.16:
        return "red" if ratios["red"] >= ratios["yellow"] * 0.75 else "yellow"
    if ratios["green"] > 0.18:
        return "green"
    if ratios["white"] > 0.35:
        return "white"
    return "other"


def orientation_from_aspect(aspect: float) -> str:
    if 0.82 <= aspect <= 1.22:
        return "square"
    if aspect > 1.22:
        return "horizontal"
    return "vertical"


class TemplateClassifier:
    def __init__(self, templates_path: str | Path = DEFAULT_TEMPLATES_PATH, template_images_dir: str | Path | None = None) -> None:
        self.templates_path = Path(templates_path)
        with self.templates_path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        self.templates = payload.get("templates", [])
        self.template_images_dir = Path(template_images_dir) if template_images_dir else self.templates_path.parent / "template_images"
        self._image_features = self._load_template_image_features()

    def classify(self, crop: np.ndarray, rough_text: str = "", qr_fields: dict | None = None, has_qr: bool = False) -> dict:
        h, w = crop.shape[:2]
        aspect = w / max(1, h)
        color = dominant_color(crop)
        orientation = orientation_from_aspect(aspect)
        text_lower = rough_text.lower()
        qr_fields = qr_fields or {}
        qr_has_wholesale = any(qr_fields.get(k) for k in ["wholesale_level_1_count", "wholesale_level_1_price", "wholesale_level_2_count", "wholesale_level_2_price"])
        qr_has_action = bool(qr_fields.get("action_price_qr") or qr_fields.get("action_code_qr"))
        feature_match = self._match_template_image(crop)

        best_template = None
        best_score = -1.0
        reasons: list[str] = []

        for template in self.templates:
            score = 0.0
            current_reasons = []
            lo, hi = template.get("aspect_ratio", [0.0, 999.0])
            if lo <= aspect <= hi:
                score += 0.22
                current_reasons.append("aspect")
            else:
                center = (lo + hi) / 2.0
                score += max(0.0, 0.12 - abs(aspect - center) * 0.04)
            if orientation == template.get("orientation"):
                score += 0.12
                current_reasons.append("orientation")
            if color in template.get("expected_colors", []):
                score += 0.22
                current_reasons.append("color")
            elif color == "red" and template.get("mechanic") in {"action", "sale", "threshold_from", "threshold_to", "BOGOF"}:
                score += 0.12
            if bool(template.get("has_qr", True)) == bool(has_qr):
                score += 0.08
            for keyword in template.get("keywords", []):
                kw = str(keyword).lower()
                if kw and kw in text_lower:
                    score += 0.08
                    current_reasons.append(f"kw:{keyword}")
            if qr_has_wholesale and template.get("mechanic") in {"threshold_from", "threshold_to"}:
                score += 0.18
                current_reasons.append("qr_wholesale")
            if qr_has_action and template.get("mechanic") in {"action", "sale"}:
                score += 0.15
                current_reasons.append("qr_action")
            if feature_match and feature_match["template_id"] == template.get("template_id"):
                score += min(0.18, feature_match["score"])
                current_reasons.append("orb")
            score = min(score, 1.0)
            if score > best_score:
                best_score = score
                best_template = template
                reasons = current_reasons

        if not best_template:
            best_template = {
                "template_id": "unknown_generic",
                "family": "unknown",
                "mechanic": "unknown",
                "orientation": orientation,
                "unit": "unknown",
                "roi": {},
            }
            best_score = 0.0

        confidence = round(float(best_score), 3)
        return {
            "template_id": best_template.get("template_id", "unknown_generic"),
            "family": best_template.get("family", "unknown"),
            "mechanic": best_template.get("mechanic", "unknown"),
            "orientation": best_template.get("orientation", orientation),
            "unit": best_template.get("unit", "unknown"),
            "confidence": confidence,
            "dominant_color": color,
            "aspect_ratio": round(float(aspect), 3),
            "has_qr": bool(has_qr),
            "reasons": reasons,
            "roi": best_template.get("roi", {}),
            "use_generic_fallback": confidence < 0.42,
        }

    def _load_template_image_features(self) -> dict:
        features = {}
        if not self.template_images_dir.exists():
            return features
        orb = cv2.ORB_create(nfeatures=500)
        for path in self.template_images_dir.glob("*.*"):
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            keypoints, descriptors = orb.detectAndCompute(image, None)
            if descriptors is not None and len(keypoints) >= 8:
                features[path.stem] = (keypoints, descriptors)
        return features

    def _match_template_image(self, crop: np.ndarray) -> dict | None:
        if not self._image_features:
            return None
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        orb = cv2.ORB_create(nfeatures=500)
        _, descriptors = orb.detectAndCompute(gray, None)
        if descriptors is None:
            return None
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        best = None
        for template_id, (_, tmpl_desc) in self._image_features.items():
            matches = matcher.match(descriptors, tmpl_desc)
            good = [m for m in matches if m.distance < 55]
            score = min(0.18, len(good) / 80.0)
            if best is None or score > best["score"]:
                best = {"template_id": template_id, "score": score}
        return best


from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import cv2
import numpy as np

from .utils import clip_bbox, nms


class Detector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[dict]:
        """Return price-tag candidates with bbox/source/confidence."""


class YOLODetector(Detector):
    """Optional local detector trained outside the baseline.

    The class imports ultralytics lazily so the baseline remains lightweight and
    works when YOLO dependencies are not installed.
    """

    def __init__(self, model_path: str, conf_threshold: float = 0.25, imgsz: int = 1280, max_candidates: int = 80):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.max_candidates = max_candidates

    def detect(self, frame: np.ndarray) -> list[dict]:
        results = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf_threshold, verbose=False)
        candidates: list[dict] = []
        for result in results or []:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
            confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
            for box, conf in zip(xyxy, confs):
                x1, y1, x2, y2 = [int(round(float(v))) for v in box]
                candidates.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "source": "yolo",
                        "confidence": float(conf),
                    }
                )
        return sorted(candidates, key=lambda item: float(item.get("confidence", 0.0)), reverse=True)[: self.max_candidates]


class OpenCVDetector(Detector):
    def __init__(
        self,
        max_dim: int = 1280,
        nms_iou: float = 0.32,
        max_candidates: int = 45,
        min_area_ratio: float = 0.00035,
        max_area_ratio: float = 0.045,
    ) -> None:
        self.max_dim = max_dim
        self.nms_iou = nms_iou
        self.max_candidates = max_candidates
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.qr_detector = cv2.QRCodeDetector()

    def detect(self, frame: np.ndarray) -> list[dict]:
        h, w = frame.shape[:2]
        scale = min(1.0, self.max_dim / max(w, h))
        small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else frame
        inv_scale = 1.0 / scale

        candidates: list[dict] = []
        candidates.extend(self._detect_qr_candidates(small, inv_scale, w, h))
        candidates.extend(self._detect_price_panel_candidates(small, inv_scale, w, h))
        color_candidates = self._detect_color_candidates(small, inv_scale, w, h)
        candidates.extend(color_candidates)
        candidates.extend(self._merge_aligned_color_candidates(color_candidates, w, h))
        candidates.extend(self._detect_contour_candidates(small, inv_scale, w, h))

        candidates = self._rescore_candidates(frame, candidates)
        return nms(candidates, iou_threshold=self.nms_iou, limit=self.max_candidates)

    def _to_original(self, bbox: Iterable[float], inv_scale: float, width: int, height: int) -> list[int]:
        x1, y1, x2, y2 = bbox
        return clip_bbox([x1 * inv_scale, y1 * inv_scale, x2 * inv_scale, y2 * inv_scale], width, height)

    def _detect_qr_candidates(self, image: np.ndarray, inv_scale: float, width: int, height: int) -> list[dict]:
        candidates: list[dict] = []
        point_sets = []
        try:
            retval, _, points, _ = self.qr_detector.detectAndDecodeMulti(image)
            if retval and points is not None:
                point_sets.extend(points)
        except Exception:
            pass
        try:
            retval, points = self.qr_detector.detectMulti(image)
            if retval and points is not None:
                point_sets.extend(points)
        except Exception:
            pass

        for pts in point_sets:
            pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
            x, y, bw, bh = cv2.boundingRect(pts)
            side = max(bw, bh, 1)
            # QR is often in the right part of a horizontal 6x6/6x12 tag.
            expansions = [
                [x - 3.2 * side, y - 0.9 * side, x + 0.8 * side, y + 1.2 * side],
                [x - 1.4 * side, y - 1.4 * side, x + 1.4 * side, y + 1.4 * side],
                [x - 0.8 * side, y - 1.2 * side, x + 3.2 * side, y + 0.9 * side],
            ]
            for box in expansions:
                candidates.append(
                    {
                        "bbox": self._to_original(box, inv_scale, width, height),
                        "source": "qr",
                        "confidence": 0.92,
                    }
                )
        return candidates

    def _merge_aligned_color_candidates(self, candidates: list[dict], width: int, height: int) -> list[dict]:
        price_like = [
            c
            for c in candidates
            if str(c.get("source", "")) in {"color_red", "color_orange_yellow"}
        ]
        merged: list[dict] = []
        used_pairs: set[tuple[int, int]] = set()
        for i, first in enumerate(price_like):
            fx1, fy1, fx2, fy2 = first["bbox"]
            fw = max(1, fx2 - fx1)
            fh = max(1, fy2 - fy1)
            for j, second in enumerate(price_like[i + 1 :], start=i + 1):
                sx1, sy1, sx2, sy2 = second["bbox"]
                sw = max(1, sx2 - sx1)
                sh = max(1, sy2 - sy1)
                overlap_x = max(0, min(fx2, sx2) - max(fx1, sx1)) / max(1, min(fw, sw))
                overlap_y = max(0, min(fy2, sy2) - max(fy1, sy1)) / max(1, min(fh, sh))
                cx_close = abs(((fx1 + fx2) / 2) - ((sx1 + sx2) / 2)) < max(fw, sw) * 0.9
                cy_close = abs(((fy1 + fy2) / 2) - ((sy1 + sy2) / 2)) < max(fh, sh) * 1.8
                vertical_gap = max(0, max(fy1, sy1) - min(fy2, sy2))
                horizontal_gap = max(0, max(fx1, sx1) - min(fx2, sx2))
                vertically_stacked = overlap_x > 0.28 and cy_close and vertical_gap < max(fh, sh) * 0.28
                horizontally_stacked = overlap_y > 0.28 and cx_close and horizontal_gap < max(fw, sw) * 0.28
                if not (vertically_stacked or horizontally_stacked):
                    continue
                pair = (i, j)
                if pair in used_pairs:
                    continue
                used_pairs.add(pair)
                x1, y1, x2, y2 = min(fx1, sx1), min(fy1, sy1), max(fx2, sx2), max(fy2, sy2)
                bw = x2 - x1
                bh = y2 - y1
                # Price panels are often only the colored half of a tag. Expand
                # asymmetrically to catch the white QR/name half too.
                pad_x = int(bw * 0.55)
                pad_y = int(bh * 0.28)
                merged_box = clip_bbox([x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y], width, height)
                mx1, my1, mx2, my2 = merged_box
                merged_area = max(0, mx2 - mx1) * max(0, my2 - my1)
                merged_aspect = (mx2 - mx1) / max(1, my2 - my1)
                if merged_area > width * height * 0.045:
                    continue
                if merged_aspect < 0.25 or merged_aspect > 4.0:
                    continue
                merged.append(
                    {
                        "bbox": merged_box,
                        "source": "color_merged",
                        "confidence": max(0.58, min(0.76, max(float(first.get("confidence", 0)), float(second.get("confidence", 0))) - 0.04)),
                    }
                )
        return merged

    def _color_masks(self, hsv: np.ndarray) -> list[tuple[str, np.ndarray]]:
        ranges = {
            "red": [((0, 45, 55), (15, 255, 255)), ((165, 45, 55), (179, 255, 255))],
            "orange_yellow": [((10, 35, 75), (38, 255, 255))],
            "green": [((38, 30, 45), (92, 255, 255))],
            "white": [((0, 0, 160), (179, 85, 255))],
        }
        masks: list[tuple[str, np.ndarray]] = []
        for name, bounds in ranges.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lo, hi in bounds:
                mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.array(lo), np.array(hi)))
            masks.append((name, mask))
        return masks

    def _detect_price_panel_candidates(self, image: np.ndarray, inv_scale: float, width: int, height: int) -> list[dict]:
        """Detect tight candidates from small warm price panels.

        In the robot videos many tags are rotated vertically on shelf rails.
        A normal color contour can merge several adjacent orange/red panels
        into one tall strip, which is bad for the bbox/time fallback. This pass
        intentionally uses light morphology and emits tight variants around
        each individual colored price panel.
        """
        ih, iw = image.shape[:2]
        frame_area = ih * iw
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        masks = [
            (
                "red",
                cv2.bitwise_or(
                    cv2.inRange(hsv, np.array((0, 45, 55)), np.array((15, 255, 255))),
                    cv2.inRange(hsv, np.array((165, 45, 55)), np.array((179, 255, 255))),
                ),
            ),
            ("orange_yellow", cv2.inRange(hsv, np.array((10, 35, 75)), np.array((38, 255, 255)))),
        ]

        candidates: list[dict] = []
        seen: set[tuple[int, int, int, int]] = set()
        variants = [
            (-0.25, -0.25, 1.20, 0.25),
            (-0.35, -0.45, 1.60, 0.45),
            (-0.80, -0.15, 0.50, 0.15),
            (0.00, -0.35, 1.30, 0.35),
        ]
        for color_name, base_mask in masks:
            mask_versions = [base_mask]
            close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
            open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            closed = cv2.morphologyEx(base_mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
            closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel, iterations=1)
            mask_versions.append(closed)

            for mask in mask_versions:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    x, y, bw, bh = cv2.boundingRect(contour)
                    area = bw * bh
                    if area < max(70, frame_area * 0.00005) or area > frame_area * 0.020:
                        continue
                    if bw < 5 or bh < 7 or bw > iw * 0.20 or bh > ih * 0.48:
                        continue
                    aspect = bw / max(1, bh)
                    if aspect < 0.08 or aspect > 8.0:
                        continue
                    component = mask[y : y + bh, x : x + bw]
                    fill_ratio = cv2.countNonZero(component) / max(1.0, float(area))
                    if fill_ratio < 0.14:
                        continue
                    for variant_idx, (lx, ty, rx, by) in enumerate(variants):
                        box = [
                            max(0, int(round(x + bw * lx))),
                            max(0, int(round(y + bh * ty))),
                            min(iw, int(round(x + bw + bw * rx))),
                            min(ih, int(round(y + bh + bh * by))),
                        ]
                        expanded_w = box[2] - box[0]
                        expanded_h = box[3] - box[1]
                        if expanded_w < 18 or expanded_h < 18:
                            continue
                        expanded_area = expanded_w * expanded_h
                        if expanded_area > frame_area * 0.035:
                            continue
                        expanded_aspect = expanded_w / max(1, expanded_h)
                        if expanded_aspect < 0.16 or expanded_aspect > 5.6:
                            continue
                        key = tuple(box)
                        if key in seen:
                            continue
                        seen.add(key)
                        patch_score = self._tag_color_patch_score(hsv[box[1] : box[3], box[0] : box[2]], color_name)
                        if patch_score <= 0:
                            continue
                        confidence = min(0.88, 0.60 + patch_score * 0.18 + min(0.10, fill_ratio * 0.12))
                        candidates.append(
                            {
                                "bbox": self._to_original(box, inv_scale, width, height),
                                "source": f"color_price_panel_{color_name}_{variant_idx}",
                                "confidence": confidence,
                            }
                        )
        candidates.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
        return candidates[:180]

    def _detect_color_candidates(self, image: np.ndarray, inv_scale: float, width: int, height: int) -> list[dict]:
        ih, iw = image.shape[:2]
        frame_area = ih * iw
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        candidates: list[dict] = []

        for name, mask in self._color_masks(hsv):
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, bw, bh = cv2.boundingRect(contour)
                area = bw * bh
                ar = bw / max(1, bh)
                if area < frame_area * self.min_area_ratio or area > frame_area * self.max_area_ratio:
                    continue
                if bw < 20 or bh < 18 or ar < 0.16 or ar > 6.5:
                    continue
                if name == "white" and area > frame_area * 0.025:
                    continue
                pad_x = int(bw * (0.45 if name != "white" else 0.18))
                pad_y = int(bh * (0.28 if name != "white" else 0.14))
                box = [x - pad_x, y - pad_y, x + bw + pad_x, y + bh + pad_y]
                box = [
                    max(0, int(box[0])),
                    max(0, int(box[1])),
                    min(iw, int(box[2])),
                    min(ih, int(box[3])),
                ]
                expanded_w = max(0, box[2] - box[0])
                expanded_h = max(0, box[3] - box[1])
                expanded_area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
                if expanded_area > frame_area * 0.035:
                    continue
                if expanded_w > iw * 0.24 or expanded_h > ih * 0.30:
                    continue
                patch_score = self._tag_color_patch_score(hsv[box[1] : box[3], box[0] : box[2]], name)
                if patch_score <= 0:
                    continue
                area_ratio = expanded_area / frame_area
                size_score = 1.0 if 0.002 <= area_ratio <= 0.018 else 0.65
                aspect_score = 1.0 if 0.45 <= (expanded_w / max(expanded_h, 1)) <= 3.2 else 0.7
                base = {"red": 0.70, "orange_yellow": 0.64, "green": 0.46, "white": 0.38}[name]
                confidence = min(0.9, base * 0.55 + patch_score * 0.30 + size_score * 0.10 + aspect_score * 0.05)
                if box[0] <= 2 or box[1] <= 2 or box[2] >= iw - 2 or box[3] >= ih - 2:
                    confidence *= 0.82
                candidates.append(
                    {
                        "bbox": self._to_original(box, inv_scale, width, height),
                        "source": f"color_{name}",
                        "confidence": confidence,
                    }
                )
                if name in {"red", "orange_yellow"}:
                    expansion_specs = [
                        (-0.10, -0.72, 2.00, 1.15),
                        (-1.00, -0.18, 1.15, 1.70),
                    ]
                    for spec_idx, (lx, ty, rx, by) in enumerate(expansion_specs):
                        alt_box = [
                            max(0, int(x + bw * lx)),
                            max(0, int(y + bh * ty)),
                            min(iw, int(x + bw * rx)),
                            min(ih, int(y + bh * by)),
                        ]
                        alt_w = max(0, alt_box[2] - alt_box[0])
                        alt_h = max(0, alt_box[3] - alt_box[1])
                        alt_area = alt_w * alt_h
                        if alt_area <= 0 or alt_area > frame_area * 0.04:
                            continue
                        if alt_w < 28 or alt_h < 22 or alt_w > iw * 0.28 or alt_h > ih * 0.34:
                            continue
                        alt_aspect = alt_w / max(1, alt_h)
                        if alt_aspect < 0.22 or alt_aspect > 5.2:
                            continue
                        edge_penalty = 0.82 if alt_box[0] <= 2 or alt_box[1] <= 2 or alt_box[2] >= iw - 2 or alt_box[3] >= ih - 2 else 1.0
                        candidates.append(
                            {
                                "bbox": self._to_original(alt_box, inv_scale, width, height),
                                "source": f"color_{name}_expanded_{spec_idx}",
                                "confidence": max(0.50, min(0.70, confidence - 0.06) * edge_penalty),
                            }
                        )
        return candidates

    def _tag_color_patch_score(self, hsv_patch: np.ndarray, source_color: str) -> float:
        if hsv_patch is None or hsv_patch.size == 0:
            return 0.0
        total = float(hsv_patch.shape[0] * hsv_patch.shape[1])
        white = cv2.inRange(hsv_patch, np.array((0, 0, 155)), np.array((179, 90, 255)))
        warm1 = cv2.inRange(hsv_patch, np.array((0, 40, 55)), np.array((38, 255, 255)))
        warm2 = cv2.inRange(hsv_patch, np.array((165, 40, 55)), np.array((179, 255, 255)))
        green = cv2.inRange(hsv_patch, np.array((38, 30, 45)), np.array((95, 255, 255)))
        white_ratio = cv2.countNonZero(white) / total
        warm_ratio = (cv2.countNonZero(warm1) + cv2.countNonZero(warm2)) / total
        green_ratio = cv2.countNonZero(green) / total
        if source_color in {"red", "orange_yellow"}:
            if white_ratio < 0.05 or warm_ratio < 0.06:
                return 0.0
            return min(1.0, white_ratio * 1.7 + warm_ratio * 1.2)
        if source_color == "green":
            if white_ratio < 0.08 or green_ratio < 0.08:
                return 0.0
            return min(0.85, white_ratio * 1.2 + green_ratio)
        if source_color == "white":
            if white_ratio < 0.28 or (warm_ratio + green_ratio) < 0.03:
                return 0.0
            return min(0.7, white_ratio)
        return 0.1

    def _rescore_candidates(self, frame: np.ndarray, candidates: list[dict]) -> list[dict]:
        h, w = frame.shape[:2]
        frame_area = max(1, h * w)
        rescored: list[dict] = []
        for cand in candidates:
            x1, y1, x2, y2 = cand["bbox"]
            patch = frame[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            ph, pw = patch.shape[:2]
            area = max(1, pw * ph)
            aspect = pw / max(1, ph)
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            total = float(area)
            red = cv2.bitwise_or(
                cv2.inRange(hsv, np.array((0, 45, 50)), np.array((16, 255, 255))),
                cv2.inRange(hsv, np.array((165, 45, 50)), np.array((179, 255, 255))),
            )
            yellow = cv2.inRange(hsv, np.array((16, 35, 70)), np.array((40, 255, 255)))
            green = cv2.inRange(hsv, np.array((40, 30, 45)), np.array((95, 255, 255)))
            white = cv2.inRange(hsv, np.array((0, 0, 155)), np.array((179, 85, 255)))
            edges = cv2.Canny(gray, 70, 180)
            red_ratio = cv2.countNonZero(red) / total
            yellow_ratio = cv2.countNonZero(yellow) / total
            green_ratio = cv2.countNonZero(green) / total
            white_ratio = cv2.countNonZero(white) / total
            edge_ratio = cv2.countNonZero(edges) / total
            area_ratio = area / frame_area

            visual = 0.0
            if 0.35 <= aspect <= 1.45:
                visual += 0.24
            elif 0.22 <= aspect <= 2.2:
                visual += 0.10
            if 0.0025 <= area_ratio <= 0.025:
                visual += 0.16
            elif 0.001 <= area_ratio <= 0.04:
                visual += 0.07
            if red_ratio > 0.16:
                visual += 0.20
            elif red_ratio > 0.08:
                visual += 0.10
            if green_ratio > 0.14:
                visual += 0.12
            if yellow_ratio > 0.06 and white_ratio > 0.08:
                visual += 0.08
            if 0.10 <= white_ratio <= 0.58:
                visual += 0.13
            if 0.035 <= edge_ratio <= 0.18:
                visual += 0.09
            if aspect > 3.2 or area_ratio > 0.05:
                visual -= 0.16
            if str(cand.get("source", "")).startswith(("contour", "color_white")):
                visual -= 0.08

            old_conf = float(cand.get("confidence", 0.0) or 0.0)
            new_conf = max(0.01, min(0.95, old_conf * 0.58 + visual * 0.62))
            updated = dict(cand)
            updated["confidence"] = new_conf
            updated["_visual_score"] = round(float(visual), 4)
            rescored.append(updated)
        return rescored

    def _detect_contour_candidates(self, image: np.ndarray, inv_scale: float, width: int, height: int) -> list[dict]:
        ih, iw = image.shape[:2]
        frame_area = ih * iw
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, 70, 180)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[dict] = []

        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            area = bw * bh
            ar = bw / max(1, bh)
            if area < frame_area * (self.min_area_ratio * 1.2) or area > frame_area * 0.045:
                continue
            if bw < 28 or bh < 22 or ar < 0.2 or ar > 5.0:
                continue
            extent = cv2.contourArea(contour) / max(1.0, float(area))
            if extent < 0.08:
                continue
            pad_x = int(bw * 0.15)
            pad_y = int(bh * 0.12)
            box = [x - pad_x, y - pad_y, x + bw + pad_x, y + bh + pad_y]
            candidates.append(
                {
                    "bbox": self._to_original(box, inv_scale, width, height),
                    "source": "contour",
                    "confidence": min(0.55, 0.32 + extent),
                }
            )
        return candidates

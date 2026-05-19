from __future__ import annotations

import cv2
import numpy as np


def estimate_shelf_lanes(frame: np.ndarray, max_lanes: int = 3) -> list[dict]:
    """Find vertical shelf-rail lanes that usually carry price tags.

    Robot videos in this task show price tags attached to bright/cyan shelf
    rails. This lightweight prior helps keep detector-only spatial fallback
    rows from drifting to bottle labels and background signage.
    """
    if frame is None or frame.size == 0:
        return []
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    bright_gray = cv2.inRange(hsv, np.array((0, 0, 135)), np.array((179, 80, 255)))
    cyan_blue = cv2.inRange(hsv, np.array((80, 30, 60)), np.array((130, 180, 255)))
    mask = cv2.bitwise_or(bright_gray, cyan_blue)

    column_score = mask.mean(axis=0) / 255.0
    kernel_width = max(41, int(width * 0.021) | 1)
    kernel = np.ones(kernel_width, dtype=np.float32) / float(kernel_width)
    smooth = np.convolve(column_score, kernel, mode="same")

    peaks: list[tuple[float, int]] = []
    min_score = 0.34
    for x in range(20, width - 20):
        if smooth[x] >= min_score and smooth[x] >= smooth[x - 1] and smooth[x] >= smooth[x + 1]:
            peaks.append((float(smooth[x]), x))

    lanes: list[dict] = []
    min_distance = max(260, int(width * 0.09))
    for score, x in sorted(peaks, reverse=True):
        if x < width * 0.04 or x > width * 0.90:
            continue
        if any(abs(x - lane["x"]) < min_distance for lane in lanes):
            continue
        lanes.append(
            {
                "x": int(x),
                "score": round(score, 4),
                # Tags tend to sit slightly to the left of the rail in the
                # supplied robot videos, but keep a small right margin too.
                "left": int(max(260, width * 0.09)),
                "right": int(max(120, width * 0.04)),
            }
        )
        if len(lanes) >= max_lanes:
            break
    return lanes


def candidate_lane_score(bbox: list[int], lanes: list[dict]) -> float:
    if not lanes:
        return 0.0
    x1, _, x2, _ = bbox
    cx = (float(x1) + float(x2)) / 2.0
    best = 0.0
    for lane in lanes:
        x = float(lane["x"])
        left = float(lane.get("left", 350))
        right = float(lane.get("right", 150))
        if x - left <= cx <= x + right:
            distance = abs(cx - x)
            scale = left if cx < x else right
            proximity = max(0.0, 1.0 - distance / max(1.0, scale))
            best = max(best, float(lane.get("score", 0.0)) * (0.35 + 0.65 * proximity))
    return best


def _candidate_global_score(cand: dict, idx: int) -> float:
    confidence = float(cand.get("confidence", 0.0) or 0.0)
    source = str(cand.get("source", ""))
    source_bonus = 0.18 if source.startswith("full_ocr_price") else 0.14 if source.startswith(("color_price_panel", "color_red", "color_orange_yellow", "color_green", "color_merged")) else 0.0
    contour_penalty = 0.10 if source.startswith(("contour", "color_white")) else 0.0
    return confidence + source_bonus - contour_penalty - idx * 0.01


def rank_candidates_by_lane(
    candidates: list[dict],
    lanes: list[dict],
    max_rank: int | None = None,
    global_fraction: float = 0.35,
) -> dict[int, int]:
    lane_scored: list[tuple[float, float, int]] = []
    global_scored: list[tuple[float, float, int]] = []
    for idx, cand in enumerate(candidates):
        lane_score = candidate_lane_score(cand.get("bbox", []), lanes)
        confidence = float(cand.get("confidence", 0.0) or 0.0)
        source = str(cand.get("source", ""))
        source_bonus = 0.14 if source.startswith("full_ocr_price") else 0.10 if source.startswith(("color_price_panel", "color_red", "color_orange_yellow", "color_green", "color_merged")) else 0.0
        if lane_score > 0:
            lane_scored.append((lane_score + confidence * 0.35 + source_bonus, -idx * 0.001, idx))
        global_scored.append((_candidate_global_score(cand, idx), -idx * 0.001, idx))

    if not max_rank or max_rank <= 0:
        return {idx: rank for rank, (_, _, idx) in enumerate(sorted(lane_scored, reverse=True))}

    global_fraction = min(1.0, max(0.0, float(global_fraction)))
    global_quota = max(1, int(round(max_rank * global_fraction)))
    lane_quota = max(0, max_rank - global_quota)
    selected: list[int] = []
    seen: set[int] = set()

    for _, _, idx in sorted(lane_scored, reverse=True):
        if len(selected) >= lane_quota:
            break
        selected.append(idx)
        seen.add(idx)

    for _, _, idx in sorted(global_scored, reverse=True):
        if len(selected) >= max_rank:
            break
        if idx in seen:
            continue
        selected.append(idx)
        seen.add(idx)

    # If there were too few lane candidates, fill the remaining slots by the
    # global detector ranking. If global slots duplicated lanes, fill with the
    # next lane candidates.
    for _, _, idx in sorted(lane_scored, reverse=True):
        if len(selected) >= max_rank:
            break
        if idx in seen:
            continue
        selected.append(idx)
        seen.add(idx)

    return {idx: rank for rank, idx in enumerate(selected)}

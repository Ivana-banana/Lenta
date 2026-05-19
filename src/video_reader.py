from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2

from .utils import ensure_dir, variance_of_laplacian, write_image


@dataclass
class FrameSample:
    image: object
    frame_index: int
    timestamp_ms: int
    sharpness: float


def get_video_metadata(video_path: str | Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    duration_sec = frame_count / fps if fps else 0.0
    return {
        "filename": Path(video_path).name,
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration_sec,
    }


def iter_video_frames(
    video_path: str | Path,
    sample_fps: float = 2.0,
    min_sharpness: float = 0.0,
    debug_dir: str | Path | None = None,
    save_frames: bool = False,
    max_frames: int | None = None,
) -> Iterator[FrameSample]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 20.0)
    step = max(1, int(round(fps / max(sample_fps, 0.01))))
    frame_idx = 0
    yielded = 0
    debug_path = ensure_dir(debug_dir) if debug_dir and save_frames else None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % step == 0:
            sharpness = variance_of_laplacian(frame)
            if sharpness >= min_sharpness:
                timestamp_ms = int(round(frame_idx * 1000.0 / fps))
                if debug_path is not None:
                    out = debug_path / f"frame_{frame_idx:06d}_ts_{timestamp_ms:08d}_sharp_{sharpness:.0f}.jpg"
                    write_image(out, frame, quality=90)
                yielded += 1
                yield FrameSample(frame, frame_idx, timestamp_ms, sharpness)
                if max_frames is not None and yielded >= max_frames:
                    break
        frame_idx += 1

    cap.release()

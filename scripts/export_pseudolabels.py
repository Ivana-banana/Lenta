from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import OpenCVDetector
from src.ocr_engine import OCREngine
from src.pipeline import full_frame_text_candidates
from src.shelf_lanes import estimate_shelf_lanes, rank_candidates_by_lane
from src.video_reader import iter_video_frames


def yolo_line(bbox: list[int], width: int, height: int, class_id: int = 0) -> str:
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def export_video(
    video_path: Path,
    out_dir: Path,
    sample_fps: float,
    max_candidates_per_frame: int,
    full_frame_ocr_candidates_per_frame: int,
    min_confidence: float,
) -> list[dict]:
    images_dir = out_dir / "images"
    labels_dir = out_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    detector = OpenCVDetector()
    ocr = OCREngine() if full_frame_ocr_candidates_per_frame > 0 else None
    manifest: list[dict] = []

    for frame_sample in iter_video_frames(video_path, sample_fps=sample_fps):
        frame = frame_sample.image
        height, width = frame.shape[:2]
        candidates = detector.detect(frame)
        lanes = estimate_shelf_lanes(frame)

        if ocr is not None and ocr.engine_name != "none":
            blocks = ocr.recognize_text(frame, try_rotations=False, angles=[0], upscale=False)
            candidates.extend(full_frame_text_candidates(frame, blocks, limit=full_frame_ocr_candidates_per_frame))

        ranks = rank_candidates_by_lane(candidates, lanes, max_rank=max_candidates_per_frame, global_fraction=0.35)
        selected = []
        for idx, cand in enumerate(candidates):
            if ranks.get(idx, 999999) >= max_candidates_per_frame:
                continue
            if float(cand.get("confidence", 0.0) or 0.0) < min_confidence:
                continue
            x1, y1, x2, y2 = [int(v) for v in cand["bbox"]]
            if x2 <= x1 or y2 <= y1:
                continue
            selected.append((idx, cand))

        if not selected:
            continue

        stem = f"{video_path.stem}_{frame_sample.frame_index:06d}_{frame_sample.timestamp_ms:08d}"
        image_path = images_dir / f"{stem}.jpg"
        label_path = labels_dir / f"{stem}.txt"
        cv2.imwrite(str(image_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        label_path.write_text(
            "\n".join(yolo_line(cand["bbox"], width, height) for _, cand in selected),
            encoding="utf-8",
        )

        for rank, (idx, cand) in enumerate(selected):
            manifest.append(
                {
                    "video": video_path.name,
                    "frame_index": frame_sample.frame_index,
                    "frame_timestamp": frame_sample.timestamp_ms,
                    "image": str(image_path),
                    "label": str(label_path),
                    "rank": rank,
                    "det_idx": idx,
                    "source": cand.get("source", ""),
                    "confidence": cand.get("confidence", 0.0),
                    "bbox": cand.get("bbox", []),
                }
            )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Export automatic pseudo-labels for local detector training")
    parser.add_argument("--videos", nargs="+", required=True, help="Input video paths")
    parser.add_argument("--out-dir", default="outputs/pseudolabels", help="Output dataset directory")
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--max-candidates-per-frame", type=int, default=20)
    parser.add_argument("--full-frame-ocr-candidates-per-frame", type=int, default=12)
    parser.add_argument("--min-confidence", type=float, default=0.50)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    all_rows: list[dict] = []
    for video in args.videos:
        all_rows.extend(
            export_video(
                Path(video),
                out_dir=out_dir,
                sample_fps=args.sample_fps,
                max_candidates_per_frame=args.max_candidates_per_frame,
                full_frame_ocr_candidates_per_frame=args.full_frame_ocr_candidates_per_frame,
                min_confidence=args.min_confidence,
            )
        )

    manifest_path = out_dir / "manifest.csv"
    pd.DataFrame(all_rows).to_csv(manifest_path, index=False, encoding="utf-8")
    print(f"Saved pseudo-label manifest: {manifest_path}")
    print(f"labels: {len(all_rows)}")


if __name__ == "__main__":
    main()

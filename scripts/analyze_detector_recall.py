from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.detector import OpenCVDetector
from src.utils import iou
from src.video_reader import iter_video_frames


def parse_float_column(series: pd.Series) -> pd.Series:
    return series.astype(str).str.replace(",", ".", regex=False).replace("", "nan").astype(float)


def load_reference(path: str | Path) -> pd.DataFrame:
    ref = pd.read_csv(path, dtype=str).fillna("")
    for col in ["frame_timestamp", "x_min", "y_min", "x_max", "y_max"]:
        ref[col] = parse_float_column(ref[col])
    return ref


def analyze(video: str | Path, ref_csv: str | Path, sample_fps: float, time_tolerance_ms: float, iou_threshold: float) -> pd.DataFrame:
    ref = load_reference(ref_csv)
    detector = OpenCVDetector()
    rows = []
    for frame in iter_video_frames(video, sample_fps=sample_fps):
        candidates = detector.detect(frame.image)
        refs = ref[(ref["frame_timestamp"] - frame.timestamp_ms).abs() <= time_tolerance_ms]
        for _, row in refs.iterrows():
            ref_box = [row.x_min, row.y_min, row.x_max, row.y_max]
            best = (0.0, 9999, "")
            for rank, candidate in enumerate(candidates):
                score = iou(ref_box, candidate["bbox"])
                if score > best[0]:
                    best = (score, rank, str(candidate.get("source", "")))
            rows.append(
                {
                    "frame_timestamp": frame.timestamp_ms,
                    "product_name": str(row.get("product_name", ""))[:80],
                    "best_iou": best[0],
                    "best_rank": best[1],
                    "best_source": best[2],
                    "matched": best[0] >= iou_threshold,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze detector candidate recall against example CSV.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--time-tolerance-ms", type=float, default=1500.0)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    df = analyze(args.video, args.ref, args.sample_fps, args.time_tolerance_ms, args.iou_threshold)
    if df.empty:
        print("No reference rows near sampled frames.")
        return
    matched = df[df["matched"]]
    print(f"reference rows near sampled frames: {len(df)}")
    print(f"IoU >= {args.iou_threshold}: {len(matched)}")
    for top_n in [5, 10, 12, 20, 30, 45]:
        count = int(((df["best_iou"] >= args.iou_threshold) & (df["best_rank"] < top_n)).sum())
        print(f"top {top_n}: {count}")
    print("")
    print(matched.sort_values("best_iou", ascending=False).head(20).to_string(index=False))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()

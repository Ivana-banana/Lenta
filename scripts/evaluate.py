from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


KEY_FIELDS = ["barcode", "qr_code_barcode", "id_sku"]
NO_VALUES = {"", "nan", "none", "нет", "РЅРµС‚"}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "wholesale_level_1_coun" in df.columns and "wholesale_level_1_count" not in df.columns:
        df = df.rename(columns={"wholesale_level_1_coun": "wholesale_level_1_count"})
    return df


def clean_value(value) -> str:
    text = str(value or "").strip()
    if text.lower() in NO_VALUES:
        return ""
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def clean_series(series: pd.Series) -> set[str]:
    values = series.dropna().astype(str).str.replace(r"\.0$", "", regex=True).str.strip()
    return {v for v in values if clean_value(v)}


def parse_number(value, default: float = 0.0) -> float:
    try:
        return float(str(value).strip().replace(",", "."))
    except Exception:
        return default


def bbox_from_row(row: pd.Series) -> list[float]:
    return [parse_number(row.get(col, 0.0)) for col in ["x_min", "y_min", "x_max", "y_max"]]


def iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def fill_rate(df: pd.DataFrame) -> dict[str, float]:
    rates = {}
    for col in df.columns:
        values = df[col].fillna("").astype(str).str.strip()
        rates[col] = round(float(values.apply(lambda value: bool(clean_value(value))).mean()), 3)
    return rates


def match_rows(pred: pd.DataFrame, ref: pd.DataFrame, time_tolerance_ms: float, iou_threshold: float) -> dict:
    matched_pred: set[int] = set()
    matched_ref: set[int] = set()
    barcode_matches = []

    ref_by_barcode: dict[str, list[int]] = {}
    if "barcode" in ref.columns:
        for ridx, row in ref.iterrows():
            barcode = clean_value(row.get("barcode", ""))
            if barcode:
                ref_by_barcode.setdefault(barcode, []).append(int(ridx))

    if "barcode" in pred.columns:
        for pidx, row in pred.iterrows():
            barcode = clean_value(row.get("barcode", ""))
            if not barcode:
                continue
            for ridx in ref_by_barcode.get(barcode, []):
                if ridx not in matched_ref:
                    matched_pred.add(int(pidx))
                    matched_ref.add(ridx)
                    barcode_matches.append((int(pidx), ridx, barcode))
                    break

    spatial_candidates = []
    for pidx, prow in pred.iterrows():
        pidx = int(pidx)
        if pidx in matched_pred:
            continue
        pts = parse_number(prow.get("frame_timestamp", 0.0))
        pbox = bbox_from_row(prow)
        for ridx, rrow in ref.iterrows():
            ridx = int(ridx)
            if ridx in matched_ref:
                continue
            rts = parse_number(rrow.get("frame_timestamp", 0.0))
            if abs(pts - rts) > time_tolerance_ms:
                continue
            score = iou(pbox, bbox_from_row(rrow))
            if score >= iou_threshold:
                spatial_candidates.append((score, pidx, ridx))

    spatial_matches = []
    for score, pidx, ridx in sorted(spatial_candidates, reverse=True):
        if pidx in matched_pred or ridx in matched_ref:
            continue
        matched_pred.add(pidx)
        matched_ref.add(ridx)
        spatial_matches.append((pidx, ridx, score))

    return {
        "barcode_matches": barcode_matches,
        "spatial_matches": spatial_matches,
        "matched_pred": matched_pred,
        "matched_ref": matched_ref,
    }


def evaluate(
    pred_csv: str | Path,
    reference_csv: str | Path,
    out: str | Path,
    time_tolerance_ms: float = 1500.0,
    iou_threshold: float = 0.3,
) -> Path:
    pred = normalize_columns(pd.read_csv(pred_csv, dtype=str).fillna(""))
    ref = normalize_columns(pd.read_csv(reference_csv, dtype=str).fillna(""))
    matching = match_rows(pred, ref, time_tolerance_ms=time_tolerance_ms, iou_threshold=iou_threshold)
    lines = [
        "# Evaluation Report",
        "",
        f"Prediction: `{pred_csv}`",
        f"Reference: `{reference_csv}`",
        "",
        f"- predicted rows: {len(pred)}",
        f"- reference rows: {len(ref)}",
        f"- barcode primary matches: {len(matching['barcode_matches'])}",
        f"- spatial fallback matches: {len(matching['spatial_matches'])} (time_tolerance_ms={time_tolerance_ms}, iou_threshold={iou_threshold})",
        f"- total matched rows: {len(matching['matched_pred'])} / pred {len(pred)} / ref {len(ref)}",
    ]
    for field in KEY_FIELDS:
        if field in pred.columns and field in ref.columns:
            p = clean_series(pred[field])
            r = clean_series(ref[field])
            inter = p & r
            lines.append(f"- {field} intersection: {len(inter)} / pred {len(p)} / ref {len(r)}")
    lines.append("")
    lines.append("## Prediction Fill Rate")
    for col, rate in fill_rate(pred).items():
        lines.append(f"- {col}: {rate}")
    lines.append("")
    lines.append("## Reference Fill Rate")
    for col, rate in fill_rate(ref).items():
        lines.append(f"- {col}: {rate}")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--out", default="outputs/evaluation_report.md")
    parser.add_argument("--time-tolerance-ms", type=float, default=1500.0)
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    args = parser.parse_args()
    print(evaluate(args.pred, args.ref, args.out, time_tolerance_ms=args.time_tolerance_ms, iou_threshold=args.iou_threshold))


if __name__ == "__main__":
    main()

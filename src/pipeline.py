from __future__ import annotations

import argparse
import math
import re
import traceback
from pathlib import Path

import cv2
import pandas as pd

from .config import CSV_COLUMNS, DEBUG_DIR, NO_VALUE, OUTPUT_DIR, QR_COLUMNS
from .csv_writer import records_to_dataframe, write_csv
from .deduplication import deduplicate
from .detector import OpenCVDetector, YOLODetector
from .field_parser import blocks_to_text, parse_fields
from .ocr_engine import OCREngine
from .qr_parser import QRParser
from .shelf_lanes import candidate_lane_score, estimate_shelf_lanes, rank_candidates_by_lane
from .template_classifier import TemplateClassifier
from .utils import bbox_area, crop_bbox, ensure_dir, nms, normalized_roi_to_bbox, rotate_image, safe_filename, save_json, setup_logger, write_image
from .video_reader import get_video_metadata, iter_video_frames


LOGGER = setup_logger()


def has_price_tag_evidence(record: dict) -> bool:
    identity_fields = [
        "barcode",
        "id_sku",
        "qr_code_barcode",
    ]
    for field in identity_fields:
        value = str(record.get(field, "") or "").strip().lower()
        if value and value not in {"nan", "нет"}:
            return True
    product_name = str(record.get("product_name", "") or "").strip()
    has_price = False
    for field in ["price_default", "price_card", "price_discount"]:
        value = str(record.get(field, "") or "").strip().lower()
        if value and value not in {"nan", "нет"}:
            has_price = True
    if product_name and has_price:
        return True
    discount = str(record.get("discount_amount", "") or "").strip().lower()
    if discount and discount not in {"nan", "нет"} and (product_name or has_price):
        return True
    code = str(record.get("code", "") or "").strip().lower()
    if product_name and code and code not in {"nan", "нет"}:
        return True
    return False


def has_rough_text_signal(text: str) -> bool:
    text = (text or "").strip()
    if not text:
        return False
    digits = sum(ch.isdigit() for ch in text)
    letters = sum(ch.isalpha() for ch in text)
    return digits >= 2 or letters >= 4 or "%" in text


def should_run_roi_ocr(template_info: dict, rough_text: str, mode: str, max_zones: int) -> bool:
    if max_zones <= 0 or mode == "off" or not template_info.get("roi"):
        return False
    if mode == "always":
        return True
    confidence = float(template_info.get("confidence", 0.0) or 0.0)
    return confidence >= 0.45 and has_rough_text_signal(rough_text)


def should_keep_spatial_fallback(
    cand: dict,
    bbox: list[int],
    frame_shape: tuple[int, int, int],
    det_idx: int,
    spatial_rank: int | None = None,
    lane_score: float = 0.0,
) -> bool:
    source = str(cand.get("source", ""))
    if not source.startswith(("color_price_panel", "color_red", "color_orange_yellow", "color_green", "color_merged", "full_ocr_price")):
        return False
    confidence = float(cand.get("confidence", 0.0) or 0.0)
    if confidence < 0.76 and not source.startswith("color_merged"):
        return False
    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    frame_h, frame_w = frame_shape[:2]
    area = bbox_area(bbox)
    if area < frame_w * frame_h * 0.0009 or area > frame_w * frame_h * 0.045:
        return False
    aspect = width / max(1, height)
    if aspect < 0.18 or aspect > 5.8:
        return False
    if source.startswith("full_ocr_price"):
        return True
    if spatial_rank is not None:
        return True
    return det_idx < 18


def make_spatial_fallback_record(video_name: str, frame_sample, bbox: list[int], cand: dict, color: str) -> dict:
    x1, y1, x2, y2 = bbox
    record = {col: "" for col in CSV_COLUMNS}
    record.update(
        {
            "filename": video_name,
            "price_discount": NO_VALUE,
            "discount_amount": NO_VALUE,
            "code": NO_VALUE,
            "additional_info": NO_VALUE,
            "special_symbols": NO_VALUE,
            "color": color,
            "frame_timestamp": frame_sample.timestamp_ms,
            "x_min": x1,
            "y_min": y1,
            "x_max": x2,
            "y_max": y2,
            "sharpness": frame_sample.sharpness,
            "_frame_index": frame_sample.frame_index,
            "_det_source": cand.get("source", ""),
            "_det_confidence": cand.get("confidence", 0.0),
            "_spatial_fallback_only": True,
        }
    )
    for col in QR_COLUMNS:
        record[col] = ""
    seed_price = guess_seed_price(cand.get("_seed_text", ""))
    if seed_price:
        record["price_card"] = seed_price
    return record


def guess_seed_price(text: str) -> str:
    text = str(text or "").strip()
    explicit = re.search(r"(?<!\d)(\d{1,4})\s*[,.:-]\s*(\d{2})(?!\d)", text)
    if explicit:
        value = int(explicit.group(1))
        cents = explicit.group(2)
        if 5 <= value <= 9999 and cents in {"99", "88", "69"}:
            return f"{value}.{explicit.group(2)}"
    # Plain OCR digits are often barcode/SKU fragments. A wrong price hurts
    # more than a blank field, so only explicit decimal-like prices are seeded.
    return ""


def choose_global_fraction_for_frame(ocr_candidates: list[dict], shelf_lanes: list[dict]) -> float:
    # Tested an adaptive global/lane split based on OCR-price lane coverage.
    # It looked plausible, but reduced matches on both 49_5 and 25_12-20.
    # Keep the empirically stabler fixed mix and let full-frame OCR proposals
    # add their own independent fallback rows.
    return 0.35


def best_ocr_angle(blocks: list[dict]) -> int:
    if not blocks:
        return 0
    scores: dict[int, float] = {}
    for block in blocks:
        text = str(block.get("text", "") or "")
        useful_chars = sum(ch.isalnum() for ch in text)
        if useful_chars < 2:
            continue
        angle = int(block.get("angle", 0) or 0) % 360
        confidence = float(block.get("confidence", 0.0) or 0.0)
        scores[angle] = scores.get(angle, 0.0) + confidence * min(24, useful_chars)
    if not scores:
        return 0
    return max(scores, key=scores.get)


def draw_bboxes(frame, candidates: list[dict]):
    overlay = frame.copy()
    for cand in candidates:
        x1, y1, x2, y2 = [int(v) for v in cand["bbox"]]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"{cand.get('source','')} {cand.get('confidence',0):.2f}"
        cv2.putText(overlay, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
    return overlay


def draw_roi_overlay(crop, roi: dict):
    overlay = crop.copy()
    h, w = crop.shape[:2]
    for name, zone in roi.items():
        x1, y1, x2, y2 = normalized_roi_to_bbox(zone, w, h)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 100, 0), 2)
        cv2.putText(overlay, name[:18], (x1, max(18, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 100, 0), 1, cv2.LINE_AA)
    return overlay


def _axis_aligned_text_bbox(block: dict) -> tuple[float, float, float, float] | None:
    points = block.get("bbox") or []
    if not points:
        return None
    xs = [float(point[0]) for point in points if len(point) >= 2]
    ys = [float(point[1]) for point in points if len(point) >= 2]
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def full_frame_text_candidates(frame, blocks: list[dict], limit: int = 20) -> list[dict]:
    if limit <= 0 or frame is None or frame.size == 0:
        return []
    height, width = frame.shape[:2]
    candidates: list[tuple[float, dict]] = []
    for block in blocks:
        text = str(block.get("text", "") or "")
        if re.search(r"\d\s*(?:g|gr|kg|ml|l|г|гр|кг|мл|л)\b", text.lower(), flags=re.IGNORECASE):
            continue
        digits = re.sub(r"\D", "", text)
        if len(digits) < 2 or len(digits) > 5:
            continue
        bbox = _axis_aligned_text_bbox(block)
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        text_area = bw * bh
        if text_area < 800 or text_area > 80_000:
            continue
        confidence = float(block.get("confidence", 0.0) or 0.0)
        if confidence < 0.45:
            continue
        if bh > bw * 1.25:
            specs = [(2.8, 0.8, 3.2, 0.8, "wide", 0.0)]
        else:
            specs = [(1.2, 1.8, 1.8, 1.8, "wide", 0.0)]
        for left, top, right, bottom, variant, penalty in specs:
            box = [x1 - left * bw, y1 - top * bh, x2 + right * bw, y2 + bottom * bh]
            clipped = [
                max(0, int(round(box[0]))),
                max(0, int(round(box[1]))),
                min(width, int(round(box[2]))),
                min(height, int(round(box[3]))),
            ]
            if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                continue
            area = (clipped[2] - clipped[0]) * (clipped[3] - clipped[1])
            area_ratio = area / max(1.0, float(width * height))
            if area_ratio < 0.001 or area_ratio > 0.035:
                continue
            candidates.append(
                (
                    confidence + penalty,
                    {
                        "bbox": clipped,
                        "source": "full_ocr_price",
                        "confidence": min(0.96, 0.80 + confidence * 0.16 + penalty),
                        "_seed_text": text,
                        "_seed_confidence": confidence,
                        "_ocr_proposal_variant": variant,
                    },
                )
            )
    return [cand for _, cand in sorted(candidates, key=lambda item: item[0], reverse=True)[:limit]]


def ocr_template_rois(crop, roi: dict, ocr: OCREngine, max_zones: int = 5) -> tuple[dict, list[dict]]:
    h, w = crop.shape[:2]
    roi_text: dict[str, str] = {}
    roi_blocks: list[dict] = []
    priority = [
        "barcode",
        "id_sku",
        "product_name",
        "price_card",
        "price_default",
        "price_discount",
        "discount_amount",
        "print_datetime",
    ]
    ordered_names = [name for name in priority if name in roi] + [name for name in roi if name not in priority and name != "qr"]
    for name in ordered_names[:max_zones]:
        zone = roi[name]
        x1, y1, x2, y2 = normalized_roi_to_bbox(zone, w, h)
        patch = crop[y1:y2, x1:x2]
        if name in {"barcode", "id_sku"}:
            blocks = ocr.recognize_text(patch, roi_name=name, try_rotations=False, angles=[0], upscale=False)
        else:
            blocks = ocr.recognize_text(patch, roi_name=name, try_rotations=False)
        roi_text[name] = blocks_to_text(blocks)
        roi_blocks.extend(blocks)
    return roi_text, roi_blocks


def process_video(
    video_path: str | Path,
    output_csv_path: str | Path | None = None,
    debug: bool = False,
    sample_fps: float = 2.0,
    min_sharpness: float = 0.0,
    max_frames: int | None = None,
    max_ocr_candidates_per_frame: int = 10,
    max_roi_zones: int = 2,
    roi_ocr_mode: str = "auto",
    qr_mode: str = "fast",
    ocr_angles: list[int] | None = None,
    max_spatial_fallbacks_per_frame: int = 0,
    full_frame_ocr_candidates_per_frame: int = 0,
    yolo_model_path: str | Path | None = None,
    debug_max_frames: int = 20,
    debug_max_crops: int = 120,
    debug_max_json: int = 120,
    progress_callback=None,
) -> tuple[str, dict]:
    video_path = Path(video_path)
    output_csv_path = Path(output_csv_path) if output_csv_path else OUTPUT_DIR / f"{video_path.stem}_result.csv"
    ensure_dir(output_csv_path.parent)

    debug_root = ensure_dir(DEBUG_DIR / video_path.stem) if debug else None
    frame_dir = ensure_dir(debug_root / "frames") if debug_root else None
    bbox_dir = ensure_dir(debug_root / "bbox_frames") if debug_root else None
    crop_dir = ensure_dir(debug_root / "crops") if debug_root else None
    roi_dir = ensure_dir(debug_root / "roi_overlays") if debug_root else None
    ocr_dir = ensure_dir(debug_root / "ocr") if debug_root else None
    parsed_dir = ensure_dir(debug_root / "parsed") if debug_root else None

    detector = OpenCVDetector()
    qr_parser = QRParser()
    ocr = OCREngine()
    template_classifier = TemplateClassifier()
    ocr_angles = ocr_angles or [0, 270]

    stats = {
        "video": str(video_path),
        "ocr_engine": ocr.engine_name,
        "frames_processed": 0,
        "raw_detections": 0,
        "unique_price_tags": 0,
        "qr_decoded_count": 0,
        "ocr_success_count": 0,
        "errors": 0,
        "debug_saved_crops": 0,
        "debug_saved_json": 0,
        "raw_spatial_fallback_records": 0,
        "full_frame_ocr_candidates": 0,
    }
    yolo_detector = None
    if yolo_model_path:
        try:
            yolo_detector = YOLODetector(str(yolo_model_path))
            LOGGER.info("Loaded YOLO detector: %s", yolo_model_path)
        except Exception as exc:
            stats["errors"] += 1
            LOGGER.warning("Cannot load YOLO detector %s: %s", yolo_model_path, exc)
    try:
        stats["metadata"] = get_video_metadata(video_path)
    except Exception as exc:
        LOGGER.warning("Cannot read metadata: %s", exc)

    raw_records: list[dict] = []
    raw_debug: list[dict] = []
    raw_candidates: list[dict] = []
    debug_crop_counter = 0
    debug_json_counter = 0
    spatial_fallback_counts: dict[int, int] = {}
    full_ocr_fallback_counts: dict[int, int] = {}
    estimated_samples = None
    if stats.get("metadata"):
        estimated_samples = int(math.ceil(float(stats["metadata"].get("duration_sec", 0.0) or 0.0) * sample_fps))
        if max_frames is not None:
            estimated_samples = min(estimated_samples, max_frames)

    for frame_sample in iter_video_frames(
        video_path,
        sample_fps=sample_fps,
        min_sharpness=min_sharpness,
        debug_dir=None,
        save_frames=False,
        max_frames=max_frames,
    ):
        stats["frames_processed"] += 1
        if progress_callback and estimated_samples:
            progress_callback(stats["frames_processed"], estimated_samples, "Reading frames and detecting tags")
        frame = frame_sample.image
        if debug and frame_dir is not None and stats["frames_processed"] <= debug_max_frames:
            write_image(
                frame_dir / f"frame_{frame_sample.frame_index:06d}_ts_{frame_sample.timestamp_ms:08d}_sharp_{frame_sample.sharpness:.0f}.jpg",
                frame,
                quality=90,
            )
        try:
            candidates = detector.detect(frame)
            if yolo_detector is not None:
                candidates.extend(yolo_detector.detect(frame))
                candidates = nms(candidates, iou_threshold=0.35, limit=80)
        except Exception as exc:
            stats["errors"] += 1
            LOGGER.warning("Detector failed on frame %s: %s", frame_sample.frame_index, exc)
            candidates = []
        shelf_lanes = estimate_shelf_lanes(frame) if max_spatial_fallbacks_per_frame > 0 else []
        ocr_candidates: list[dict] = []
        if full_frame_ocr_candidates_per_frame > 0 and ocr.engine_name != "none":
            try:
                frame_blocks = ocr.recognize_text(frame, try_rotations=False, angles=[0], upscale=False)
                ocr_candidates = full_frame_text_candidates(frame, frame_blocks, limit=full_frame_ocr_candidates_per_frame)
                candidates.extend(ocr_candidates)
                stats["full_frame_ocr_candidates"] += len(ocr_candidates)
            except Exception as exc:
                stats["errors"] += 1
                LOGGER.warning("Full-frame OCR candidates failed on frame %s: %s", frame_sample.frame_index, exc)
        stats["raw_detections"] += len(candidates)
        global_fraction = choose_global_fraction_for_frame(ocr_candidates, shelf_lanes)
        lane_ranks = (
            rank_candidates_by_lane(candidates, shelf_lanes, max_rank=max_spatial_fallbacks_per_frame, global_fraction=global_fraction)
            if shelf_lanes
            else {}
        )

        if debug and bbox_dir is not None and stats["frames_processed"] <= debug_max_frames:
            overlay = draw_bboxes(frame, candidates)
            write_image(bbox_dir / f"frame_{frame_sample.frame_index:06d}_bbox.jpg", overlay, quality=88)

        for det_idx, cand in enumerate(candidates):
            try:
                crop, clipped_bbox = crop_bbox(frame, cand["bbox"], pad=2)
                x1, y1, x2, y2 = clipped_bbox
                stem = f"frame_{frame_sample.frame_index:06d}_det_{det_idx:03d}_{safe_filename(cand.get('source','det'))}"
                spatial_rank = lane_ranks.get(det_idx)
                if shelf_lanes and max_spatial_fallbacks_per_frame > 0:
                    global_budget = min(max_ocr_candidates_per_frame, 3)
                    do_expensive_read = (
                        det_idx < global_budget
                        or (spatial_rank is not None and spatial_rank < max_ocr_candidates_per_frame)
                    ) or cand.get("source") == "qr"
                else:
                    do_expensive_read = det_idx < max_ocr_candidates_per_frame or cand.get("source") == "qr"
                raw_candidates.append(
                    {
                        "frame_index": frame_sample.frame_index,
                        "frame_timestamp": frame_sample.timestamp_ms,
                        "det_idx": det_idx,
                        "bbox": clipped_bbox,
                        "source": cand.get("source", ""),
                        "confidence": cand.get("confidence", 0.0),
                        "ocr_attempted": bool(do_expensive_read),
                        "shelf_lane_rank": lane_ranks.get(det_idx, ""),
                        "shelf_lane_score": round(candidate_lane_score(clipped_bbox, shelf_lanes), 4) if shelf_lanes else 0.0,
                    }
                )
                if do_expensive_read:
                    qr_fields = qr_parser.read(crop, mode=qr_mode)
                    raw_payloads = qr_fields.pop("_raw_payloads", [])
                    barcode_payloads = qr_fields.pop("_barcode_payloads", [])
                    if any(qr_fields.get(k) for k in qr_fields if not str(k).startswith("_")):
                        stats["qr_decoded_count"] += 1
                    qr_fields["_barcode_payloads"] = barcode_payloads
                    has_qr = bool(raw_payloads)
                    rough_blocks = ocr.recognize_text(crop, try_rotations=False, angles=ocr_angles, upscale=False)
                else:
                    qr_fields = {}
                    raw_payloads = []
                    barcode_payloads = []
                    has_qr = False
                    rough_blocks = []
                if rough_blocks:
                    stats["ocr_success_count"] += 1
                rough_text = blocks_to_text(rough_blocks)
                ocr_angle = best_ocr_angle(rough_blocks)
                analysis_crop = rotate_image(crop, ocr_angle)

                template_info = template_classifier.classify(analysis_crop, rough_text=rough_text, qr_fields=qr_fields, has_qr=has_qr)
                roi_text: dict[str, str] = {}
                roi_blocks: list[dict] = []
                if do_expensive_read and not template_info.get("use_generic_fallback") and should_run_roi_ocr(template_info, rough_text, roi_ocr_mode, max_roi_zones):
                    roi_text, roi_blocks = ocr_template_rois(analysis_crop, template_info["roi"], ocr, max_zones=max_roi_zones)
                    if debug and roi_dir is not None and debug_crop_counter < debug_max_crops:
                        write_image(roi_dir / f"{stem}_{template_info['template_id']}_angle_{ocr_angle}.jpg", draw_roi_overlay(analysis_crop, template_info["roi"]), quality=92)

                all_blocks = rough_blocks + roi_blocks
                parsed = parse_fields(all_blocks, analysis_crop, template_info=template_info, qr_fields=qr_fields, roi_text=roi_text)
                parsed.update(
                    {
                        "filename": video_path.name,
                        "frame_timestamp": frame_sample.timestamp_ms,
                        "x_min": x1,
                        "y_min": y1,
                        "x_max": x2,
                        "y_max": y2,
                        "sharpness": frame_sample.sharpness,
                        "_frame_index": frame_sample.frame_index,
                        "_det_source": cand.get("source", ""),
                        "_det_confidence": cand.get("confidence", 0.0),
                        "_template_id": template_info.get("template_id", ""),
                        "_template_confidence": template_info.get("confidence", 0.0),
                        "_ocr_best_angle": ocr_angle,
                    }
                )
                has_evidence = has_price_tag_evidence(parsed)
                lane_score = candidate_lane_score(clipped_bbox, shelf_lanes) if shelf_lanes else 0.0
                is_full_ocr_candidate = str(cand.get("source", "")) == "full_ocr_price"
                within_spatial_budget = (
                    max_spatial_fallbacks_per_frame > 0
                    and spatial_fallback_counts.get(frame_sample.frame_index, 0) < max_spatial_fallbacks_per_frame
                    and spatial_rank is not None
                    and spatial_rank < max_spatial_fallbacks_per_frame
                )
                within_full_ocr_budget = (
                    is_full_ocr_candidate
                    and full_frame_ocr_candidates_per_frame > 0
                    and full_ocr_fallback_counts.get(frame_sample.frame_index, 0) < full_frame_ocr_candidates_per_frame
                )
                keep_spatial_fallback = (
                    not has_evidence
                    and (within_spatial_budget or within_full_ocr_budget)
                    and should_keep_spatial_fallback(cand, clipped_bbox, frame.shape, det_idx, spatial_rank=spatial_rank, lane_score=lane_score)
                )
                if keep_spatial_fallback:
                    if within_full_ocr_budget:
                        full_ocr_fallback_counts[frame_sample.frame_index] = full_ocr_fallback_counts.get(frame_sample.frame_index, 0) + 1
                    else:
                        spatial_fallback_counts[frame_sample.frame_index] = spatial_fallback_counts.get(frame_sample.frame_index, 0) + 1
                    parsed = make_spatial_fallback_record(video_path.name, frame_sample, clipped_bbox, cand, parsed.get("color", ""))
                    parsed["_shelf_lane_rank"] = spatial_rank
                    parsed["_shelf_lane_score"] = round(lane_score, 4)
                    parsed["_full_frame_seed_text"] = cand.get("_seed_text", "")
                    stats["raw_spatial_fallback_records"] += 1
                if has_evidence or keep_spatial_fallback or (do_expensive_read and ocr.engine_name == "none"):
                    raw_records.append(parsed)

                if debug and crop_dir is not None and (do_expensive_read or has_evidence or keep_spatial_fallback) and debug_crop_counter < debug_max_crops:
                    write_image(crop_dir / f"{stem}.jpg", crop, quality=95)
                    debug_crop_counter += 1
                    stats["debug_saved_crops"] = debug_crop_counter

                debug_item = {
                    "candidate": cand,
                    "bbox": clipped_bbox,
                    "frame_index": frame_sample.frame_index,
                    "frame_timestamp": frame_sample.timestamp_ms,
                    "qr_payloads": raw_payloads,
                    "barcode_payloads": barcode_payloads,
                    "rough_ocr": rough_blocks,
                    "ocr_best_angle": ocr_angle,
                    "template": template_info,
                    "roi_text": roi_text,
                    "parsed": parsed,
                }
                if do_expensive_read or has_evidence or keep_spatial_fallback:
                    raw_debug.append(debug_item)
                if debug and ocr_dir is not None and parsed_dir is not None and (do_expensive_read or has_evidence or keep_spatial_fallback) and debug_json_counter < debug_max_json:
                    save_json(ocr_dir / f"{stem}.json", {"rough_ocr": rough_blocks, "roi_text": roi_text, "roi_blocks": roi_blocks})
                    save_json(parsed_dir / f"{stem}.json", debug_item)
                    debug_json_counter += 1
                    stats["debug_saved_json"] = debug_json_counter
            except Exception as exc:
                stats["errors"] += 1
                LOGGER.warning("Candidate failed: %s", exc)
                LOGGER.debug(traceback.format_exc())

    unique_records = deduplicate(raw_records)
    stats["unique_price_tags"] = len(unique_records)
    stats["raw_records"] = len(raw_records)
    stats["unique_with_barcode"] = sum(1 for record in unique_records if str(record.get("barcode", "") or "").strip())
    stats["unique_spatial_fallback"] = stats["unique_price_tags"] - stats["unique_with_barcode"]
    write_csv(unique_records, output_csv_path)

    if debug and debug_root is not None:
        save_json(debug_root / "raw_detections.json", raw_debug)
        save_json(debug_root / "raw_candidates.json", raw_candidates)
        pd.DataFrame(raw_candidates).to_csv(debug_root / "raw_candidates.csv", index=False, encoding="utf-8")
        records_to_dataframe(raw_records).to_csv(debug_root / "raw_detections.csv", index=False, encoding="utf-8")
        save_json(debug_root / "stats.json", stats)

    return str(output_csv_path), stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Lenta price tag recognition baseline")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--out", default=None, help="Output CSV path")
    parser.add_argument("--debug", action="store_true", help="Save debug artifacts")
    parser.add_argument("--sample-fps", type=float, default=2.0, help="Frame sampling FPS")
    parser.add_argument("--min-sharpness", type=float, default=0.0, help="Skip frames below Laplacian variance")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional limit for quick experiments")
    parser.add_argument("--max-ocr-candidates-per-frame", type=int, default=10, help="Run QR/OCR only for top N candidates per sampled frame")
    parser.add_argument("--max-roi-zones", type=int, default=2, help="Run ROI OCR for at most N template zones")
    parser.add_argument("--roi-ocr-mode", choices=["auto", "always", "off"], default="auto", help="Template ROI OCR strategy")
    parser.add_argument("--qr-mode", choices=["fast", "full"], default="fast", help="QR decode strategy")
    parser.add_argument("--ocr-angles", default="0,270", help="Comma-separated crop rotations for rough OCR, e.g. 0 or 0,270")
    parser.add_argument("--max-spatial-fallbacks-per-frame", type=int, default=0, help="Experimental: keep up to N detector-only fallback rows per frame")
    parser.add_argument("--full-frame-ocr-candidates-per-frame", type=int, default=0, help="Use full-frame OCR text boxes as additional price-tag proposals")
    parser.add_argument("--yolo-model", default=None, help="Optional local YOLO/Ultralytics detector weights")
    parser.add_argument("--debug-max-frames", type=int, default=20, help="Save debug frame/overlay images for first N sampled frames")
    parser.add_argument("--debug-max-crops", type=int, default=120, help="Save at most N debug crop images")
    parser.add_argument("--debug-max-json", type=int, default=120, help="Save at most N detailed debug JSON files")
    args = parser.parse_args()

    csv_path, stats = process_video(
        args.video,
        output_csv_path=args.out,
        debug=args.debug,
        sample_fps=args.sample_fps,
        min_sharpness=args.min_sharpness,
        max_frames=args.max_frames,
        max_ocr_candidates_per_frame=args.max_ocr_candidates_per_frame,
        max_roi_zones=args.max_roi_zones,
        roi_ocr_mode=args.roi_ocr_mode,
        qr_mode=args.qr_mode,
        ocr_angles=[int(x) for x in args.ocr_angles.split(",") if x.strip()],
        max_spatial_fallbacks_per_frame=args.max_spatial_fallbacks_per_frame,
        full_frame_ocr_candidates_per_frame=args.full_frame_ocr_candidates_per_frame,
        yolo_model_path=args.yolo_model,
        debug_max_frames=args.debug_max_frames,
        debug_max_crops=args.debug_max_crops,
        debug_max_json=args.debug_max_json,
    )
    print(f"CSV saved: {csv_path}")
    for key in [
        "frames_processed",
        "raw_detections",
        "raw_records",
        "full_frame_ocr_candidates",
        "unique_price_tags",
        "unique_with_barcode",
        "unique_spatial_fallback",
        "qr_decoded_count",
        "ocr_success_count",
        "ocr_engine",
        "errors",
    ]:
        print(f"{key}: {stats.get(key)}")


if __name__ == "__main__":
    main()

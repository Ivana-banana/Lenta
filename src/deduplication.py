from __future__ import annotations

import hashlib
import re
from collections import Counter
from difflib import SequenceMatcher

from .utils import quality_score


NO_VALUES = {"", "nan", "none", "нет"}


def _clean(value) -> str:
    text = str(value or "").strip()
    if text.lower() in NO_VALUES:
        return ""
    return text


def _primary_key(record: dict) -> str | None:
    for field in ["barcode", "qr_code_barcode", "id_sku"]:
        value = _clean(record.get(field))
        if value and value != "нет":
            return f"{field}:{value}"
    return None


def _fallback_key(record: dict) -> str:
    name = _clean(record.get("product_name")).lower()
    price = _clean(record.get("price_card")) or _clean(record.get("price_default"))
    if name or price:
        digest = hashlib.md5(f"{name[:80]}|{price}".encode("utf-8")).hexdigest()[:12]
        return f"text:{digest}"
    # Last-resort grouping by candidate source/color and coarse shelf position.
    cx = (float(record.get("x_min", 0) or 0) + float(record.get("x_max", 0) or 0)) / 2.0
    cy = (float(record.get("y_min", 0) or 0) + float(record.get("y_max", 0) or 0)) / 2.0
    if record.get("_spatial_fallback_only"):
        timestamp = int(float(record.get("frame_timestamp", 0) or 0))
        frame = int(float(record.get("_frame_index", 0) or 0))
        # Spatial fallback is judged by timestamp+bbox. Merging across adjacent
        # frames can accidentally keep a worse box, so only collapse near
        # duplicates inside the same sampled frame.
        return f"spatial:{record.get('color','')}:f{frame}:t{timestamp // 1000}:x{int(cx // 120)}:y{int(cy // 120)}"
    return f"unknown:{record.get('color','')}:x{int(cx // 220)}:y{int(cy // 220)}"


def _price_score(value: str, count: int) -> tuple:
    try:
        numeric = float(str(value).replace(",", "."))
    except ValueError:
        numeric = 0.0
    cents = int(round(numeric * 100)) % 100
    non_zero_cents = 1 if cents not in {0, 90} else 0
    plausible = 1 if 1.0 <= numeric <= 30000 else 0
    return (count, plausible, non_zero_cents, numeric)


def _choose_price(items: list[dict], field: str) -> str:
    values = [_clean(item.get(field)) for item in items]
    values = [v.replace(",", ".") for v in values if v]
    if not values:
        return ""
    counts = Counter(values)
    return max(counts, key=lambda value: _price_score(value, counts[value]))


def _choose_identifier(items: list[dict], field: str) -> str:
    values = [_clean(item.get(field)) for item in items]
    values = [re.sub(r"\D", "", v) if field in {"barcode", "qr_code_barcode", "id_sku"} else v for v in values if v]
    values = [v for v in values if v]
    if not values:
        return ""
    counts = Counter(values)
    return max(counts, key=lambda value: (counts[value], len(value)))


def _name_score(value: str) -> tuple:
    letters = sum(ch.isalpha() for ch in value)
    digits = sum(ch.isdigit() for ch in value)
    vowels = sum(ch.lower() in "aeiouаеёиоуыэюя" for ch in value)
    vowel_ratio = vowels / max(1, letters)
    sane_vowels = 1 if 0.18 <= vowel_ratio <= 0.75 else 0
    return (sane_vowels, letters - digits, len(value))


def _choose_text(items: list[dict], field: str) -> str:
    values = [_clean(item.get(field)) for item in items]
    values = [v for v in values if v]
    if not values:
        return ""
    counts = Counter(values)
    return max(counts, key=lambda value: (counts[value], *_name_score(value)))


def _merge_group(items: list[dict]) -> dict:
    best = max(items, key=quality_score).copy()
    for field in ["price_default", "price_card", "price_discount"]:
        chosen = _choose_price(items, field)
        if chosen:
            best[field] = chosen
    for field in ["barcode", "qr_code_barcode", "id_sku"]:
        chosen = _choose_identifier(items, field)
        if chosen:
            best[field] = chosen
    for field in ["product_name", "discount_amount", "print_datetime", "code", "additional_info", "special_symbols"]:
        chosen = _choose_text(items, field)
        if chosen:
            best[field] = chosen
    return best


def _bbox(record: dict) -> list[float]:
    return [float(record.get(col, 0) or 0) for col in ["x_min", "y_min", "x_max", "y_max"]]


def _iou(box_a: list[float], box_b: list[float]) -> float:
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


def suppress_same_frame_duplicates(records: list[dict], iou_threshold: float = 0.55) -> list[dict]:
    by_frame: dict[int, list[dict]] = {}
    passthrough: list[dict] = []
    for record in records:
        if _primary_key(record):
            passthrough.append(record)
            continue
        frame = int(float(record.get("_frame_index", 0) or 0))
        by_frame.setdefault(frame, []).append(record)

    kept: list[dict] = passthrough[:]
    for frame_records in by_frame.values():
        selected: list[dict] = []
        for record in sorted(frame_records, key=quality_score, reverse=True):
            box = _bbox(record)
            if any(_iou(box, _bbox(existing)) >= iou_threshold for existing in selected):
                continue
            selected.append(record)
        kept.extend(selected)
    return kept


def deduplicate(records: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    fuzzy_pool: list[tuple[str, dict]] = []

    for record in records:
        key = _primary_key(record)
        if key:
            groups.setdefault(key, []).append(record)
        else:
            fuzzy_pool.append((_fallback_key(record), record))

    for key, record in fuzzy_pool:
        if key.startswith("text:"):
            placed = False
            name = _clean(record.get("product_name")).lower()
            price = _clean(record.get("price_card")) or _clean(record.get("price_default"))
            for existing_key, existing_records in list(groups.items()):
                exemplar = existing_records[0]
                ex_name = _clean(exemplar.get("product_name")).lower()
                ex_price = _clean(exemplar.get("price_card")) or _clean(exemplar.get("price_default"))
                if price == ex_price and name and ex_name and SequenceMatcher(None, name, ex_name).ratio() > 0.82:
                    groups[existing_key].append(record)
                    placed = True
                    break
            if not placed:
                groups.setdefault(key, []).append(record)
        else:
            groups.setdefault(key, []).append(record)

    unique = []
    for key, items in groups.items():
        best = _merge_group(items)
        best["_dedup_key"] = key
        best["_duplicates"] = len(items)
        unique.append(best)
    unique.sort(key=lambda r: (int(float(r.get("frame_timestamp", 0) or 0)), float(r.get("x_min", 0) or 0)))
    return unique

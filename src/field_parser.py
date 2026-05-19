from __future__ import annotations

import re
from collections import Counter

import numpy as np

from .barcode_engine import choose_best_ean13, choose_best_ean13_from_values, ean13_candidates_from_blocks, ean13_candidates_from_text, is_ean13
from .config import NO_VALUE, QR_COLUMNS, TAG_FIELDS
from .template_classifier import dominant_color
from .utils import normalize_text


PRICE_RE = re.compile(r"(?<!\d)(\d{1,5})[,.:-](\d{2})(?!\d)")
INT_PRICE_RE = re.compile(r"(?<!\d)(\d{2,5})(?!\d)")
BARCODE_RE = re.compile(r"(?<!\d)((?:\d[\s.,-]?){8,14})(?!\d)")
SKU_RE = re.compile(r"(?<!\d)((?:\d[\s-]?){9,13})(?!\d)")
DATE_RE = re.compile(r"\b(\d{2}[./-]\d{2}[./-]\d{2,4})(?:\s+(\d{1,2}:\d{2}))?\b|\b(\d{4}-\d{2}-\d{2})(?:\s+(\d{1,2}:\d{2}))?\b")
CODE_RE = re.compile(r"\b\d{2}_\d{5,6}(?:\s*[-–]\s*\d{5,6})?\b")
DISCOUNT_RE = re.compile(r"[-−]?\s*\d{1,3}\s*%|\bвыгода\b|\bскидк\w*\b", re.IGNORECASE)
UNIT_VALUE_RE = re.compile(
    r"(?<!\d)\d+(?:[,.]\d+)?\s*(?:л|l|литр\w*|мл|ml|кг|kg|гр?|g|шт|pcs?|pc|cl|сл)(?![a-zа-я])",
    re.IGNORECASE,
)
PRICE_TOKEN_RE = re.compile(r"(?<!\d)\d{1,5}(?!\d)")
LATIN_BRAND_TOKENS = {
    "active",
    "awake",
    "exponenta",
    "go",
    "hyper",
    "lets",
    "let",
    "life",
    "molvest",
    "protein",
    "velle",
    "vivo",
}


def blocks_to_text(blocks: list[dict]) -> str:
    return "\n".join(normalize_text(b.get("text", "")) for b in blocks if normalize_text(b.get("text", "")))


def normalize_price(text: str) -> str:
    text = text.replace(" ", "").replace(",", ".").replace(":", ".").replace("-", ".")
    m = re.search(r"(\d{1,5})[.](\d{2})", text)
    if m:
        return f"{int(m.group(1))}.{m.group(2)}"
    return text


def split_text_lines(text: str) -> list[str]:
    return [normalize_text(line) for line in re.split(r"[\n\r]+", str(text or "")) if normalize_text(line)]


def _identifier_like_line(line: str) -> bool:
    compact = re.sub(r"\D", "", line)
    if len(compact) < 8:
        return False
    non_space = re.sub(r"\s+", "", line)
    if not non_space:
        return False
    digit_sep_chars = sum(ch.isdigit() or ch in "- " for ch in non_space)
    return digit_sep_chars / max(1, len(non_space)) > 0.82


def _looks_like_ean13(digits: str) -> bool:
    return is_ean13(digits)


def _text_for_price_search(text: str) -> str:
    lines = []
    for line in split_text_lines(text):
        if _identifier_like_line(line):
            continue
        line = DISCOUNT_RE.sub(" ", line)
        line = UNIT_VALUE_RE.sub(" ", line)
        lines.append(line)
    return "\n".join(lines)


def extract_prices(text: str) -> list[str]:
    text_for_prices = _text_for_price_search(text)
    prices = [normalize_price(".".join(m.groups())) for m in PRICE_RE.finditer(text_for_prices)]
    cleaned = []
    for price in prices:
        try:
            value = float(price)
        except ValueError:
            continue
        if 1.0 <= value <= 30000 and price not in cleaned:
            cleaned.append(price)
    return cleaned


def _block_geometry(block: dict) -> dict:
    pts = block.get("bbox") or []
    if not pts:
        return {"cx": 0.0, "cy": 0.0, "w": 0.0, "h": 0.0, "area": 0.0}
    xs = [float(p[0]) for p in pts if len(p) >= 2]
    ys = [float(p[1]) for p in pts if len(p) >= 2]
    if not xs or not ys:
        return {"cx": 0.0, "cy": 0.0, "w": 0.0, "h": 0.0, "area": 0.0}
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    return {
        "cx": (max(xs) + min(xs)) / 2.0,
        "cy": (max(ys) + min(ys)) / 2.0,
        "w": w,
        "h": h,
        "area": max(0.0, w) * max(0.0, h),
    }


def _numeric_block_items(text_blocks: list[dict]) -> tuple[list[dict], list[dict], list[tuple[str, float]]]:
    wholes: list[dict] = []
    cents: list[dict] = []
    explicit_prices: list[tuple[str, float]] = []
    for idx, block in enumerate(text_blocks):
        raw = normalize_text(block.get("text", ""))
        if not raw or _identifier_like_line(raw) or UNIT_VALUE_RE.search(raw) or DISCOUNT_RE.search(raw):
            continue
        geom = _block_geometry(block)
        confidence = float(block.get("confidence", 0.0) or 0.0)
        visual = min(4.0, (geom["area"] ** 0.5) / 42.0) if geom["area"] else 0.8
        base_score = confidence * 1.2 + visual
        for match in PRICE_RE.finditer(raw):
            explicit_prices.append((normalize_price(".".join(match.groups())), base_score + 0.8))
        cleaned = raw.replace("O", "0").replace("О", "0").replace("o", "0")
        cleaned = re.sub(r"[^\d]", "", cleaned)
        if not re.fullmatch(r"\d{1,5}", cleaned):
            continue
        value = int(cleaned)
        item = {
            "idx": idx,
            "text": cleaned,
            "value": value,
            "angle": int(block.get("angle", 0) or 0) % 360,
            "score": base_score,
            **geom,
        }
        if 1 <= len(cleaned) <= 4 and 1 <= value <= 30000:
            wholes.append(item)
        if len(cleaned) == 2 and 0 <= value <= 99:
            cents.append(item)
    return wholes, cents, explicit_prices


def _price_pair_spatial_score(whole: dict, cents: dict) -> float:
    if whole["idx"] == cents["idx"] or whole["angle"] != cents["angle"]:
        return -1.0
    if whole["area"] <= 0 or cents["area"] <= 0:
        return 0.15
    if whole["area"] < cents["area"] * 1.2:
        return -1.0
    dx = cents["cx"] - whole["cx"]
    dy = abs(cents["cy"] - whole["cy"])
    near_right = dx > -whole["w"] * 0.10 and dx < whole["w"] * 1.30
    aligned = dy < max(whole["h"], cents["h"]) * 0.85
    if near_right and aligned:
        return 1.2
    # Some OCR boxes for vertical shelf labels put kopecks slightly below.
    below = cents["cy"] > whole["cy"] and abs(dx) < whole["w"] * 0.75 and dy < whole["h"] * 1.25
    return 0.45 if below else -1.0


def extract_ranked_prices_from_blocks(text_blocks: list[dict]) -> list[str]:
    wholes, cents, explicit_prices = _numeric_block_items(text_blocks)
    scored: list[tuple[str, float]] = list(explicit_prices)
    cross_angle_scored: list[tuple[str, float]] = []
    for whole in wholes:
        for cent in cents:
            spatial = _price_pair_spatial_score(whole, cent)
            value = float(f"{whole['value']}.{cent['value']:02d}")
            if not (1.0 <= value <= 30000):
                continue
            price = f"{whole['value']}.{cent['value']:02d}"
            if spatial >= 0:
                score = whole["score"] * 0.78 + cent["score"] * 0.22 + spatial
                scored.append((price, score))
                continue
            if whole["angle"] != cent["angle"] and whole["area"] > cent["area"] * 3.0 and whole["score"] > 1.7:
                # Mixed-angle OCR happens on side-facing tags: the large ruble
                # part may be read in one orientation and kopecks in another.
                cross_angle_scored.append((price, whole["score"] * 0.72 + cent["score"] * 0.18))

    if not scored:
        scored.extend(cross_angle_scored)

    best: dict[str, float] = {}
    for price, score in scored:
        if price not in best or score > best[price]:
            best[price] = score
    return [price for price, _ in sorted(best.items(), key=lambda item: item[1], reverse=True)]


def _price_value(price: str) -> float | None:
    try:
        return float(str(price).replace(",", "."))
    except ValueError:
        return None


def filter_visible_ocr_prices(prices: list[str]) -> list[str]:
    """Keep only conservative OCR-visible shelf prices.

    The price blocks in the robot videos are small and often sit next to
    product packaging text like "30g". Pairing those numbers creates attractive
    but wrong values such as 169.30. QR payload prices are handled separately;
    this filter only guards OCR-derived visible prices.
    """
    trusted = []
    for price in prices:
        normalized = normalize_price(str(price))
        value = _price_value(normalized)
        if value is None or not (5.0 <= value <= 30000):
            continue
        match = re.search(r"\.(\d{2})$", normalized)
        cents = match.group(1) if match else ""
        if cents in {"99", "88", "69"}:
            trusted.append(normalized)
    return trusted


def assign_prices(result: dict, prices: list[str], mechanic: str) -> None:
    unique: list[str] = []
    for price in prices:
        normalized = normalize_price(str(price))
        if _price_value(normalized) is not None and normalized not in unique:
            unique.append(normalized)
    if not unique:
        return

    promo_mechanics = {"action", "sale", "threshold_from", "threshold_to", "BOGOF", "MNC"}
    if mechanic in promo_mechanics:
        promo = unique[0]
        result["price_card"] = promo
        promo_value = _price_value(promo)
        default_candidates = []
        for price in unique[1:]:
            value = _price_value(price)
            if value is None or promo_value is None:
                continue
            if value > promo_value * 1.02 and value <= promo_value * 5.0:
                default_candidates.append((value, price))
        if default_candidates:
            result["price_default"] = max(default_candidates)[1]
        return

    card_price = unique[0]
    result["price_card"] = card_price
    card_value = _price_value(card_price)
    values = [(_price_value(price), price) for price in unique[1:]]
    higher_prices = [(value, price) for value, price in values if value is not None and card_value is not None and value > card_value * 1.02]
    if higher_prices:
        result["price_default"] = max(higher_prices)[1]


def score_product_name(text: str) -> float:
    line = normalize_text(text)
    if not line:
        return -999.0
    letters = sum(ch.isalpha() for ch in line)
    digits = sum(ch.isdigit() for ch in line)
    if letters < 4 or digits > letters * 2:
        return -999.0
    cyrillic_letters = sum("а" <= ch.lower() <= "я" or ch.lower() == "ё" for ch in line)
    latin_tokens = set(re.findall(r"[a-z]{2,}", line.lower()))
    word_count = len(re.findall(r"[A-Za-zА-Яа-яЁё]{2,}", line))
    has_latin_brand_signal = bool(latin_tokens & LATIN_BRAND_TOKENS) and word_count >= 2 and letters >= 10
    if cyrillic_letters < 3 and not has_latin_brand_signal:
        return -999.0
    bad_tokens = ("qr", "скид", "руб", "%", "цена", "итого", "карта", "штрих", "id", "sku")
    lowered = line.lower()
    if any(tok in lowered for tok in bad_tokens):
        return -999.0
    unit_penalty = 0.0
    if UNIT_VALUE_RE.search(line):
        unit_penalty += 14.0
    if any(country in lowered for country in ["россия", "франция", "италия", "испания", "чили", "аргентина"]):
        unit_penalty += 8.0
    if re.fullmatch(r"[\W_]*(?:росс\w*|pocc\w*|pocch\w*)[\W_\d.,lл ]*", lowered):
        unit_penalty += 18.0
    uppercase_bonus = min(8.0, sum(ch.isupper() for ch in line) * 0.35)
    return letters * 2.0 + len(line) * 0.35 + uppercase_bonus - digits * 1.1 - unit_penalty


def extract_product_name(text: str) -> str:
    candidates: list[tuple[float, str]] = []
    for line in split_text_lines(text):
        score = score_product_name(line)
        if score > -999.0:
            candidates.append((score, line))
    if not candidates:
        return ""
    return max(candidates, key=lambda item: item[0])[1][:300]


def choose_barcode(text: str, qr_fields: dict, text_blocks: list[dict] | None = None) -> str:
    qr_barcode = choose_best_ean13_from_values([str(qr_fields.get("qr_code_barcode", "") or "")], source="qr")
    if qr_barcode:
        return qr_barcode

    zxing_barcode = choose_best_ean13_from_values(qr_fields.get("_barcode_payloads", []) or [], source="zxing")
    if zxing_barcode:
        return zxing_barcode

    block_candidates = ean13_candidates_from_blocks(text_blocks or [], allow_repair=False)
    text_candidates = []
    for line in split_text_lines(text):
        text_candidates.extend(ean13_candidates_from_text(line, allow_repair=False, source="ocr_text", base_score=0.75))
    barcode = choose_best_ean13(block_candidates + text_candidates)
    if barcode:
        return barcode

    repair_candidates = ean13_candidates_from_blocks(text_blocks or [], allow_repair=True)
    for line in split_text_lines(text):
        repair_candidates.extend(ean13_candidates_from_text(line, allow_repair=True, source="ocr_text_repair", base_score=0.45))
    return choose_best_ean13(repair_candidates)


def choose_sku(text: str, barcode: str) -> str:
    lowered = text.lower()
    labeled = re.search(r"(?:sku|id[_\s-]*sku|артикул|код\s+товара)\D{0,12}(\d{8,13})", lowered, flags=re.IGNORECASE)
    if labeled:
        return labeled.group(1)
    candidates = []
    for line in split_text_lines(text):
        if _identifier_like_line(line):
            continue
        if re.search(r"\b\d{1,4}\s+\d{2}\b", line):
            continue
        for match in SKU_RE.findall(line):
            candidates.append(re.sub(r"\D", "", match))
    candidates = [m for m in candidates if m != barcode]
    if not candidates:
        return ""
    counts = Counter(candidates)
    return counts.most_common(1)[0][0]


def extract_datetime(text: str) -> str:
    match = DATE_RE.search(text)
    if not match:
        return ""
    groups = [g for g in match.groups() if g]
    return " ".join(groups[:2])


def extract_discount(text: str) -> str:
    match = DISCOUNT_RE.search(text)
    if not match:
        return ""
    value = normalize_text(match.group(0).replace("−", "-"))
    percent = re.search(r"(\d{1,3})\s*%", value)
    if percent and int(percent.group(1)) > 99:
        return ""
    return value


def extract_special_symbols(text: str) -> str:
    lowered = text.lower()
    symbols = []
    for token, label in [("штуки", "Ш"), ("ш", "Ш"), ("лоток", "Л"), ("л", "Л"), ("короб", "К"), ("к", "К")]:
        if re.search(rf"(?<![а-яa-z]){re.escape(token)}(?![а-яa-z])", lowered):
            symbols.append(label)
    return ",".join(sorted(set(symbols))) if symbols else NO_VALUE


def extract_additional_info(text: str) -> str:
    lowered = text.lower()
    keywords = ["сухое", "полуслад", "номер на весах", "удачная упаковка", "шелфтокер", "топ 1000", "столото"]
    found = [kw for kw in keywords if kw in lowered]
    return "; ".join(found) if found else NO_VALUE


def parse_fields(
    text_blocks: list[dict],
    crop: np.ndarray,
    template_info: dict | None = None,
    qr_fields: dict | None = None,
    roi_text: dict | None = None,
) -> dict:
    qr_fields = qr_fields or {}
    roi_text = roi_text or {}
    text = blocks_to_text(text_blocks)
    zone_text = "\n".join(str(v) for v in roi_text.values() if v)
    all_text = "\n".join([text, zone_text])

    result = {field: "" for field in TAG_FIELDS}
    result["color"] = dominant_color(crop)
    mechanic = (template_info or {}).get("mechanic", "")

    prices = extract_ranked_prices_from_blocks(text_blocks)
    for price in extract_prices(all_text):
        if price not in prices:
            prices.append(price)
    prices = filter_visible_ocr_prices(prices)
    qr_prices = [qr_fields.get(k, "") for k in ["price1_qr", "price2_qr", "price3_qr", "price4_qr", "action_price_qr"] if qr_fields.get(k)]
    for price in qr_prices:
        if price not in prices:
            prices.append(str(price))

    assign_prices(result, prices, mechanic)
    if qr_fields.get("price1_qr"):
        result["price_default"] = str(qr_fields["price1_qr"])
    if qr_fields.get("price4_qr"):
        result["price_card"] = str(qr_fields["price4_qr"])
    if qr_fields.get("action_price_qr"):
        result["price_discount"] = str(qr_fields["action_price_qr"])
    else:
        result["price_discount"] = NO_VALUE

    zone_name = extract_product_name(zone_text)
    rough_name = extract_product_name(text)
    result["product_name"] = max([zone_name, rough_name], key=score_product_name) if (zone_name or rough_name) else ""
    result["barcode"] = choose_barcode(all_text, qr_fields, text_blocks=text_blocks)
    result["id_sku"] = choose_sku(all_text, result["barcode"])
    result["print_datetime"] = extract_datetime(all_text)
    code = CODE_RE.search(all_text)
    result["code"] = code.group(0) if code else NO_VALUE
    result["discount_amount"] = extract_discount(all_text)
    result["additional_info"] = extract_additional_info(all_text)
    result["special_symbols"] = extract_special_symbols(all_text)

    if mechanic == "regular" and not result["discount_amount"]:
        result["discount_amount"] = NO_VALUE
    if not result["discount_amount"] and mechanic in {"action", "sale", "threshold_from", "threshold_to", "BOGOF"}:
        result["discount_amount"] = ""
    elif not result["discount_amount"]:
        result["discount_amount"] = NO_VALUE

    for col in QR_COLUMNS:
        result[col] = str(qr_fields.get(col, "") or "")
    if result["barcode"] and not result["qr_code_barcode"]:
        # In the supplied templates the QR barcode payload is the same product
        # EAN. If QR decoding fails but OCR gives a checksum-valid EAN-13, keep
        # that value in both key columns.
        result["qr_code_barcode"] = result["barcode"]
    return result

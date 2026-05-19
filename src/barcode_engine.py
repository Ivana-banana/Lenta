from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from .utils import normalize_text


COMMON_EAN13_PREFIXES = {
    "20",
    "21",
    "22",
    "23",
    "24",
    "25",
    "26",
    "27",
    "28",
    "29",
    "30",
    "31",
    "32",
    "33",
    "34",
    "35",
    "36",
    "37",
    "38",
    "40",
    "41",
    "42",
    "43",
    "44",
    "45",
    "46",
    "47",
    "48",
    "49",
    "50",
    "54",
    "56",
    "57",
    "59",
    "64",
    "70",
    "73",
    "76",
    "80",
    "84",
    "87",
    "90",
}
PREFERRED_REPAIR_PREFIXES = {
    "30",
    "31",
    "32",
    "33",
    "34",
    "35",
    "36",
    "37",
    "40",
    "41",
    "42",
    "43",
    "46",
    "47",
    "48",
    "50",
    "80",
    "84",
}

OCR_DIGIT_TRANSLATION = str.maketrans(
    {
        "O": "0",
        "o": "0",
        "О": "0",
        "о": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "l": "1",
        "|": "1",
        "S": "5",
        "s": "5",
        "B": "8",
    }
)


@dataclass(frozen=True)
class BarcodeCandidate:
    barcode: str
    score: float
    raw: str
    source: str
    repaired: bool = False


def digits_from_ocr(text: str) -> str:
    return re.sub(r"\D", "", str(text or "").translate(OCR_DIGIT_TRANSLATION))


def ean13_check_digit(first_12_digits: str) -> str | None:
    if not re.fullmatch(r"\d{12}", first_12_digits or ""):
        return None
    nums = [int(ch) for ch in first_12_digits]
    checksum = (10 - ((sum(nums[0:12:2]) + 3 * sum(nums[1:12:2])) % 10)) % 10
    return str(checksum)


def is_ean13(digits: str) -> bool:
    if not re.fullmatch(r"\d{13}", digits or ""):
        return False
    return ean13_check_digit(digits[:12]) == digits[-1]


def has_plausible_prefix(digits: str) -> bool:
    return len(digits) >= 2 and digits[:2] in COMMON_EAN13_PREFIXES


def repair_prefix_score(digits: str) -> float:
    if len(digits) < 2:
        return 0.0
    if digits[:2] == "46":
        return 2.0
    if digits[:2] in PREFERRED_REPAIR_PREFIXES:
        return 1.0
    if digits[:2] in COMMON_EAN13_PREFIXES:
        return 0.25
    return 0.0


def _digit_runs(text: str) -> list[str]:
    translated = str(text or "").translate(OCR_DIGIT_TRANSLATION)
    runs = re.findall(r"(?:\d[\s.,:\-_/\\]*){8,18}", translated)
    out = []
    for run in runs:
        digits = re.sub(r"\D", "", run)
        if len(digits) >= 8:
            out.append(digits)
    compact = digits_from_ocr(translated)
    if len(compact) >= 8:
        out.append(compact)
    return out


def ean13_candidates_from_text(text: str, *, allow_repair: bool = False, source: str = "ocr", base_score: float = 1.0) -> list[BarcodeCandidate]:
    candidates: list[BarcodeCandidate] = []
    for digits in _digit_runs(text):
        seen_for_run: set[str] = set()
        if is_ean13(digits):
            seen_for_run.add(digits)
            candidates.append(BarcodeCandidate(digits, base_score + 3.0, text, source))

        if len(digits) == 14:
            for idx in range(2):
                shortened = digits[idx : idx + 13]
                if shortened not in seen_for_run and is_ean13(shortened):
                    seen_for_run.add(shortened)
                    candidates.append(BarcodeCandidate(shortened, base_score + 2.5, text, source))

        if len(digits) > 14:
            for idx in range(0, len(digits) - 12):
                window = digits[idx : idx + 13]
                if window not in seen_for_run and is_ean13(window):
                    seen_for_run.add(window)
                    candidates.append(BarcodeCandidate(window, base_score + 2.0, text, source))

        if allow_repair and len(digits) == 13 and has_plausible_prefix(digits):
            repaired = digits[:12] + (ean13_check_digit(digits[:12]) or "")
            if repaired != digits and repaired not in seen_for_run and is_ean13(repaired):
                candidates.append(BarcodeCandidate(repaired, base_score + 0.65, text, source, repaired=True))

        if allow_repair and len(digits) == 12:
            repaired_seen = set(seen_for_run)
            safe_repairs: list[tuple[str, float]] = []
            if digits.startswith(("6", "7", "8")):
                # Common OCR miss on local tags: the leading country digit "4"
                # is clipped or fused into the barcode guard.
                safe_repairs.append(("4" + digits, 1.65))
            for repaired, repair_bonus in safe_repairs:
                if repaired in repaired_seen or not is_ean13(repaired):
                    continue
                prefix_score = repair_prefix_score(repaired)
                if prefix_score < 1.0:
                    continue
                repaired_seen.add(repaired)
                candidates.append(
                    BarcodeCandidate(
                        repaired,
                        base_score + repair_bonus + prefix_score,
                        text,
                        source,
                        repaired=True,
                    )
                )
    return candidates


def ean13_candidates_from_blocks(blocks: list[dict], *, allow_repair: bool = False) -> list[BarcodeCandidate]:
    candidates: list[BarcodeCandidate] = []
    for block in blocks:
        text = normalize_text(block.get("text", ""))
        if not text:
            continue
        digit_count = sum(ch.isdigit() for ch in text)
        if digit_count < 8:
            continue
        confidence = float(block.get("confidence", 0.0) or 0.0)
        source = f"ocr_angle_{int(block.get('angle', 0) or 0) % 360}"
        candidates.extend(
            ean13_candidates_from_text(
                text,
                allow_repair=allow_repair,
                source=source,
                base_score=1.0 + confidence,
            )
        )
    return candidates


def choose_best_ean13(candidates: list[BarcodeCandidate]) -> str:
    if not candidates:
        return ""
    grouped: dict[str, list[BarcodeCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.barcode, []).append(candidate)

    def rank(barcode: str) -> tuple[float, int, float, int, str]:
        items = grouped[barcode]
        total_score = sum(item.score for item in items)
        unrepaired = sum(1 for item in items if not item.repaired)
        prefix = repair_prefix_score(barcode)
        plausible = 1 if has_plausible_prefix(barcode) else 0
        return (total_score, unrepaired, prefix, plausible, barcode)

    return max(grouped, key=rank)


def choose_best_ean13_from_values(values: list[str], *, allow_repair: bool = False, source: str = "value") -> str:
    candidates: list[BarcodeCandidate] = []
    counts = Counter(str(value or "") for value in values if str(value or "").strip())
    for value, count in counts.items():
        candidates.extend(
            ean13_candidates_from_text(
                value,
                allow_repair=allow_repair,
                source=source,
                base_score=float(count),
            )
        )
    return choose_best_ean13(candidates)

from __future__ import annotations

import json
import re
from urllib.parse import parse_qsl, urlparse

import cv2
import numpy as np

from .barcode_engine import choose_best_ean13_from_values
from .config import QR_COLUMNS
from .utils import rotate_image


ALIASES = {
    "barcode": "qr_code_barcode",
    "b": "qr_code_barcode",
    "price1": "price1_qr",
    "p1": "price1_qr",
    "price2": "price2_qr",
    "p2": "price2_qr",
    "price3": "price3_qr",
    "p3": "price3_qr",
    "price4": "price4_qr",
    "p4": "price4_qr",
    "wholesalelevel1count": "wholesale_level_1_count",
    "wl1c": "wholesale_level_1_count",
    "wholesalelevel1price": "wholesale_level_1_price",
    "wl1p": "wholesale_level_1_price",
    "wholesalelevel2count": "wholesale_level_2_count",
    "wl2c": "wholesale_level_2_count",
    "wholesalelevel2price": "wholesale_level_2_price",
    "wl2p": "wholesale_level_2_price",
    "actionprice": "action_price_qr",
    "ap": "action_price_qr",
    "actioncode": "action_code_qr",
    "ac": "action_code_qr",
}


def empty_qr_fields() -> dict:
    return {col: "" for col in QR_COLUMNS}


def normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(key).strip().lower())


def normalize_value(value) -> str:
    if value is None:
        return ""
    text = str(value).strip().strip('"').strip("'")
    if re.fullmatch(r"\d+[,.]\d+", text):
        text = text.replace(",", ".")
    return text


def _pairs_from_payload(payload: str) -> list[tuple[str, str]]:
    payload = payload.strip()
    if not payload:
        return []

    parsed = urlparse(payload)
    query = parsed.query if parsed.query else payload
    pairs = parse_qsl(query, keep_blank_values=True)
    if pairs:
        return pairs

    for candidate in [payload, payload.replace("'", '"')]:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return list(obj.items())
        except Exception:
            pass

    pairs = []
    for chunk in re.split(r"[&;|\n\r]+", payload):
        chunk = chunk.strip().strip("{}")
        if not chunk:
            continue
        if "=" in chunk:
            key, value = chunk.split("=", 1)
        elif ":" in chunk:
            key, value = chunk.split(":", 1)
        else:
            continue
        pairs.append((key.strip(), value.strip()))
    return pairs


def parse_qr_payload(payload: str) -> dict:
    fields = empty_qr_fields()
    payload = str(payload or "").strip()
    if not payload:
        return fields

    if re.fullmatch(r"\d{8,14}", payload):
        fields["qr_code_barcode"] = choose_best_ean13_from_values([payload], source="qr_payload")
        return fields

    for key, value in _pairs_from_payload(payload):
        mapped = ALIASES.get(normalize_key(key))
        if mapped:
            normalized = normalize_value(value)
            if mapped == "qr_code_barcode":
                normalized = choose_best_ean13_from_values([normalized], source="qr_payload")
            fields[mapped] = normalized
    return fields


class QRParser:
    def __init__(self) -> None:
        self.cv_detector = cv2.QRCodeDetector()
        self._pyzbar = None
        self._zxingcpp = None
        try:
            from pyzbar.pyzbar import decode as pyzbar_decode

            self._pyzbar = pyzbar_decode
        except Exception:
            self._pyzbar = None
        try:
            import zxingcpp

            self._zxingcpp = zxingcpp
        except Exception:
            self._zxingcpp = None

    def read(self, image: np.ndarray, mode: str = "fast") -> dict:
        payloads: list[str] = []
        barcode_payloads: list[str] = []

        for variant in self._variants(image, mode=mode):
            payloads.extend(self._read_cv2(variant))
            if payloads and mode == "fast":
                break
            if mode != "fast":
                payloads.extend(self._read_pyzbar(variant))
                zxing_texts, zxing_barcodes = self._read_zxing(variant)
                payloads.extend(zxing_texts)
                barcode_payloads.extend(zxing_barcodes)

        if not payloads and mode == "fast":
            # A short rotated zxing sweep is still cheap compared with OCR and
            # catches many EAN/barcode lines on side-facing shelf labels.
            for variant in self._variants(image, mode="fast"):
                zxing_texts, zxing_barcodes = self._read_zxing(variant)
                payloads.extend(zxing_texts)
                barcode_payloads.extend(zxing_barcodes)
                if payloads or barcode_payloads:
                    break

        fields = empty_qr_fields()
        for payload in payloads:
            parsed = parse_qr_payload(payload)
            for key, value in parsed.items():
                if value and not fields.get(key):
                    fields[key] = value

        fields["_raw_payloads"] = sorted(set(payloads))
        fields["_barcode_payloads"] = sorted(set(barcode_payloads))
        return fields

    def has_qr(self, image: np.ndarray) -> bool:
        try:
            ok, _ = self.cv_detector.detect(image)
            if ok:
                return True
        except Exception:
            pass
        if self._zxingcpp is not None:
            try:
                return any(str(r.format).upper().find("QR") >= 0 for r in self._zxingcpp.read_barcodes(image))
            except Exception:
                return False
        return False

    def _variants(self, image: np.ndarray, mode: str = "fast"):
        angles = (0, 270) if mode == "fast" else (0, 90, 180, 270)
        for angle in angles:
            rotated = rotate_image(image, angle)
            yield rotated
            if mode != "fast" and max(rotated.shape[:2]) < 700:
                yield cv2.resize(rotated, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    def _read_cv2(self, image: np.ndarray) -> list[str]:
        payloads: list[str] = []
        try:
            data, _, _ = self.cv_detector.detectAndDecode(image)
            if data:
                payloads.append(data)
        except Exception:
            pass
        try:
            retval, decoded_info, _, _ = self.cv_detector.detectAndDecodeMulti(image)
            if retval:
                payloads.extend([d for d in decoded_info if d])
        except Exception:
            pass
        return payloads

    def _read_pyzbar(self, image: np.ndarray) -> list[str]:
        if self._pyzbar is None:
            return []
        try:
            return [obj.data.decode("utf-8", errors="ignore") for obj in self._pyzbar(image)]
        except Exception:
            return []

    def _read_zxing(self, image: np.ndarray) -> tuple[list[str], list[str]]:
        if self._zxingcpp is None:
            return [], []
        qr_payloads: list[str] = []
        barcodes: list[str] = []
        try:
            for result in self._zxingcpp.read_barcodes(image):
                text = str(getattr(result, "text", "") or "")
                fmt = str(getattr(getattr(result, "format", ""), "name", getattr(result, "format", ""))).upper()
                if not text:
                    continue
                if "QR" in fmt:
                    qr_payloads.append(text)
                else:
                    barcodes.append(text)
        except Exception:
            pass
        return qr_payloads, barcodes

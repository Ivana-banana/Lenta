---
title: Lenta Price Tags OCR
emoji: 🏷️
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: 6.14.0
app_file: app.py
pinned: false
license: mit
short_description: Распознавание ценников с видео робота-инвентаризатора (хакатон Lenta Tech)
---

# Lenta Price Tags OCR

Решение хакатона **Lenta Tech Life Hack**: видео робота, едущего вдоль
стеллажей, превращается в CSV с одной строкой на каждый уникальный
обнаруженный ценник. Все компоненты работают локально, без облачных API.

## Архитектура

```
video  →  кадры (sample_fps)  →  кандидаты ценников
              ↓
       crops + classification (templates.yaml)
              ↓
   QR/barcode (zxing + OpenCV) | OCR (RapidOCR) | ROI OCR | regex parser
              ↓
       deduplication (barcode → qr_code_barcode → id_sku → spatial)
              ↓
                            CSV (29 колонок)
```

Главные модули в `src/`:

| Файл | Роль |
| --- | --- |
| `pipeline.py` | оркестратор: чтение видео, сэмплинг кадров, склейка результатов |
| `video_reader.py` | sampling кадров с FPS-контролем и фильтром по sharpness |
| `detector.py` | OpenCV-детектор ценников (color/contour/shape) и tight red/orange `color_price_panel_*` |
| `shelf_lanes.py` | spatial-fallback по полочным линиям |
| `template_classifier.py` | классификация формы ценника по `configs/templates.yaml` |
| `ocr_engine.py` | обёртка над RapidOCR (с возможностью замены) |
| `qr_parser.py` / `barcode_engine.py` | декодирование QR/EAN-13 с checksum-валидацией |
| `field_parser.py` | regex/geometry-парсер полей с ценника |
| `deduplication.py` | склейка дублей с приоритетом по barcode, затем spatial |
| `csv_writer.py` | финальный CSV в строгой 29-колоночной схеме |

## Веб-интерфейс

Запускается командой `python app.py`. Принимает видео, отдаёт CSV + статистику
+ галерею найденных crops. На HuggingFace Spaces поднимается автоматически.

## Локальный запуск

```bash
cd lenta-price-tags
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate      # Linux/Mac
pip install -r requirements.txt
python app.py                   # UI на http://127.0.0.1:7860
```

CLI-режим (без UI):

```bash
python -m src.pipeline \
    --video ../data/49_5/49_5.mp4 \
    --out outputs/result.csv \
    --sample-fps 1.0 \
    --max-ocr-candidates-per-frame 3 \
    --max-spatial-fallbacks-per-frame 20 \
    --full-frame-ocr-candidates-per-frame 12 \
    --max-roi-zones 1 \
    --roi-ocr-mode auto \
    --ocr-angles 0,270
```

Полезные флаги: `--debug` сохраняет crops/ROI overlays/OCR JSON в
`debug_outputs/<video>/`, `--max-frames N` ограничивает обработку для
быстрых проверок.

## Docker

```bash
docker build -t lenta-price-tags .
docker run --rm -p 7860:7860 -v "%cd%/..:/data" lenta-price-tags
```

## Evaluation

```bash
python scripts/evaluate.py \
    --pred outputs/result.csv \
    --ref ../data/49_5/49_5.csv \
    --out outputs/evaluation_report.md \
    --time-tolerance-ms 1500 \
    --iou-threshold 0.3
```

Скрипт считает:
- **Primary match** по `barcode` (ключ N1 от организаторов).
- **Spatial fallback** по `frame_timestamp` + IoU(bbox) (ключ N2).
- Заполненность ключевых полей.

## Образ результата (29 колонок)

`filename, product_name, price_default, price_card, price_discount, barcode,
discount_amount, id_sku, print_datetime, code, additional_info, color,
special_symbols, frame_timestamp, x_min, y_min, x_max, y_max,
qr_code_barcode, price1_qr, price2_qr, price3_qr, price4_qr,
wholesale_level_1_count, wholesale_level_1_price, wholesale_level_2_count,
wholesale_level_2_price, action_price_qr, action_code_qr`

Соглашение: отсутствующее на ценнике поле → `"нет"`, не распознанное → пусто.

## Ограничения текущего baseline

- **QR/barcode**: на 4K-видео с движущегося робота QR-коды получаются мелкими
  и смазанными; локальные декодеры (zxing-cpp, OpenCV QRCodeDetector) часто
  не справляются даже с GT-crops. Это основной bottleneck.
- **Поворот и блики**: OCR на повернутых/бликующих ценниках выдаёт фрагменты;
  template-aware ROI снижает шум, но не заменяет хорошее выпрямление.
- **Без обучения детектора** часть ценников пропускается, а часть товарных
  этикеток попадает в кандидаты.

## Идеи масштабирования

- Применить undistortion камеры робота (коэффициенты есть в
  `data/example_undistort.py` от организаторов) до OCR.
- Локальный YOLO/RT-DETR detector, дообученный по pseudo-labels из
  heuristic-кандидатов (см. `scripts/export_pseudolabels.py` и `colab/`).
- Super-resolution или multi-frame burst aggregation для QR.
- Multi-frame voting OCR по нескольким кадрам одного ценника до дедупликации.
- Конвертация моделей в `rknn` (int8) для edge-устройств — соответствует
  рекомендации организаторов.

## Соответствие условиям хакатона

- Никаких облачных API: всё локально (RapidOCR ONNX, zxing-cpp,
  OpenCV).
- Никакой ручной разметки: pseudo-labels строятся автоматически из
  цвето-контурных эвристик и QR-кандидатов.
- Лёгкость: на CPU поднимается без GPU; RapidOCR + ONNX занимают ~10MB.
- Воспроизводимость: чистый `requirements.txt`, Dockerfile, app.py для UI.

См. также `solution_notes.md` для журнала экспериментов и `analysis_report.md`
для текущей метрики на тестовых видео.

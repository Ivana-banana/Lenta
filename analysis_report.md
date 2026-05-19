# Analysis Report

## Входные файлы

В корне репозитория найдены PDF с постановкой задачи, две презентации по ценникам, три пары `mp4 + csv` и несколько unlabeled-видео.

Видео:

| file | resolution | fps | frames | duration |
|---|---:|---:|---:|---:|
| 25_12-20.mp4 | 3840x2160 | 19.578 | 823 | 42.04s |
| 25_12-20_unlabeled.mp4 | 3840x2160 | 19.964 | 865 | 43.33s |
| 26_12-20.mp4 | 3840x2160 | 19.980 | 1776 | 88.89s |
| 26_12-20_unlabeled.mp4 | 3840x2160 | 20.008 | 634 | 31.69s |
| 26_2-10_unlabeled.mp4 | 3840x2160 | 20.003 | 613 | 30.65s |
| 43_15.mp4 | 3840x2160 | 19.946 | 299 | 14.99s |

## CSV структура

Примеры CSV уже соответствуют требуемому формату результата: 29 колонок. В `26_12-20.csv` и `43_15.csv` есть опечатка `wholesale_level_1_coun`; в проекте она нормализуется в evaluation script к `wholesale_level_1_count`.

Уточнение по проверке от организаторов:

- главный ключ сопоставления с GT - `barcode`;
- если `barcode` отсутствует или не распознан, используется пространственно-временной ключ `frame_timestamp + bbox`;
- timestamp и bbox должны соответствовать кадру с лучшим качеством определения ценника;
- CSV должен быть стандартным UTF-8 с разделителем `,` и десятичной точкой.

Размеры примеров:

- `25_12-20.csv`: 57 строк.
- `26_12-20.csv`: 71 строка.
- `43_15.csv`: 29 строк.

Заполненность высокая по ключевым полям: `product_name`, `price_default`, `price_card`, `barcode`, координаты и цвет почти всегда заполнены. `print_datetime`, `id_sku`, `code` иногда отсутствуют. QR-поля часто есть в reference CSV, но на исходных crop QR очень мелкие и трудно читаются локальными декодерами.

## Документы

PDF фиксирует ограничения: локальное воспроизведение, без облачных API, без ручной разметки, итоговый CSV и UI.

`Расшифровка ценники.pptx` показывает расположение полей: название, скидка, QR, цены, id_sku, дата печати, barcode, код зоны, тип выкладки и дополнительная информация.

`ГМ для ТК.pptx` содержит template registry из 66 слайдов. Основные механики:

- regular
- action
- sale / распродажа
- threshold_from / цена от N/Х
- threshold_to / цена с ограничением до Х
- BOGOF / богоф
- MNC

Размеры и ориентации: 6x6, 6x12, A5, A4, A3, A2; square, horizontal, vertical. Единицы: 1kg, 100g, 1pc.

## Sample frames

Кадры и reference overlays сохранены в корневой папке:

- `debug_outputs/frames`
- `debug_outputs/reference_overlays`
- `debug_outputs/reference_crops`

На `25_12-20.mp4` видно, что ценники часто занимают 150-300 px по ширине на 4K-кадре, бывают повернуты, частично обрезаны, сняты через стекло и с бликами. QR-код визуально присутствует, но в sample crop OpenCV и zxing-cpp не смогли его стабильно декодировать.

## Сложности

- Мелкий текст и QR при 4K-сцене с большим количеством товара.
- Поворот ценников на 90 градусов и перспективные искажения.
- Блики стекла и сильные перепады освещения.
- Похожие по цвету товарные этикетки создают ложные candidates.
- Без ручной разметки baseline должен опираться на QR, цвет, геометрию, шаблоны и OCR.

## Вывод

Нужен воспроизводимый end-to-end baseline с хорошим debug-контуром. Текущее решение реализует template-aware pipeline и оставляет место для последующего локального ML detector без изменения внешнего интерфейса.

## Baseline run

Для `25_12-20.mp4` выполнен демонстрационный запуск:

```bash
python -m src.pipeline --video ../25_12-20.mp4 --out outputs/result_25_12-20.csv --debug --sample-fps 0.25 --max-ocr-candidates-per-frame 8
```

Статистика запуска:

- frames_processed: 11
- raw_detections: 470
- unique_price_tags: 13
- qr_decoded_count: 0
- ocr_success_count: 25
- ocr_engine: rapidocr

Качество baseline ограничено тем, что QR на crop не декодируется, а OCR на мелких повернутых ценниках часто дает частичные латинские/цифровые фрагменты. Это рабочий end-to-end каркас, но для соревновательного качества нужен следующий слой: rectification, агрегация нескольких кадров и локальный detector.
## 2026-05-15 update

New local files were added:

- `49_5.mp4`: 3840x2160, 19.16 fps, 481 frames, 25.10s.
- `49_5.csv`: 61 reference rows.
- `sample.csv`: 29 columns, same required output schema as `src.config.CSV_COLUMNS`.

Current scoring-oriented finding:

- `barcode` remains the best key, but local OCR/QR decoders often cannot recover it even from GT crops.
- The strongest practical gain comes from better spatial matching: `frame_timestamp + bbox`.
- Added shelf-lane spatial fallback: detects bright/cyan shelf rails and keeps strong tag candidates near them.
- Added full-frame OCR proposals: RapidOCR runs once on the full frame, finds large price text, and creates additional bbox proposals around these text blocks.
- OCR-derived EAN repair was made conservative: arbitrary checksum-valid repairs are rejected to avoid wrong primary keys.

Approximate evaluation snapshots with local `scripts/evaluate.py`:

| video | mode | matched / ref | notes |
|---|---:|---:|---|
| `25_12-20.mp4` | spatial lane, 1 fps, 12/frame | 37 / 57 | before full-frame OCR proposals |
| `25_12-20.mp4` | detector + full-frame OCR proposals | 41 / 57 | much better spatial recall than original OCR-only |
| `26_12-20.mp4` | spatial lane, 1 fps, 12/frame | 46 / 71 | good fit for shelf-lane prior |
| `43_15.mp4` | spatial lane, 1 fps, 12/frame | 16 / 29 | short video, moderate recall |
| `49_5.mp4` | spatial lane only | 7 / 61 | lane prior alone fails on new geometry |
| `49_5.mp4` | detector + full-frame OCR proposals | 17 / 61 | full-frame OCR helps new layout |

The result is still not final competition quality, but the project now has two complementary local proposal sources: OpenCV color/shape detector and OCR-text detector. This is the right base for the next jump: local pseudo-label detector or temporal tracking/super-resolution.

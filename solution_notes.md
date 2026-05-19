# Solution Notes

## 2026-05-16: проверенное текущее состояние

Это актуальный честный статус после повторной проверки локального Gradio CSV,
обновленного локального baseline и первого Colab YOLO эксперимента.

### Что уже попробовали

1. Первый локальный Gradio/baseline на `49_5`.
   - Проверенный файл: `outputs/evaluation_downloaded_gradio_49.md`.
   - Строк в предсказании: 357.
   - Barcode matches: 0.
   - Spatial fallback matches: 11 / 61.
   - Вывод: результат слишком слабый. Низкий `sample_fps` и шумный OCR/price
     seeding давали плохой CSV и мало совпадений по `bbox + timestamp`.

2. Обновленный локальный OpenCV baseline.
   - Добавлен tight detector `color_price_panel_*` для маленьких
     красно-оранжевых ценовых панелей на повернутых ценниках.
   - Усилен spatial fallback по shelf lanes и full-frame OCR proposals.
   - OCR price seeding сделан консервативным.
   - Угаданные OCR-цены больше не записываются в QR-поля.
   - Отфильтрованы явно шумные `product_name`.
   - Текущий проверенный CSV: `outputs/49_5_scoremode_sp20_ff12.csv`.
   - Evaluation: `outputs/evaluation_49_5_scoremode_sp20_ff12.md`.
   - Строк в предсказании: 645.
   - Barcode matches: 0.
   - Spatial fallback matches: 31 / 61.
   - Вывод: это текущий лучший локальный fallback по
     `frame_timestamp + bbox`, но barcode/QR он все еще не решает.

3. Первый Colab YOLO эксперимент.
   - Артефакты сохранены локально:
     `models/lenta_yolo_best.pt`,
     `outputs/colab_yolo_49.csv`,
     `outputs/evaluation_colab_yolo_49.md`.
   - Строк в предсказании: 338.
   - Barcode matches: 0.
   - Spatial fallback matches: 12 / 61.
   - Вывод: первая YOLO-модель пока не годится как замена OpenCV detector.
     Она обучалась на шумных pseudo-labels из старого pipeline snapshot и
     частично выучила ошибки генератора разметки.

4. Проверка QR/barcode.
   - Даже GT crops, вырезанные прямо по `49_5.csv`, не прочитались локальными
     OpenCV/zxing декодерами на проверенных примерах.
   - Вывод: главный bottleneck сейчас barcode/QR readability. Сначала нужны
     более стабильные и качественные crops, затем уже имеет смысл усиливать
     barcode recovery.

### Текущая лучшая команда

Быстрый score-oriented local mode:

```bash
python -m src.pipeline --video ../49_5.mp4 --out outputs/scoremode_49.csv --sample-fps 1 --max-ocr-candidates-per-frame 0 --max-spatial-fallbacks-per-frame 20 --full-frame-ocr-candidates-per-frame 12 --max-roi-zones 0 --roi-ocr-mode off
```

Режим с попыткой OCR:

```bash
python -m src.pipeline --video ../49_5.mp4 --out outputs/ocr_attempt_49.csv --sample-fps 1 --max-ocr-candidates-per-frame 3 --max-spatial-fallbacks-per-frame 20 --full-frame-ocr-candidates-per-frame 12 --max-roi-zones 1 --roi-ocr-mode auto --ocr-angles 0,270
```

OCR-режим медленнее и пока шумный. Его стоит использовать для исследований,
а не как самый безопасный submission mode.

### Что пробуем дальше

1. Пересобрать Colab bundle из последнего локального кода.
   - Причина: предыдущий Colab запуск использовал старый snapshot проекта.
   - В Colab обязательно должны попасть:
     `color_price_panel_*` detector, conservative price seeding, запрет на
     запись угаданных OCR-цен в QR-колонки.

2. Сгенерировать более чистые pseudo-labels перед новым YOLO training.
   - Предпочитать labels из `color_price_panel_*` и high-confidence color
     candidates.
   - Фильтровать labels по площади, aspect ratio, source и размеру bbox.
   - Не использовать reference CSV как training labels, если правила явно это
     не разрешают. CSV оставляем для evaluation и анализа.

3. Обучать YOLO как дополнительный candidate source, а не как полную замену.
   - Финальный detector должен быть ensemble:
     OpenCV price-panel candidates + shelf-lane candidates + optional YOLO.
   - YOLO включаем по умолчанию только если evaluation лучше текущего local
     baseline `31/61` на `49_5` и не ухудшает другие видео.

4. Добавить multi-frame aggregation/tracking.
   - Группировать близкие detections на соседних sampled frames.
   - Выбирать crop с лучшей sharpness/size для итогового timestamp/bbox.
   - Это соответствует уточнению организаторов: timestamp должен относиться к
     кадру лучшего окончательного определения ценника.

5. Вернуться к barcode после улучшения crop quality.
   - Пробовать de-rotation/perspective rectification.
   - Пробовать локальные upscale/super-resolution variants.
   - Запускать zxing/OpenCV/optional OCR digit extraction по barcode/QR zones.
   - Принимать barcode только если EAN-13 checksum валиден. Неверный barcode
     хуже пустого.

6. Контекст товара выше ценника оставляем optional.
   - Может помочь для `product_name`.
   - Нельзя использовать для barcode, price, id_sku и key fields: легко
     привязать соседний товар к неправильному ценнику.

### Формат работы дальше

- Локально: правки кода, быстрый evaluation, `solution_notes.md`, README,
  Gradio/CLI packaging.
- Colab: тяжелое YOLO training и GPU inference.
- После каждого Colab запуска сохраняем и возвращаем:
  `best.pt`, generated CSV, evaluation report и по возможности несколько debug
  bbox images.
- Decision rule:
  если новая модель не бьет локальный `31/61` spatial baseline на `49_5` и не
  улучшает другие видео, не включаем ее в default pipeline.

## 2026-05-15 Max-Score Direction

Update after first Colab YOLO run:

- User ran the Colab pseudo-label training and exported:
  `models/lenta_yolo_best.pt`, `outputs/colab_yolo_49.csv`,
  `outputs/evaluation_colab_yolo_49.md`.
- Result on `49_5`: 338 predicted rows, 0 barcode matches, 12 spatial
  matches out of 61 reference rows.
- This is worse than the updated local score-mode result (`31/61` spatial on
  the same video), so this first YOLO model should not replace the OpenCV
  detector yet.
- Reason: the first model was trained from noisy pseudo-labels generated by an
  older heuristic pipeline. It learned the noise instead of a clean price-tag
  shape prior.
- Also, the Colab CSV was generated with an older project snapshot that still
  allowed OCR seed prices to leak into QR price columns. Rebuild the Colab
  bundle before the next run.
- Next Colab direction: export cleaner pseudo-labels from the updated
  `color_price_panel_*` detector, filter labels by size/aspect/source, train on
  multiple videos, then evaluate YOLO as an additional candidate source rather
  than a full replacement.

Update after inspecting the local Gradio result for `49_5`:

- The downloaded Gradio CSV had 357 rows, 0 barcode matches, and only 11
  spatial matches against `49_5.csv`.
- The bad-looking price fields came from over-trusting noisy OCR seeds, so OCR
  price seeding is now conservative and never writes guessed values into QR
  columns.
- Added a tight `color_price_panel_*` detector pass. It searches small
  red/orange price panels without merging neighboring vertical tags into one
  tall contour.
- Detector recall on `49_5` improved from 40 to 60 possible IoU>=0.3 matches
  near sampled frames; top-12 recall improved from 18 to 35.
- Current stronger local command produced
  `outputs/49_5_scoremode_sp20_ff12.csv`: 26 frames, 1412 detections, 645
  output rows, 31 spatial matches, still 0 barcode/QR matches.
- Even GT crops from `49_5.csv` fail local QR/zxing decoding on sampled
  examples, so the main remaining bottleneck is barcode/QR readability, not
  only bbox detection.

New organizer data added `49_5.mp4`, `49_5.csv`, and `sample.csv`.
`sample.csv` confirms the same 29-column output schema.

What changed in code:

- Added `src/shelf_lanes.py` and integrated shelf-lane spatial fallback.
- Added optional full-frame OCR proposal generation:
  `--full-frame-ocr-candidates-per-frame N`.
- Gradio UI now exposes `max spatial fallback/frame` and
  `full-frame OCR proposals/frame`; defaults are aimed at higher recall.
- `barcode` repair is conservative: only checksum-valid, safe repairs survive.
- If a checksum-valid `barcode` is found but QR decoding fails,
  `qr_code_barcode` is filled with the same EAN because in the templates these
  fields refer to the same product code.

Current best local command:

```bash
python -m src.pipeline --video ../49_5.mp4 --out outputs/maxscore_49.csv --sample-fps 1 --max-ocr-candidates-per-frame 3 --max-spatial-fallbacks-per-frame 12 --full-frame-ocr-candidates-per-frame 12 --max-roi-zones 1 --roi-ocr-mode auto --ocr-angles 0,270
```

Honest status:

- This is much stronger than the first OCR-only baseline for spatial matching.
- It is still weak on true barcode extraction because local decoders fail on
  small/blurred QR/EAN crops.
- The next major quality jump should be local training/pseudo-labeling of a
  detector or multi-frame super-resolution/tracking, not more regex.

Experiments rejected after local evaluation:

- Adaptive lane/global split based on OCR-price lane coverage: plausible, but
  reduced matches on `49_5` and `25_12-20`.
- Same-frame NMS over final spatial records: reduced row count but also removed
  correct bboxes.
- Two bbox sizes per OCR-price block: improved isolated OCR-proposal recall, but
  worsened the integrated pipeline after deduplication.
- zxing-cpp on full frames: returned zero barcodes/QRs on checked frames.

# Solution Notes

## Актуальные правила проверки

Новая информация от организаторов меняет приоритеты пайплайна:

- Ключ N1 для сопоставления с GT: `barcode`.
- Если `barcode` распознан правильно, строка результата матчится с GT напрямую по нему.
- Ключ N2: пространственно-временное сопоставление по `frame_timestamp` и `bbox`.
- `frame_timestamp` и координаты нужно брать с кадра, где ценник определен лучше всего, а не обязательно с первого появления.
- Если строка не матчится ни по `barcode`, ни по пространственно-временному ключу, остальные поля уже не помогут.
- CSV: разделитель `,`, десятичный разделитель `.`, кодировка UTF-8, стандартное quoting для строк.

## Что это значит для архитектуры

Главный фокус качества теперь такой:

1. Максимально надежно распознавать `barcode`.
2. Если `barcode` не распознан, сохранять максимально точный `bbox` и timestamp лучшего кадра.
3. Дедупликация должна выбирать representative record по качеству OCR/crop, чтобы координаты и timestamp соответствовали лучшему наблюдению ценника.
4. Лучше оставить `barcode` пустым, чем записать неправильный OCR-barcode: неправильный ключ может увести строку к чужому GT.

## Текущее состояние реализации

Сейчас pipeline делает:

- sampling видео через OpenCV;
- поиск кандидатов ценников через QR/color/contours;
- crop ценника;
- local QR/barcode decoding через OpenCV/zxing-cpp, без облака;
- local OCR через RapidOCR;
- template-aware classification по `configs/templates.yaml`;
- ROI OCR по шаблонам;
- regex/block-geometry parsing полей;
- дедупликацию с приоритетом `barcode`, затем `qr_code_barcode`, затем `id_sku`, затем fallback;
- CSV строго в нужных колонках.

Важные последние изменения:

- OCR-barcode из текста теперь принимается строже: валидный EAN-13 предпочтительнее malformed partial code.
- Добавлен отдельный `src/barcode_engine.py`: EAN-13 checksum, OCR digit normalization и выбор лучшего barcode-кандидата.
- CSV writer переключен на чистый `utf-8`.
- Цена больше не собирается из единиц объема/веса вроде `0.25L`.
- Дедупликация внутри группы собирает лучшие поля по отдельности, а не выбирает одну строку целиком.
- `frame_timestamp`/bbox итоговой строки остаются от best record группы.
- `quality_score` теперь штрафует сильно раздутые bbox, потому что spatial fallback зависит от IoU с GT.
- `scripts/evaluate.py` теперь считает не только пересечение barcode/id, но и приближенный `barcode primary matches + spatial fallback matches`.
- Detector-only spatial fallback добавлен как экспериментальный CLI-флаг `--max-spatial-fallbacks-per-frame`, но выключен по умолчанию: быстрый тест показал, что наивное добавление top-detector bbox дает много false positives.
- Глубокий barcode ROI OCR с x3/threshold/multi-angle протестирован: прироста на `25_12-20` не дал, зато сильно замедлил прогон. По умолчанию отключен; текущий лучший рычаг - detector ranking + больше sampled frames.

## Текущие проблемы качества

- QR на видео очень мелкий и локальные декодеры часто не могут его прочитать.
- OCR на повернутых и бликующих ценниках часто дает фрагменты вместо полного `barcode`.
- Детектор без обучения все еще путает ценники и товарные этикетки.
- При низком `sample_fps` легко промахнуться мимо лучшего кадра, поэтому качество `bbox/timestamp` резко падает.

## Ближайший план улучшений

1. Усилить barcode-first стратегию:
   - OCR только barcode-зоны с несколькими preprocessing variants;
   - EAN-13 checksum voting по нескольким кадрам;
   - восстановление OCR-ошибок только если checksum подтверждается.
2. Улучшить spatial-temporal fallback:
   - хранить несколько наблюдений одного кандидата;
   - выбирать кадр с максимальным `quality_score`;
   - не усреднять bbox, а брать bbox лучшего crop.
3. Добавить crop rectification:
   - искать четыре угла ценника;
   - выпрямлять perspective перед OCR/barcode.
4. Аккуратно добавить context OCR товара выше ценника:
   - использовать только как вспомогательный источник `product_name`;
   - не использовать context для `barcode`, цен и ключевых полей.
5. Для качества выше baseline рассмотреть локальный detector без ручной разметки:
   - pseudo-labels из текущих heuristics/reference-free candidates;
   - легкий YOLO/RT-DETR как optional модуль.

## Рекомендуемые команды

Быстрый preview:

```bash
python -m src.pipeline --video ../25_12-20.mp4 --out outputs/preview.csv --sample-fps 0.2 --max-frames 3 --max-ocr-candidates-per-frame 8 --max-roi-zones 1
```

Компромисс recall/speed:

```bash
python -m src.pipeline --video ../25_12-20.mp4 --out outputs/result_improved_recall_25_12-20.csv --sample-fps 0.2 --max-ocr-candidates-per-frame 20 --max-roi-zones 1 --roi-ocr-mode auto --ocr-angles 0,270
```

Экспериментальный spatial fallback:

```bash
python -m src.pipeline --video ../25_12-20.mp4 --out outputs/result_spatial_exp.csv --sample-fps 0.5 --max-ocr-candidates-per-frame 12 --max-spatial-fallbacks-per-frame 2
```

Evaluation по примеру:

```bash
python scripts/evaluate.py --pred outputs/result_improved_recall_25_12-20.csv --ref ../25_12-20.csv --out outputs/evaluation_result_improved_recall_25_12-20.md --time-tolerance-ms 1500 --iou-threshold 0.3
```

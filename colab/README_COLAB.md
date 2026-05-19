# Lenta Price Tags Colab Bundle

This folder is a ready-to-upload package for GPU experiments.

## What is inside

- `project/` - current reproducible pipeline code.
- `data/` - videos, CSV references/sample and task documents copied from the local workspace.
- `notebooks/lenta_yolo_pseudolabel_colab.ipynb` - Colab notebook for automatic pseudo-label export and YOLO training.
- `data_manifest.csv` - file sizes for quick upload checks.
- `requirements_colab.txt` - Colab-side packages.

## When Colab is useful

Local CPU is enough for the baseline and pseudo-label export. Colab/GPU is useful for:

- training a local detector from automatic pseudo-labels;
- testing heavier OCR/detection models;
- exporting `best.pt` weights and running them locally with `--yolo-model`.

This path stays within the hackathon rules: no cloud OCR/API, no manual labeling.

## Colab flow

1. Upload `colab_bundle` to Google Drive or zip and upload it to Colab.
2. Open `notebooks/lenta_yolo_pseudolabel_colab.ipynb`.
3. Run cells top to bottom:
   - install dependencies;
   - export pseudo-labels from videos;
   - train YOLO;
   - run inference/evaluation;
   - download `best.pt`.
4. Put `best.pt` into local `lenta-price-tags/models/`.
5. Run local pipeline with:

```bash
python -m src.pipeline --video ../49_5.mp4 --out outputs/yolo_49.csv --sample-fps 1 --yolo-model models/best.pt --max-ocr-candidates-per-frame 3 --max-spatial-fallbacks-per-frame 12 --full-frame-ocr-candidates-per-frame 12 --max-roi-zones 1
```


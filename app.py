from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import pandas as pd

from src.config import OUTPUT_DIR
from src.pipeline import process_video

IS_HF_SPACE = bool(os.environ.get("SPACE_ID"))


def run(
    video_file,
    sample_fps: float,
    max_ocr_candidates: int,
    max_spatial_fallbacks: int,
    full_frame_ocr_candidates: int,
    max_roi_zones: int,
    max_frames: int,
    debug: bool,
    progress=gr.Progress(),
):
    if video_file is None:
        return pd.DataFrame(), None, "Загрузите видео.", []
    video_path = Path(video_file)
    out_path = OUTPUT_DIR / f"{video_path.stem}_gradio_result.csv"

    def progress_callback(done: int, total: int, message: str) -> None:
        if total > 0:
            progress(min(done / total, 1.0), desc=f"{message}: {done}/{total}")

    csv_path, stats = process_video(
        video_path,
        out_path,
        debug=debug,
        sample_fps=sample_fps,
        max_frames=max_frames if max_frames > 0 else None,
        max_ocr_candidates_per_frame=max_ocr_candidates,
        max_spatial_fallbacks_per_frame=max_spatial_fallbacks,
        full_frame_ocr_candidates_per_frame=full_frame_ocr_candidates,
        max_roi_zones=max_roi_zones,
        roi_ocr_mode="auto",
        qr_mode="fast",
        ocr_angles=[0, 270],
        debug_max_frames=12,
        debug_max_crops=80,
        debug_max_json=80,
        progress_callback=progress_callback,
    )
    df = pd.read_csv(csv_path)
    stats_text = "\n".join(f"{k}: {v}" for k, v in stats.items() if k != "metadata")
    crops = []
    if debug:
        crop_dir = Path("debug_outputs") / video_path.stem / "crops"
        crops = [str(p) for p in sorted(crop_dir.glob("*.jpg"))[:24]]
    return df, str(csv_path), stats_text, crops


INTRO_MD = """
# Lenta Price Tags OCR

Прототип решения хакатона **Lenta Tech Life Hack** для автоматического
распознавания ценников с видео робота-инвентаризатора. Полностью локальная
обработка: OpenCV-детектор, RapidOCR, zxing-cpp, без облачных API.

**Как пользоваться:**
1. Загрузите видео (рекомендуем короткий клип 5–20 секунд для онлайн-демо).
2. Подкрутите `sample_fps` (1.0 — баланс качества и скорости).
3. Нажмите «Распознать ценники». Через несколько минут получите CSV
   с 29 полями согласно ТЗ организаторов.

На HF Spaces CPU обработка занимает заметное время; для полноценного
прогона видео используйте локальную установку (см. README репозитория).
""".strip()

EXAMPLES_DIR = Path(__file__).parent / "examples"


def _gather_examples() -> list[list]:
    if not EXAMPLES_DIR.exists():
        return []
    items: list[list] = []
    for path in sorted(EXAMPLES_DIR.glob("*.mp4")):
        items.append([str(path), 1.0, 3, 20, 12, 1, 20, False])
    return items


with gr.Blocks(title="Lenta Price Tags OCR") as demo:
    gr.Markdown(INTRO_MD)
    with gr.Row():
        video = gr.Video(label="Видео робота")
        with gr.Column():
            sample_fps = gr.Slider(0.2, 3.0, value=1.0, step=0.2, label="sample_fps")
            max_ocr_candidates = gr.Slider(0, 30, value=3, step=1, label="max OCR candidates/frame")
            max_spatial_fallbacks = gr.Slider(0, 30, value=20, step=1, label="max spatial fallback/frame")
            full_frame_ocr_candidates = gr.Slider(0, 30, value=12, step=1, label="full-frame OCR proposals/frame")
            max_roi_zones = gr.Slider(0, 6, value=1, step=1, label="max ROI zones/candidate")
            max_frames = gr.Slider(
                0, 200, value=20 if IS_HF_SPACE else 0, step=1,
                label="max sampled frames (0 = весь видеоряд; на онлайн-демо лучше 10-30)",
            )
            debug = gr.Checkbox(value=False, label="debug mode")
            button = gr.Button("Распознать ценники", variant="primary")
            csv_file = gr.File(label="Скачать CSV")
    stats = gr.Textbox(label="Статистика", lines=10)
    table = gr.Dataframe(label="Результат")
    gallery = gr.Gallery(label="Найденные crop-примеры", columns=4, height=420)

    examples = _gather_examples()
    if examples:
        gr.Examples(
            examples=examples,
            inputs=[video, sample_fps, max_ocr_candidates, max_spatial_fallbacks,
                    full_frame_ocr_candidates, max_roi_zones, max_frames, debug],
            label="Примеры (клик — загрузка)",
        )

    button.click(
        run,
        inputs=[video, sample_fps, max_ocr_candidates, max_spatial_fallbacks,
                full_frame_ocr_candidates, max_roi_zones, max_frames, debug],
        outputs=[table, csv_file, stats, gallery],
    )


if __name__ == "__main__":
    import time

    demo.queue().launch(
        server_name="0.0.0.0" if IS_HF_SPACE else "127.0.0.1",
        server_port=int(os.environ.get("PORT", 7860)),
        show_error=True,
        # on HF Spaces .launch() must block; locally prevent_thread_lock avoids
        # a Gradio self-check that flakes on some Windows setups
        prevent_thread_lock=not IS_HF_SPACE,
    )
    if not IS_HF_SPACE:
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass

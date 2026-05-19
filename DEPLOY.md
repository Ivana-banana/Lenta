# Деплой на Hugging Face Spaces

Инструкция, чтобы поднять UI публично на HF Spaces за 5 минут.
Это даёт бонус к оценке хакатона (явно прописано в ТЗ).

## 1. Подготовка аккаунта (один раз)

1. Зарегистрируйся на https://huggingface.co (бесплатно).
2. Открой https://huggingface.co/settings/tokens и создай токен с правом
   **write** (например, имя `lenta-deploy`). Скопируй его — больше HF его
   не покажет.

## 2. Создать пустой Space

1. Перейди https://huggingface.co/new-space
2. Параметры:
   - **Owner**: твой ник.
   - **Space name**: `lenta-price-tags` (или другое).
   - **License**: MIT.
   - **SDK**: **Gradio**.
   - **Hardware**: **CPU basic** (бесплатно, 2 vCPU / 16GB).
   - **Visibility**: Public.
3. Жми `Create Space`. HF создаст git-репозиторий вида
   `https://huggingface.co/spaces/<твой-ник>/lenta-price-tags`.

## 3. Запушить код

Из папки `lenta-price-tags/`:

```powershell
# первый раз нужен git, проверь: git --version
git init
git lfs install                          # если есть лимит на большие файлы
git remote add hf https://huggingface.co/spaces/<твой-ник>/lenta-price-tags
git add .
git commit -m "Initial deploy"
git branch -M main

# при push HF спросит логин/пароль:
#  - User: твой ник на HF
#  - Password: токен с шага 1 (write)
git push hf main
```

Альтернатива через CLI HuggingFace (удобнее, токен сохраняется):

```powershell
pip install -U huggingface_hub
huggingface-cli login          # вставь токен
huggingface-cli repo create lenta-price-tags --type space --space_sdk gradio
git clone https://huggingface.co/spaces/<твой-ник>/lenta-price-tags hf-space
# скопируй файлы из lenta-price-tags/ в hf-space/
cd hf-space
git add . && git commit -m "Initial deploy" && git push
```

## 4. Проверка

1. Открой `https://huggingface.co/spaces/<твой-ник>/lenta-price-tags`.
2. HF автоматически:
   - прочитает frontmatter из `README.md` (sdk=gradio, app_file=app.py);
   - установит зависимости из `requirements.txt`;
   - запустит `python app.py`.
3. Первая сборка занимает 3–8 минут (видны логи во вкладке `Logs`).

## 5. Возможные грабли

- **Out of memory / Killed**: онлайн CPU имеет 16GB RAM, но 4K-видео + RapidOCR
  могут разогнать пик. Решение — в UI ограничить `max sampled frames` (уже
  стоит дефолт 20 для онлайн-режима).
- **Slow first run**: RapidOCR качает ~10MB ONNX-моделей при старте.
  Это разово.
- **Slow processing of full 4K video**: на CPU 4K кадр обрабатывается
  ~2-5 секунд. Для онлайн-демо рекомендуй жюри короткое видео или малый
  `sample_fps`.
- **Push отклонён по размеру**: убедись что `.gitignore` исключает
  `outputs/`, `debug_outputs/`, `*.mp4` (кроме `examples/`).

## 6. Ссылка для презентации

После успешного запуска у Space будет:
- **Live URL**: `https://<твой-ник>-lenta-price-tags.hf.space`
- **Repo URL** (для презентации): `https://huggingface.co/spaces/<твой-ник>/lenta-price-tags`

Обе ссылки положи на финальный слайд презентации.

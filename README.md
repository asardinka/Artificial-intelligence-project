# Сегментация и классификация опухолей мозга по МРТ-снимкам

Проект в рабочем состоянии: подготовка данных, обучение, HTTP API, CLI и Docker-образ для инференса.

## Задача

Построить многозадачную систему, которая по одному МРТ-срезу выполняет **сегментацию** опухоли и **классификацию** её типа (плоскости **ax**, **sa**, **co** задаются отдельно).

## Датасет

**BRISC 2025** — [kaggle.com/datasets/briscdataset/brisc2025](https://www.kaggle.com/datasets/briscdataset/brisc2025)

6 000 контрастно-усиленных T1-взвешенных МРТ-срезов (5 000 train / 1 000 test), четыре класса: глиома, менингиома, опухоль гипофиза, норма. Для каждого среза есть попиксельная маска. Данные сбалансированы по классам и представлены в трёх анатомических плоскостях.

---

## Первый запуск после клонирования (Git)

Все команды ниже выполняются из каталога `**AIP`** (корень репозитория: там `pyproject.toml` и папка `src`). Если открыть терминал в родительской папке, импорт `src.*` не сработает.

### Без uv (Python + pip)

```bash
cd AIP
python -m venv .venv
```

Активация окружения:

- **Windows (cmd):** `.venv\Scripts\activate.bat`
- **Windows (PowerShell):** `.\.venv\Scripts\Activate.ps1`
- **Linux / macOS:** `source .venv/bin/activate`

Установка пакета в режиме разработки (инференс и API):

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

Для **скачивания датасета** (`data_load.py`), ноутбуков и полного обучения нужны дополнительные пакеты:

```bash
python -m pip install -e ".[train]"
```

### С uv

```bash
cd AIP
uv sync
uv sync --extra train
```

(`--extra train` — по желанию, для данных и тяжёлых зависимостей обучения.)

---

## Чекпоинты и переменная `ARTIFACTS_DIR`

Инференс ожидает **шесть файлов** в каталоге артефактов (по одной паре моделей на каждую плоскость):


| Задача        | Файлы                                                                                    |
| ------------- | ---------------------------------------------------------------------------------------- |
| Классификация | `classification_ax_model.pt`, `classification_sa_model.pt`, `classification_co_model.pt` |
| Сегментация   | `segmentation_ax_model.pt`, `segmentation_sa_model.pt`, `segmentation_co_model.pt`       |


По умолчанию используется каталог `**artifacts/`** рядом с проектом (путь считается от `src/config.py`, не от текущей папки терминала).

Чтобы указать другой каталог (например, на сервере), задайте абсолютный путь:

```bash
export ARTIFACTS_DIR=/path/to/weights   # Linux / macOS
set ARTIFACTS_DIR=D:\weights            # Windows cmd
$env:ARTIFACTS_DIR = "D:\weights"       # PowerShell
```

Для сравнения с первой версией пайплайна можно хранить старые веса под теми же именами в `**artifacts/v1/**` (`OUTPUT_DIR_V1` в `src/config.py`, см. `test.ipynb`).

---

## HTTP API (инференс)

### Смена порта

Параметр `**--port**` у uvicorn:

```bash
python -m uvicorn src.tumor_app.api:app --host 127.0.0.1 --port 8080
```

В `**run-api.bat**` и `**run-api.ps1**` замените `8000` на нужный порт.

В **Docker** измените проброс в `docker-compose.yml`, например `"8080:8080"`, и в `Dockerfile` в `CMD` укажите `--port 8080` (и при необходимости `EXPOSE`).

### Локальная разработка (Windows)

- Дважды щёлкнуть `**run-api.bat`** (переходит в папку скрипта и запускает uvicorn).
- Или в **PowerShell** из `AIP`: `**.\run-api.ps1`** (без `.\` PowerShell не запускает скрипт из текущей папки).
- Или вручную с автоперезагрузкой:

```bash
python -m uvicorn src.tumor_app.api:app --reload --host 127.0.0.1 --port 8000
```

С **uv**:

```bash
uv run uvicorn src.tumor_app.api:app --reload --host 127.0.0.1 --port 8000
```

### Linux-сервер (продакшен)

Режим `**--reload**` для продакшена не используют. Слушать все интерфейсы:

```bash
cd /path/to/AIP
python -m uvicorn src.tumor_app.api:app --host 0.0.0.0 --port 8000
```

Перед этим: `pip install -e .`, вынести веса в `artifacts/` или задать `ARTIFACTS_DIR`. Обычно перед приложением ставят **nginx** / **Caddy** (HTTPS, лимит размера тела для `POST /predict`).

### Docker

Из каталога `AIP`, при условии что в `**./artifacts`** лежат все шесть `.pt`:

```bash
docker compose up --build
```

По умолчанию контейнер слушает **8000**. Образ собирает **CPU-сборки** PyTorch. Порт на хосте меняется в `docker-compose.yml` в секции `ports`.

Страница в браузере: `http://127.0.0.1:8000/` (или ваш хост и порт). Документация API: `/docs`.

- `GET /health` — проверка сервиса  
- `POST /predict` — `multipart/form-data`: поле `file` (изображение), поле `plane` (`ax` / `sa` / `co`, по умолчанию `ax`). Ответ JSON: `predicted_class`, `plane`, `image_original_b64`, `image_overlay_b64` (PNG в base64).

---

## CLI (Typer)

Из каталога `AIP`:

```bash
python -m src.tumor_app.cli predict path/to/image.jpg --plane ax --out-dir out
```

С uv:

```bash
uv run tumor-cli predict path/to/image.jpg --plane ax --out-dir out
```

В `out/` сохраняются `original.png` и `overlay.png`.

---

## Обучение своих моделей

1. **Данные**
  - Скрипт `**data_load.py`** или ноутбук `**data_load.ipynb**`: скачивание BRISC и разложение по структуре `data/classification_task/...` и `data/segmentation_task/...`.  
  - Нужны зависимости `**[train]**` и доступ к Kaggle (как в `kagglehub`).
2. **Архитектура и гиперпараметры**
  - Модели: `**src/models/`** (`classification.py`, `segmentation.py`).  
  - Эпохи, размеры, батчи, пути к данным: `**src/config.py**`.
3. **Запуск обучения**
  Точка входа — `src/training/train_models.py`; сами циклы обучения — `src/training/classification.py` и `src/training/segmentation.py` (предзагрузка данных в тензоры, без DataLoader).

Из корня `AIP`:

```bash
python -m src.training.train_models
```

или после установки пакета:

```bash
aip-train
```

С uv:

```bash
uv run python -m src.training.train_models
uv run aip-train
```

Чекпоинты по умолчанию пишутся в `**artifacts/**` (`OUTPUT_DIR`). Логи — в `**artifacts/logs/logsN.log**`.

**Замечание:** в `train_models.py` вызов **классификации** по умолчанию закомментирован; раскомментируйте блок `train_classification_task(...)`, чтобы обучать обе задачи для всех плоскостей. Иначе в `artifacts/` нужно положить уже готовые `classification_*_model.pt` или обучить классификацию отдельно.

Для сравнения с **первой версией** архитектур (`SmallResNet` + `SimpleUNet`, см. `src/models/legacy_aip_v1.py`) сохраняйте такие веса под теми же именами файлов в `**artifacts/v1/`** (`OUTPUT_DIR_V1` в `src/config.py`).

---

## Как подключить новые обученные модели

1. **Те же архитектура и формат чекпоинта**
  Положите файлы с **теми же именами** в каталог артефактов (или в каталог, на который указывает `ARTIFACTS_DIR`). Инференс читает их в `**src/tumor_app/infer.py`** через `load_checkpoint` и классы из `**src/models/**`.
2. **Другая архитектура или число классов**
  Нужно согласовать: конструктор модели и загрузка `state_dict` в `**infer.py`**, при необходимости поля в чекпоинте (`classes`, `norm_type` и т.д.) и константу `**CLASSES**` в `**src/config.py**`.
3. **Другие имена файлов**
  Имена заданы в `**src/config.py`** (`classification_output_filename`, `segmentation_output_filename`) и должны совпадать с тем, что ожидает `**infer.py**` (`classification_{plane}_model.pt` и `segmentation_{plane}_model.pt`).

---

## Структура проекта


| Путь                              | Назначение                              |
| --------------------------------- | --------------------------------------- |
| `src/tumor_app/api.py`            | FastAPI: `/`, `/health`, `/predict`     |
| `src/tumor_app/static/index.html` | Веб-форма                               |
| `src/tumor_app/infer.py`          | Загрузка моделей, предсказание          |
| `src/tumor_app/cli.py`            | `tumor-cli predict`                     |
| `src/training/train_models.py`    | Точка входа обучения по всем плоскостям |
| `src/training/classification.py`  | Обучение классификации                  |
| `src/training/segmentation.py`    | Обучение сегментации                    |
| `src/config.py`                   | Пути, гиперпараметры, имена чекпоинтов  |
| `src/models/`                     | Архитектуры сетей                       |
| `data_load.py`, `data_load.ipynb` | Загрузка и подготовка датасета          |


Источник зависимостей для приложения — `pyproject.toml`.

---

## Если что-то не запускается


| Симптом                             | Что сделать                                                                                     |
| ----------------------------------- | ----------------------------------------------------------------------------------------------- |
| `uv` не найден                      | Используйте `python -m pip install -e .` и `python -m uvicorn …` из **AIP**.                    |
| `No module named 'src'`             | Выполните `cd` в **AIP** и `pip install -e .`.                                                  |
| `run-api.ps1` не находится          | В PowerShell: `**.\run-api.ps1`**, не `run-api.ps1`.                                            |
| `Нет чекпоинта` / 503 на `/predict` | В `artifacts/` должны быть все шесть `.pt` для нужных плоскостей, либо задайте `ARTIFACTS_DIR`. |
| Порт занят                          | Укажите другой `--port` (см. выше).                                                             |



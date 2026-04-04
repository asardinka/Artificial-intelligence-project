# Сегментация и классификация опухолей мозга по МРТ

По одному срезу: **классификация** типа опухоли и **сегментация** (плоскости **ax** / **sa** / **co**). Датасет обучения: [BRISC 2025](https://www.kaggle.com/datasets/briscdataset/brisc2025).

## Запуск (Windows, PowerShell)

Все команды — из **корня репозитория** (рядом `pyproject.toml` и папка `src`).

```powershell
git clone https://github.com/asardinka/Artificial-intelligence-project
cd Artificial-intelligence-project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m uvicorn src.tumor_app.api:app --host 127.0.0.1 --port 8000
```

Браузер: [http://127.0.0.1:8000/](http://127.0.0.1:8000/) — форма; [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) — OpenAPI.

Если `Activate.ps1` блокируется политикой выполнения: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` или используйте **cmd** и `.venv\Scripts\activate.bat`. Без активации venv можно вызывать `.\.venv\Scripts\python.exe -m pip ...` и `.\.venv\Scripts\python.exe -m uvicorn ...`.

**Linux / macOS:** вместо активации PowerShell — `source .venv/bin/activate`, остальные шаги те же.

**Доступ с других машин в сети:** `--host 0.0.0.0` вместо `127.0.0.1`. **Другой порт:** добавьте, например, `--port 8080`.

## Веса моделей

В каталоге `**artifacts/`** должны лежать шесть чекпоинтов:  
`classification_ax_model.pt`, `classification_sa_model.pt`, `classification_co_model.pt`,  
`segmentation_ax_model.pt`, `segmentation_sa_model.pt`, `segmentation_co_model.pt`.

Другой каталог — переменная окружения `**ARTIFACTS_DIR**` (абсолютный путь). Путь к `artifacts/` по умолчанию считается от расположения `src/config.py`, не от текущей папки в терминале.

Старые архитектуры для сравнения — `**artifacts/v1/**` и `src/models/legacy_aip_v1.py`.

## Обучение и данные

```powershell
python -m pip install -e ".[train]"
```

Данные: `data_load.py` / `data_load.ipynb`. Обучение: `python -m src.training.train_models` (или `aip-train`). Гиперпараметры и пути — `src/config.py`, сети — `src/models/`. В `train_models.py` вызов классификации может быть закомментирован — тогда нужны уже обученные `classification_*_model.pt`.

## Прочее

- **CLI:** `python -m src.tumor_app.cli predict путь\к.jpg --plane ax --out-dir out`
- **Docker:** из корня репозитория `docker compose up --build` (веса монтируются в `./artifacts`).
- `**run-api.bat`** / `**run-api.ps1**` — запуск uvicorn без ручного ввода; в PowerShell скрипт: `.\run-api.ps1`.
- Зависимости описаны в `**pyproject.toml**`.


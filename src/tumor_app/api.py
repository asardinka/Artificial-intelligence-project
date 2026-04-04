from __future__ import annotations

import base64
from pathlib import Path
from typing import cast

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.tumor_app.infer import Plane, predict_from_bytes

app = FastAPI(
    title="AIP Tumor inference",
    version="0.1.0",
    description="Классификация и сегментация опухоли по МРТ-срезу (чекпоинты в ARTIFACTS_DIR).",
    docs_url="/docs",
    redoc_url=None,
)

_STATIC = Path(__file__).resolve().parent / "static" / "index.html"


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(_STATIC, media_type="text/html; charset=utf-8")


class PredictResponse(BaseModel):
    predicted_class: str = Field(..., description="Предсказанный класс")
    plane: str = Field(..., description="Использованная плоскость ax|sa|co")
    image_original_b64: str = Field(..., description="Исходное изображение, PNG base64")
    image_overlay_b64: str = Field(..., description="Подсветка области (сегментация), PNG base64")


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok", "service": "aip-tumor", "version": "0.1.0"}


@app.post("/predict", response_model=PredictResponse, tags=["predict"])
async def predict(
    file: UploadFile = File(..., description="Изображение (JPG/PNG и т.д.)"),
    plane: str = Form("ax", description="Плоскость: ax, sa, co"),
) -> PredictResponse:
    pl = plane.lower().strip()
    if pl not in ("ax", "sa", "co"):
        raise HTTPException(status_code=400, detail="plane должен быть ax, sa или co")

    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Пустой файл")
        result = predict_from_bytes(data, cast(Plane, pl))
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except OSError as e:
        raise HTTPException(status_code=400, detail=f"Не удалось прочитать изображение: {e}") from e

    return PredictResponse(
        predicted_class=result.predicted_class,
        plane=result.plane,
        image_original_b64=base64.b64encode(result.original_png).decode("ascii"),
        image_overlay_b64=base64.b64encode(result.overlay_png).decode("ascii"),
    )

FROM python:3.12-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir "fastapi>=0.115.0" "numpy>=1.26.0" "pillow>=10.0.0" "python-multipart>=0.0.9" "typer>=0.12.0" "uvicorn[standard]>=0.30.0" \
    && pip install --no-cache-dir --no-deps .

ENV ARTIFACTS_DIR=/app/artifacts

EXPOSE 8000

CMD ["uvicorn", "src.tumor_app.api:app", "--host", "0.0.0.0", "--port", "8000"]

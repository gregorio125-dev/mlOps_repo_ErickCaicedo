# Imagen de la API de riesgo de crédito (FastAPI + Uvicorn)
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencias: se descartan las de desarrollo (a partir del marcador en requirements.txt)
COPY requirements.txt .
RUN sed '/# --- Desarrollo/,$d' requirements.txt > requirements-docker.txt \
    && pip install -r requirements-docker.txt

# Código fuente y datos
COPY Base_de_datos.csv .
COPY mlops_pipeline ./mlops_pipeline

# Se entrena dentro de la imagen: el modelo queda compatible con las librerías instaladas
RUN python mlops_pipeline/model_training_evaluation.py

# Usuario sin privilegios
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "--app-dir", "mlops_pipeline", "model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]

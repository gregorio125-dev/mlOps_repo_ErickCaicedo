"""
model_deploy.py
===============
Núcleo del despliegue: expone el mejor modelo como servicio REST con FastAPI.

Responsabilidades
-----------------
* Cargar el modelo entrenado (``best_model.joblib``).
* Recibir datos (JSON o CSV), aplicar las transformaciones (van dentro del pipeline) y
  devolver las predicciones.
* Soportar predicción por lotes: varios registros en una sola solicitud.

Ejecución local (desde la raíz del proyecto):
    uvicorn --app-dir mlops_pipeline model_deploy:app --reload --port 8000
Documentación interactiva: http://localhost:8000/docs
"""
from __future__ import annotations

import io
import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Union

# Permite que joblib encuentre el módulo ``ft_engineering`` usado al entrenar.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402
from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field  # noqa: E402

import ft_engineering as fe  # noqa: E402

MAX_REGISTROS = 10_000
ETIQUETAS = {1: "Paga a tiempo", 0: "Riesgo de no pago"}


# ---------------------------------------------------------------------------
# Carga del modelo
# ---------------------------------------------------------------------------
def ruta_modelo() -> Path:
    """Ruta del modelo; se puede cambiar con la variable de entorno ``MODEL_PATH``."""
    return Path(os.getenv("MODEL_PATH", str(fe.RUTA_MODELO)))


@lru_cache(maxsize=1)
def cargar_modelo():
    """Carga (una sola vez) el pipeline entrenado desde disco."""
    ruta = ruta_modelo()
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {ruta}. "
            "Ejecuta primero: python mlops_pipeline/model_training_evaluation.py"
        )
    return joblib.load(ruta)


def cargar_info_modelo() -> dict[str, Any]:
    ruta = Path(os.getenv("MODEL_INFO_PATH", str(fe.RUTA_INFO_MODELO)))
    if ruta.exists():
        return json.loads(ruta.read_text(encoding="utf-8"))
    return {"umbral_decision": 0.5, "modelo": "desconocido"}


# ---------------------------------------------------------------------------
# Esquemas de entrada / salida
# ---------------------------------------------------------------------------
class Cliente(BaseModel):
    """Un registro de crédito. Todos los campos son opcionales: el pipeline imputa los faltantes."""

    model_config = ConfigDict(extra="ignore")

    tipo_credito: Optional[Union[int, str]] = Field(default=4, examples=[4])
    fecha_prestamo: Optional[str] = Field(default=None, examples=["2025-03-27 14:57:24"])
    capital_prestado: Optional[float] = Field(default=None, examples=[1852560.0])
    plazo_meses: Optional[float] = Field(default=None, examples=[12])
    edad_cliente: Optional[float] = Field(default=None, examples=[32])
    tipo_laboral: Optional[str] = Field(default=None, examples=["Empleado"])
    salario_cliente: Optional[float] = Field(default=None, examples=[3500000])
    total_otros_prestamos: Optional[float] = Field(default=None, examples=[1000000])
    cuota_pactada: Optional[float] = Field(default=None, examples=[128650])
    puntaje_datacredito: Optional[float] = Field(default=None, examples=[795])
    cant_creditosvigentes: Optional[float] = Field(default=None, examples=[0])
    huella_consulta: Optional[float] = Field(default=None, examples=[2])
    saldo_mora: Optional[float] = Field(default=None, examples=[0])
    saldo_total: Optional[float] = Field(default=None, examples=[0])
    saldo_principal: Optional[float] = Field(default=None, examples=[0])
    saldo_mora_codeudor: Optional[float] = Field(default=None, examples=[0])
    creditos_sectorFinanciero: Optional[float] = Field(default=None, examples=[0])
    creditos_sectorCooperativo: Optional[float] = Field(default=None, examples=[0])
    creditos_sectorReal: Optional[float] = Field(default=None, examples=[0])
    promedio_ingresos_datacredito: Optional[float] = Field(default=None, examples=[916148])
    tendencia_ingresos: Optional[str] = Field(default=None, examples=["Creciente"])


class SolicitudPrediccion(BaseModel):
    registros: list[Cliente] = Field(..., min_length=1, max_length=MAX_REGISTROS)


class Prediccion(BaseModel):
    indice: int
    prediccion: int
    etiqueta: str
    probabilidad_pago_atiempo: float
    probabilidad_incumplimiento: float


class RespuestaPrediccion(BaseModel):
    modelo: str
    total_registros: int
    predicciones: list[Prediccion]


# ---------------------------------------------------------------------------
# Lógica de predicción
# ---------------------------------------------------------------------------
def predecir_dataframe(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Aplica el pipeline (limpieza + features + modelo) y devuelve las predicciones."""
    modelo = cargar_modelo()
    umbral = float(cargar_info_modelo().get("umbral_decision", 0.5))
    probabilidad = modelo.predict_proba(df)[:, 1]
    resultados = []
    for i, p in enumerate(probabilidad):
        clase = int(p >= umbral)
        resultados.append(
            {
                "indice": i,
                "prediccion": clase,
                "etiqueta": ETIQUETAS[clase],
                "probabilidad_pago_atiempo": round(float(p), 6),
                "probabilidad_incumplimiento": round(float(1 - p), 6),
            }
        )
    return resultados


# ---------------------------------------------------------------------------
# Aplicación FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="API de riesgo de pago de créditos",
    description="Predice si un cliente pagará a tiempo su crédito (Pago_atiempo).",
    version="1.0.0",
)


@app.get("/")
def raiz() -> dict[str, str]:
    return {"mensaje": "API de predicción de pago a tiempo. Ver /docs para la documentación."}


@app.get("/health")
def salud() -> dict[str, Any]:
    """Verifica que el servicio y el modelo estén disponibles."""
    try:
        cargar_modelo()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    info = cargar_info_modelo()
    return {"status": "ok", "modelo": info.get("modelo"), "umbral_decision": info.get("umbral_decision")}


@app.post("/predict", response_model=RespuestaPrediccion)
def predict(solicitud: SolicitudPrediccion) -> RespuestaPrediccion:
    """Predicción por lotes a partir de JSON: ``{"registros": [{...}, {...}]}``."""
    df = pd.DataFrame([registro.model_dump() for registro in solicitud.registros])
    try:
        predicciones = predecir_dataframe(df)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return RespuestaPrediccion(
        modelo=cargar_info_modelo().get("modelo", "desconocido"),
        total_registros=len(predicciones),
        predicciones=predicciones,
    )


@app.post("/predict_csv", response_model=RespuestaPrediccion)
async def predict_csv(archivo: UploadFile = File(...)) -> RespuestaPrediccion:
    """Predicción por lotes a partir de un archivo CSV con las mismas columnas de la base."""
    contenido = await archivo.read()
    try:
        df = pd.read_csv(io.BytesIO(contenido))
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"CSV inválido: {error}") from error
    if df.empty:
        raise HTTPException(status_code=400, detail="El CSV no contiene registros.")
    if len(df) > MAX_REGISTROS:
        raise HTTPException(status_code=413, detail=f"Máximo {MAX_REGISTROS} registros por solicitud.")
    df = df.drop(columns=[fe.TARGET], errors="ignore")
    try:
        predicciones = predecir_dataframe(df)
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return RespuestaPrediccion(
        modelo=cargar_info_modelo().get("modelo", "desconocido"),
        total_registros=len(predicciones),
        predicciones=predicciones,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))

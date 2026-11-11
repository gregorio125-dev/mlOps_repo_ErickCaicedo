"""Configuración compartida de las pruebas.

Las pruebas no dependen del modelo guardado en ``mlops_pipeline/artifacts``: entrenan un modelo
rápido y lo guardan en una carpeta temporal.
"""
import json
import sys
from pathlib import Path

import joblib
import pytest

PIPELINE_DIR = Path(__file__).resolve().parent.parent / "mlops_pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

import ft_engineering as fe  # noqa: E402
import model_training_evaluation as mt  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402


@pytest.fixture(scope="session")
def datos_crudos():
    return fe.cargar_datos()


@pytest.fixture(scope="session")
def modelo_rapido(datos_crudos):
    """Pipeline completo (features + preprocesador + regresión logística) entrenado con una muestra."""
    muestra = datos_crudos.sample(3000, random_state=0)
    X = muestra.drop(columns=[fe.TARGET])
    y = muestra[fe.TARGET].astype(int)
    modelo = mt.build_model(LogisticRegression(max_iter=500, class_weight="balanced"))
    modelo.fit(X, y)
    return modelo


@pytest.fixture()
def modelo_en_disco(tmp_path, modelo_rapido, monkeypatch):
    """Guarda el modelo en una carpeta temporal y apunta la API a él."""
    import model_deploy as md

    ruta = tmp_path / "modelo.joblib"
    info = tmp_path / "info.json"
    joblib.dump(modelo_rapido, ruta)
    info.write_text(json.dumps({"modelo": "Prueba", "umbral_decision": 0.5}), encoding="utf-8")
    monkeypatch.setenv("MODEL_PATH", str(ruta))
    monkeypatch.setenv("MODEL_INFO_PATH", str(info))
    md.cargar_modelo.cache_clear()
    yield ruta
    md.cargar_modelo.cache_clear()

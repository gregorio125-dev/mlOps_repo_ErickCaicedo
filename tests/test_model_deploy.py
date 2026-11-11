import pandas as pd
import pytest
from fastapi.testclient import TestClient

import model_deploy as md

REGISTRO = {
    "tipo_credito": 4, "capital_prestado": 1852560, "plazo_meses": 12, "edad_cliente": 32,
    "tipo_laboral": "Empleado", "salario_cliente": 3500000, "cuota_pactada": 128650,
    "puntaje_datacredito": 795, "tendencia_ingresos": "Creciente",
}


@pytest.fixture()
def cliente(modelo_en_disco):
    return TestClient(md.app)


def test_raiz(cliente):
    assert cliente.get("/").status_code == 200


def test_health_ok(cliente):
    respuesta = cliente.get("/health")
    assert respuesta.status_code == 200
    assert respuesta.json()["status"] == "ok"


def test_predict_un_registro(cliente):
    respuesta = cliente.post("/predict", json={"registros": [REGISTRO]})
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["total_registros"] == 1
    prediccion = cuerpo["predicciones"][0]
    assert prediccion["prediccion"] in (0, 1)
    assert prediccion["probabilidad_pago_atiempo"] + prediccion["probabilidad_incumplimiento"] == pytest.approx(1)


def test_predict_por_lotes_y_registro_vacio(cliente):
    respuesta = cliente.post("/predict", json={"registros": [REGISTRO, REGISTRO, {}]})
    assert respuesta.status_code == 200
    assert respuesta.json()["total_registros"] == 3


def test_predict_rechaza_lista_vacia(cliente):
    assert cliente.post("/predict", json={"registros": []}).status_code == 422


def test_predict_csv(cliente, datos_crudos):
    contenido = datos_crudos.head(50).to_csv(index=False).encode()
    respuesta = cliente.post("/predict_csv", files={"archivo": ("datos.csv", contenido, "text/csv")})
    assert respuesta.status_code == 200
    assert respuesta.json()["total_registros"] == 50


def test_predict_csv_invalido(cliente):
    respuesta = cliente.post("/predict_csv", files={"archivo": ("v.csv", b"", "text/csv")})
    assert respuesta.status_code == 400


def test_predict_csv_sin_filas(cliente, datos_crudos):
    solo_encabezado = datos_crudos.head(0).to_csv(index=False).encode()
    respuesta = cliente.post("/predict_csv", files={"archivo": ("v.csv", solo_encabezado, "text/csv")})
    assert respuesta.status_code == 400


def test_predict_csv_demasiados_registros(cliente, monkeypatch, datos_crudos):
    monkeypatch.setattr(md, "MAX_REGISTROS", 10)
    contenido = datos_crudos.head(20).to_csv(index=False).encode()
    respuesta = cliente.post("/predict_csv", files={"archivo": ("d.csv", contenido, "text/csv")})
    assert respuesta.status_code == 413


def test_sin_modelo_devuelve_503(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "no_existe.joblib"))
    md.cargar_modelo.cache_clear()
    cliente = TestClient(md.app)
    assert cliente.get("/health").status_code == 503
    assert cliente.post("/predict", json={"registros": [REGISTRO]}).status_code == 503
    md.cargar_modelo.cache_clear()


def test_predecir_dataframe_respeta_umbral(modelo_en_disco):
    resultados = md.predecir_dataframe(pd.DataFrame([REGISTRO]))
    assert resultados[0]["etiqueta"] in md.ETIQUETAS.values()

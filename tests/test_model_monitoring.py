import numpy as np
import pandas as pd
import pytest

import ft_engineering as fe
import model_monitoring as mm


@pytest.fixture()
def serie_normal():
    return pd.Series(np.random.default_rng(0).normal(0, 1, 3000))


def test_psi_distribuciones_iguales_es_bajo(serie_normal):
    otra = pd.Series(np.random.default_rng(1).normal(0, 1, 3000))
    assert mm.calcular_psi(serie_normal, otra) < 0.1


def test_psi_con_desplazamiento_es_critico(serie_normal):
    desplazada = serie_normal + 1.5
    assert mm.calcular_psi(serie_normal, desplazada) > 0.25


def test_ks_detecta_cambio(serie_normal):
    estadistico, p_valor = mm.calcular_ks(serie_normal, serie_normal + 1)
    assert estadistico > 0.3
    assert p_valor < 0.01


def test_js_rango_valido(serie_normal):
    js = mm.calcular_js(serie_normal, serie_normal + 2)
    assert 0 < js <= 1
    assert mm.calcular_js(serie_normal, serie_normal) == pytest.approx(0, abs=1e-9)


def test_chi2_categoricas():
    ref = pd.Series(["a"] * 80 + ["b"] * 20)
    igual = pd.Series(["a"] * 80 + ["b"] * 20)
    distinta = pd.Series(["a"] * 20 + ["b"] * 80)
    assert mm.calcular_chi2(ref, igual)[2] == pytest.approx(0, abs=1e-9)
    chi2, p_valor, cramers_v = mm.calcular_chi2(ref, distinta)
    assert p_valor < 0.001
    assert cramers_v > 0.5


def test_chi2_con_una_sola_categoria():
    assert mm.calcular_chi2(pd.Series(["a", "a"]), pd.Series(["a"])) == (0.0, 1.0, 0.0)


def test_psi_variable_constante():
    assert mm.calcular_psi(pd.Series([1.0] * 50), pd.Series([1.0] * 50)) == pytest.approx(0, abs=1e-9)


def test_clasificar_nivel():
    assert mm.clasificar_nivel(0.05, mm.UMBRALES["psi"]) == "Verde"
    assert mm.clasificar_nivel(0.15, mm.UMBRALES["psi"]) == "Amarillo"
    assert mm.clasificar_nivel(0.40, mm.UMBRALES["psi"]) == "Rojo"


def test_asignar_periodos_agrupa_periodos_pequenos():
    fechas = pd.to_datetime(["2025-01-05"] * 120 + ["2025-02-05"] * 20 + ["2025-03-05"] * 10)
    periodos = mm.asignar_periodos(pd.DataFrame({fe.COLUMNA_FECHA: fechas}), "M", min_registros=100)
    assert periodos.nunique() == 1  # febrero y marzo se unen al grupo anterior
    periodos2 = mm.asignar_periodos(pd.DataFrame({fe.COLUMNA_FECHA: fechas[:140]}), "M", min_registros=100)
    assert periodos2.nunique() == 1


def _datos_sinteticos(desplazamiento_final=0.0):
    rng = np.random.default_rng(5)
    n = 2400
    fechas = pd.date_range("2025-01-01", periods=n, freq="6h")
    valor = rng.normal(100, 10, n)
    valor[n // 2:] += np.linspace(0, desplazamiento_final, n - n // 2)
    return pd.DataFrame(
        {
            fe.COLUMNA_FECHA: fechas,
            "capital_prestado": valor,
            "tipo_laboral": rng.choice(["Empleado", "Independiente"], n),
            mm.COLUMNA_PROBABILIDAD: rng.uniform(0.5, 1, n),
            mm.COLUMNA_PREDICCION: 1,
            fe.TARGET: rng.choice([0, 1], n, p=[0.05, 0.95]),
        }
    )


def test_calcular_drift_sin_cambio_es_verde():
    datos = _datos_sinteticos(0.0)
    ref, act = mm.dividir_referencia_actual(datos, 0.5)
    drift = mm.calcular_drift(ref, act, ["capital_prestado"], ["tipo_laboral"])
    assert set(drift["nivel"]) == {"Verde"}


def test_drift_temporal_detecta_cambio_y_genera_alertas():
    datos = _datos_sinteticos(desplazamiento_final=60)
    detalle, resumen = mm.drift_temporal(datos, "M", 0.5, 100)
    assert (detalle["nivel"] == "Rojo").any()
    alertas = mm.generar_alertas(detalle, resumen)
    assert len(alertas) > 0
    recomendaciones = mm.generar_recomendaciones(detalle, resumen)
    assert recomendaciones[0].startswith(("🔴", "🟠"))
    tendencias = mm.detectar_tendencias(detalle)
    fila = tendencias[tendencias["variable"] == "capital_prestado"].iloc[0]
    assert fila["tendencia_creciente"]


def test_recomendacion_sin_drift_es_verde():
    datos = _datos_sinteticos(0.0)
    detalle, resumen = mm.drift_temporal(datos, "M", 0.5, 600)
    assert mm.generar_recomendaciones(detalle, resumen)[0].startswith("🟢")


def test_recomendaciones_sin_periodos():
    assert mm.generar_recomendaciones(pd.DataFrame(), pd.DataFrame()) == ["No hay periodos nuevos para analizar."]


def test_cargar_datos_con_predicciones(modelo_rapido):
    datos = mm.cargar_datos_con_predicciones(modelo=modelo_rapido)
    assert {mm.COLUMNA_PROBABILIDAD, mm.COLUMNA_PREDICCION, fe.TARGET}.issubset(datos.columns)
    assert datos[fe.COLUMNA_FECHA].is_monotonic_increasing


def test_ejecutar_y_guardar_monitoreo(modelo_rapido, tmp_path, monkeypatch):
    original = mm.cargar_datos_con_predicciones
    monkeypatch.setattr(mm, "cargar_datos_con_predicciones", lambda: original(modelo=modelo_rapido))
    resultados = mm.ejecutar_monitoreo("M", 0.5, 300)
    assert not resultados["detalle"].empty
    mm.guardar_resultados(resultados, tmp_path)
    for archivo in ["drift_por_variable_y_periodo.csv", "resumen_por_periodo.csv", "tendencias_drift.csv", "alertas.json"]:
        assert (tmp_path / archivo).exists()


def test_estilo_nivel():
    assert "#ffc7ce" in mm._estilo_nivel("Rojo")

"""
model_monitoring.py
===================
Monitoreo del modelo y detección de *data drift*.

Trae en una tabla los datos junto con los pronósticos del modelo, los divide en un periodo de
referencia (histórico) y periodos nuevos (muestreo periódico) y calcula métricas que detectan
cambios en la población que pueden afectar el desempeño del modelo:

* KS (Kolmogorov-Smirnov)          -> variables numéricas
* PSI (Population Stability Index) -> numéricas y categóricas
* Jensen-Shannon                   -> numéricas y categóricas
* Chi-cuadrado (+ V de Cramér)     -> variables categóricas

Uso
---
Trabajo de monitoreo por consola (genera tablas y alertas en artifacts/monitoring):
    python mlops_pipeline/model_monitoring.py

Aplicación Streamlit (tablero visual):
    streamlit run mlops_pipeline/model_monitoring.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial.distance import jensenshannon  # noqa: E402
from scipy.stats import chi2_contingency, ks_2samp  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import ft_engineering as fe  # noqa: E402

DIRECTORIO_MONITOREO = fe.DIRECTORIO_ARTEFACTOS / "monitoring"
COLUMNA_PROBABILIDAD = "probabilidad_pago_atiempo"
COLUMNA_PREDICCION = "prediccion"

# Umbrales (amarillo, rojo) de cada métrica. Para PSI son los valores estándar de la industria.
UMBRALES = {
    "psi": (0.10, 0.25),
    "ks": (0.10, 0.20),
    "js": (0.10, 0.20),
    "cramers_v": (0.10, 0.30),
}
NIVELES = ["Verde", "Amarillo", "Rojo"]
SEMAFORO = {"Verde": "🟢", "Amarillo": "🟡", "Rojo": "🔴"}

VARIABLES_NUMERICAS_MONITOREO = [
    v for v in fe.VARIABLES_NUMERICAS if v != "tiene_mora"
] + [COLUMNA_PROBABILIDAD]
VARIABLES_CATEGORICAS_MONITOREO = fe.VARIABLES_CATEGORICAS + fe.VARIABLES_ORDINALES + ["tiene_mora"]


# ---------------------------------------------------------------------------
# 1. Tabla de datos + pronósticos
# ---------------------------------------------------------------------------
def cargar_datos_con_predicciones(ruta=None, modelo=None) -> pd.DataFrame:
    """Devuelve los datos limpios junto con la predicción y la probabilidad del modelo."""
    crudo = fe.cargar_datos(ruta)
    modelo = modelo if modelo is not None else joblib.load(fe.RUTA_MODELO)
    umbral = 0.5
    if fe.RUTA_INFO_MODELO.exists():
        umbral = float(json.loads(fe.RUTA_INFO_MODELO.read_text(encoding="utf-8")).get("umbral_decision", 0.5))

    probabilidad = modelo.predict_proba(crudo.drop(columns=[fe.TARGET], errors="ignore"))[:, 1]
    datos = fe.crear_variables_derivadas(fe.limpiar_datos(crudo))
    datos[COLUMNA_PROBABILIDAD] = probabilidad
    datos[COLUMNA_PREDICCION] = (probabilidad >= umbral).astype(int)
    datos = datos.dropna(subset=[fe.COLUMNA_FECHA]).sort_values(fe.COLUMNA_FECHA)
    return datos.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Muestreo periódico
# ---------------------------------------------------------------------------
def dividir_referencia_actual(datos: pd.DataFrame, fraccion_referencia: float = 0.5):
    """Separa el histórico (referencia) de los datos nuevos usando la fecha del préstamo."""
    fecha_corte = datos[fe.COLUMNA_FECHA].quantile(fraccion_referencia)
    referencia = datos[datos[fe.COLUMNA_FECHA] <= fecha_corte]
    actual = datos[datos[fe.COLUMNA_FECHA] > fecha_corte]
    return referencia, actual


def asignar_periodos(datos: pd.DataFrame, freq: str = "M", min_registros: int = 300) -> pd.Series:
    """Asigna cada registro a un periodo (M = mensual, W = semanal, Q = trimestral).

    Los periodos consecutivos con pocos registros se agrupan hasta alcanzar ``min_registros``
    para que las pruebas estadísticas sean confiables: con muestras pequeñas el PSI se infla por
    puro ruido (aprox. n_intervalos / n), lo que produciría falsas alertas.
    """
    periodo = datos[fe.COLUMNA_FECHA].dt.to_period(freq)
    conteo = periodo.value_counts().sort_index()

    grupos, actual, acumulado = [], [], 0
    for p, n in conteo.items():
        actual.append(p)
        acumulado += n
        if acumulado >= min_registros:
            grupos.append(actual)
            actual, acumulado = [], 0
    if actual:  # remanente pequeño: se une al último grupo
        if grupos:
            grupos[-1].extend(actual)
        else:
            grupos.append(actual)

    etiquetas = {}
    for grupo in grupos:
        nombre = str(grupo[0]) if len(grupo) == 1 else f"{grupo[0]} a {grupo[-1]}"
        for p in grupo:
            etiquetas[p] = nombre
    return periodo.map(etiquetas)


# ---------------------------------------------------------------------------
# 3. Métricas de data drift
# ---------------------------------------------------------------------------
def _proporciones_numericas(ref: pd.Series, act: pd.Series, bins: int = 10):
    """Proporciones por intervalo usando cuantiles de la referencia."""
    ref, act = ref.dropna().astype(float), act.dropna().astype(float)
    cortes = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(cortes) < 2:
        return np.array([1.0]), np.array([1.0])
    cortes[0], cortes[-1] = -np.inf, np.inf
    p = np.histogram(ref, cortes)[0] / len(ref)
    q = np.histogram(act, cortes)[0] / len(act)
    return p, q


def _proporciones_categoricas(ref: pd.Series, act: pd.Series):
    ref, act = ref.dropna().astype(str), act.dropna().astype(str)
    categorias = sorted(set(ref) | set(act))
    p = np.array([(ref == c).mean() for c in categorias])
    q = np.array([(act == c).mean() for c in categorias])
    return p, q


def psi_desde_proporciones(p: np.ndarray, q: np.ndarray, eps: float = 1e-4) -> float:
    p, q = np.clip(p, eps, None), np.clip(q, eps, None)
    return float(np.sum((q - p) * np.log(q / p)))


def js_desde_proporciones(p: np.ndarray, q: np.ndarray) -> float:
    if len(p) < 2:
        return 0.0
    return float(jensenshannon(p, q, base=2))


def calcular_psi(ref: pd.Series, act: pd.Series, bins: int = 10) -> float:
    """Population Stability Index de una variable numérica. <0.1 estable, 0.1-0.25 alerta, >0.25 crítico."""
    return psi_desde_proporciones(*_proporciones_numericas(ref, act, bins))


def calcular_js(ref: pd.Series, act: pd.Series, bins: int = 10) -> float:
    """Distancia de Jensen-Shannon (base 2, entre 0 y 1) de una variable numérica."""
    return js_desde_proporciones(*_proporciones_numericas(ref, act, bins))


def calcular_ks(ref: pd.Series, act: pd.Series) -> tuple[float, float]:
    """Estadístico y p-valor de Kolmogorov-Smirnov de dos muestras."""
    resultado = ks_2samp(ref.dropna().astype(float), act.dropna().astype(float))
    return float(resultado.statistic), float(resultado.pvalue)


def calcular_chi2(ref: pd.Series, act: pd.Series) -> tuple[float, float, float]:
    """Chi-cuadrado, p-valor y V de Cramér para variables categóricas."""
    ref, act = ref.dropna().astype(str), act.dropna().astype(str)
    categorias = sorted(set(ref) | set(act))
    tabla = np.array(
        [[(ref == c).sum() for c in categorias], [(act == c).sum() for c in categorias]]
    )
    tabla = tabla[:, tabla.sum(axis=0) > 0]
    if tabla.shape[1] < 2:
        return 0.0, 1.0, 0.0
    chi2, p_valor, _, _ = chi2_contingency(tabla)
    cramers_v = float(np.sqrt(chi2 / (tabla.sum() * (min(tabla.shape) - 1))))
    return float(chi2), float(p_valor), cramers_v


def clasificar_nivel(valor: float, umbrales: tuple[float, float]) -> str:
    amarillo, rojo = umbrales
    if valor >= rojo:
        return "Rojo"
    if valor >= amarillo:
        return "Amarillo"
    return "Verde"


def _peor_nivel(niveles: list[str]) -> str:
    return max(niveles, key=NIVELES.index)


def calcular_drift(
    referencia: pd.DataFrame,
    actual: pd.DataFrame,
    numericas: list[str] | None = None,
    categoricas: list[str] | None = None,
) -> pd.DataFrame:
    """Calcula las métricas de drift por variable entre la referencia y el periodo actual."""
    numericas = numericas if numericas is not None else VARIABLES_NUMERICAS_MONITOREO
    categoricas = categoricas if categoricas is not None else VARIABLES_CATEGORICAS_MONITOREO
    filas = []

    for variable in numericas:
        if variable not in referencia or variable not in actual:
            continue
        ref, act = referencia[variable], actual[variable]
        if ref.dropna().empty or act.dropna().empty:
            continue
        p, q = _proporciones_numericas(ref, act)
        psi, js = psi_desde_proporciones(p, q), js_desde_proporciones(p, q)
        ks, ks_p = calcular_ks(ref, act)
        nivel = _peor_nivel(
            [
                clasificar_nivel(psi, UMBRALES["psi"]),
                clasificar_nivel(ks, UMBRALES["ks"]),
                clasificar_nivel(js, UMBRALES["js"]),
            ]
        )
        filas.append(
            {"variable": variable, "tipo": "numerica", "psi": psi, "ks": ks, "ks_pvalue": ks_p,
             "js": js, "chi2": np.nan, "chi2_pvalue": np.nan, "cramers_v": np.nan, "nivel": nivel}
        )

    for variable in categoricas:
        if variable not in referencia or variable not in actual:
            continue
        ref, act = referencia[variable], actual[variable]
        if ref.dropna().empty or act.dropna().empty:
            continue
        p, q = _proporciones_categoricas(ref, act)
        psi, js = psi_desde_proporciones(p, q), js_desde_proporciones(p, q)
        chi2, chi2_p, cramers_v = calcular_chi2(ref, act)
        nivel = _peor_nivel(
            [
                clasificar_nivel(psi, UMBRALES["psi"]),
                clasificar_nivel(js, UMBRALES["js"]),
                clasificar_nivel(cramers_v, UMBRALES["cramers_v"]),
            ]
        )
        filas.append(
            {"variable": variable, "tipo": "categorica", "psi": psi, "ks": np.nan, "ks_pvalue": np.nan,
             "js": js, "chi2": chi2, "chi2_pvalue": chi2_p, "cramers_v": cramers_v, "nivel": nivel}
        )
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# 4. Análisis temporal, alertas y recomendaciones
# ---------------------------------------------------------------------------
def drift_temporal(
    datos: pd.DataFrame, freq: str = "M", fraccion_referencia: float = 0.5, min_registros: int = 300
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula el drift de cada periodo nuevo contra la referencia.

    Devuelve (drift_por_periodo_y_variable, resumen_por_periodo).
    """
    referencia, actual = dividir_referencia_actual(datos, fraccion_referencia)
    actual = actual.copy()
    actual["periodo"] = asignar_periodos(actual, freq, min_registros)

    detalle, resumen = [], []
    for periodo, grupo in actual.groupby("periodo", sort=False):
        drift = calcular_drift(referencia, grupo)
        drift.insert(0, "periodo", periodo)
        detalle.append(drift)

        fila = {
            "periodo": periodo,
            "fecha_inicio": grupo[fe.COLUMNA_FECHA].min(),
            "registros": len(grupo),
            "variables_rojo": int((drift["nivel"] == "Rojo").sum()),
            "variables_amarillo": int((drift["nivel"] == "Amarillo").sum()),
            "tasa_riesgo_predicha": float((grupo[COLUMNA_PREDICCION] == 0).mean()),
            "tasa_incumplimiento_real": (
                float((grupo[fe.TARGET] == 0).mean()) if fe.TARGET in grupo else np.nan
            ),
            "auc": np.nan,
        }
        if fe.TARGET in grupo and (grupo[fe.TARGET] == 0).sum() >= 5 and grupo[fe.TARGET].nunique() == 2:
            fila["auc"] = float(roc_auc_score(grupo[fe.TARGET], grupo[COLUMNA_PROBABILIDAD]))
        psi_pred = drift.loc[drift["variable"] == COLUMNA_PROBABILIDAD, "psi"]
        fila["psi_prediccion"] = float(psi_pred.iloc[0]) if len(psi_pred) else np.nan
        resumen.append(fila)

    detalle_df = pd.concat(detalle, ignore_index=True) if detalle else pd.DataFrame()
    return detalle_df, pd.DataFrame(resumen)


def detectar_tendencias(detalle: pd.DataFrame, salto_abrupto: float = 0.10) -> pd.DataFrame:
    """Detecta tendencia creciente del PSI y cambios abruptos entre periodos consecutivos."""
    filas = []
    for variable, grupo in detalle.groupby("variable"):
        psi = grupo["psi"].to_numpy()
        pendiente = float(np.polyfit(np.arange(len(psi)), psi, 1)[0]) if len(psi) >= 3 else 0.0
        salto = float(np.max(np.abs(np.diff(psi)))) if len(psi) >= 2 else 0.0
        filas.append(
            {
                "variable": variable,
                "psi_ultimo": float(psi[-1]),
                "pendiente_psi": pendiente,
                "salto_maximo": salto,
                "tendencia_creciente": bool(pendiente > 0.01),
                "cambio_abrupto": bool(salto >= salto_abrupto),
            }
        )
    return pd.DataFrame(filas).sort_values("psi_ultimo", ascending=False).reset_index(drop=True)


def generar_alertas(detalle: pd.DataFrame, resumen: pd.DataFrame) -> list[dict]:
    """Genera alertas para cada variable en nivel Rojo y para el drift de las predicciones."""
    alertas = []
    rojas = detalle[detalle["nivel"] == "Rojo"]
    for _, fila in rojas.iterrows():
        alertas.append(
            {
                "severidad": "critica",
                "periodo": fila["periodo"],
                "variable": fila["variable"],
                "mensaje": f"Drift crítico en '{fila['variable']}' ({fila['periodo']}): PSI={fila['psi']:.3f}",
            }
        )
    if len(resumen):
        ultimo = resumen.iloc[-1]
        if pd.notna(ultimo["psi_prediccion"]) and ultimo["psi_prediccion"] >= UMBRALES["psi"][1]:
            alertas.append(
                {
                    "severidad": "critica",
                    "periodo": ultimo["periodo"],
                    "variable": COLUMNA_PROBABILIDAD,
                    "mensaje": "Las predicciones del modelo cambiaron de forma crítica respecto al histórico.",
                }
            )
    return alertas


def generar_recomendaciones(detalle: pd.DataFrame, resumen: pd.DataFrame) -> list[str]:
    """Mensajes automáticos de acción según el estado del drift en el último periodo."""
    if detalle.empty:
        return ["No hay periodos nuevos para analizar."]
    ultimo = detalle[detalle["periodo"] == detalle["periodo"].iloc[-1]]
    rojas = ultimo.loc[ultimo["nivel"] == "Rojo", "variable"].tolist()
    amarillas = ultimo.loc[ultimo["nivel"] == "Amarillo", "variable"].tolist()
    mensajes = []
    proporcion_roja = len(rojas) / max(len(ultimo), 1)
    psi_pred = resumen.iloc[-1]["psi_prediccion"] if len(resumen) else np.nan

    if proporcion_roja >= 0.25 or (pd.notna(psi_pred) and psi_pred >= UMBRALES["psi"][1]):
        mensajes.append("🔴 Se superó el umbral crítico: se recomienda REENTRENAR el modelo con datos recientes.")
    elif rojas:
        mensajes.append("🟠 Hay variables con drift crítico: revisar su calidad y su fuente antes de reentrenar.")
    elif amarillas:
        mensajes.append("🟡 Hay variables en alerta: mantener monitoreo cercano el próximo periodo.")
    else:
        mensajes.append("🟢 No se detecta drift relevante: el modelo puede seguir en producción.")
    if rojas:
        mensajes.append("Variables a revisar: " + ", ".join(rojas))
    if amarillas:
        mensajes.append("Variables en observación: " + ", ".join(amarillas))
    return mensajes


# ---------------------------------------------------------------------------
# 5. Trabajo de monitoreo (consola)
# ---------------------------------------------------------------------------
def ejecutar_monitoreo(freq: str = "M", fraccion_referencia: float = 0.5, min_registros: int = 300) -> dict:
    """Corre todo el monitoreo y devuelve un diccionario con los resultados."""
    datos = cargar_datos_con_predicciones()
    detalle, resumen = drift_temporal(datos, freq, fraccion_referencia, min_registros)
    referencia, actual = dividir_referencia_actual(datos, fraccion_referencia)
    actual = actual.assign(periodo=asignar_periodos(actual, freq, min_registros))
    return {
        "datos": datos,
        "referencia": referencia,
        "actual": actual,
        "detalle": detalle,
        "resumen": resumen,
        "tendencias": detectar_tendencias(detalle) if len(detalle) else pd.DataFrame(),
        "alertas": generar_alertas(detalle, resumen),
        "recomendaciones": generar_recomendaciones(detalle, resumen),
    }


def _redondear(df: pd.DataFrame, decimales: int = 5) -> pd.DataFrame:
    """Redondea solo las columnas decimales (evita avisos con fechas)."""
    return df.apply(lambda c: c.round(decimales) if pd.api.types.is_float_dtype(c) else c)


def guardar_resultados(resultados: dict, directorio: Path = DIRECTORIO_MONITOREO) -> None:
    directorio.mkdir(parents=True, exist_ok=True)
    _redondear(resultados["detalle"]).to_csv(directorio / "drift_por_variable_y_periodo.csv", index=False)
    _redondear(resultados["resumen"]).to_csv(directorio / "resumen_por_periodo.csv", index=False)
    _redondear(resultados["tendencias"]).to_csv(directorio / "tendencias_drift.csv", index=False)
    contenido = {"alertas": resultados["alertas"], "recomendaciones": resultados["recomendaciones"]}
    (directorio / "alertas.json").write_text(
        json.dumps(contenido, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


def main() -> None:
    resultados = ejecutar_monitoreo()
    guardar_resultados(resultados)
    print("=== Resumen por periodo ===")
    print(_redondear(resultados["resumen"], 4).to_string(index=False))
    print("\n=== Recomendaciones ===")
    for mensaje in resultados["recomendaciones"]:
        print(" -", mensaje)
    print(f"\nAlertas críticas generadas: {len(resultados['alertas'])}")
    print(f"Resultados guardados en: {DIRECTORIO_MONITOREO}")


# ---------------------------------------------------------------------------
# 6. Aplicación Streamlit
# ---------------------------------------------------------------------------
def _estilo_nivel(valor: str) -> str:
    colores = {"Verde": "#c6efce", "Amarillo": "#ffeb9c", "Rojo": "#ffc7ce"}
    return f"background-color: {colores.get(valor, 'white')}"


def app() -> None:  # pragma: no cover - interfaz visual
    import matplotlib.pyplot as plt
    import streamlit as st

    st.set_page_config(page_title="Monitoreo de data drift", page_icon="📊", layout="wide")
    st.title("📊 Monitoreo del modelo de riesgo de crédito")
    st.caption("Detección de data drift: histórico (referencia) vs. periodos recientes.")

    with st.sidebar:
        st.header("Parámetros")
        opciones = {"Mensual": "M", "Semanal": "W", "Trimestral": "Q"}
        freq = opciones[st.selectbox("Periodicidad del muestreo", list(opciones))]
        fraccion = st.slider("Proporción de datos usada como histórico", 0.3, 0.8, 0.5, 0.05)
        minimo = st.number_input("Mínimo de registros por periodo", 50, 1000, 300, 50)
        st.markdown("**Umbrales PSI:** <0.10 🟢 · 0.10-0.25 🟡 · >0.25 🔴")

    calcular = st.cache_data(show_spinner="Calculando drift...")(ejecutar_monitoreo)
    try:
        res = calcular(freq, fraccion, int(minimo))
    except FileNotFoundError:
        st.error("No se encontró el modelo. Ejecuta primero model_training_evaluation.py")
        return
    detalle, resumen = res["detalle"], res["resumen"]
    if detalle.empty:
        st.warning("No hay periodos nuevos con los parámetros elegidos.")
        return

    periodos = list(resumen["periodo"])
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["🚦 Resumen y alertas", "📈 Histórico vs actual", "⏱️ Análisis temporal", "🎯 Desempeño", "🗂️ Datos + pronósticos"]
    )

    with tab1:
        periodo = st.selectbox("Periodo a evaluar", periodos, index=len(periodos) - 1)
        actual = detalle[detalle["periodo"] == periodo]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Variables monitoreadas", len(actual))
        c2.metric("🔴 Críticas", int((actual["nivel"] == "Rojo").sum()))
        c3.metric("🟡 En alerta", int((actual["nivel"] == "Amarillo").sum()))
        c4.metric("🟢 Estables", int((actual["nivel"] == "Verde").sum()))

        for mensaje in res["recomendaciones"]:
            (st.error if mensaje.startswith("🔴") else st.warning if mensaje[0] in "🟠🟡" else st.success)(mensaje)

        col_a, col_b = st.columns([3, 2])
        with col_a:
            tabla = actual.drop(columns=["periodo"]).copy()
            tabla.insert(0, "semáforo", tabla["nivel"].map(SEMAFORO))
            st.subheader("Métricas de drift por variable")
            st.dataframe(
                tabla.style.map(_estilo_nivel, subset=["nivel"]).format(
                    {c: "{:.4f}" for c in ["psi", "ks", "ks_pvalue", "js", "chi2", "chi2_pvalue", "cramers_v"]}
                ),
                width="stretch", height=420,
            )
        with col_b:
            st.subheader("Barras de riesgo (PSI)")
            orden = actual.sort_values("psi")
            fig, ax = plt.subplots(figsize=(5, 6))
            colores = orden["nivel"].map({"Verde": "#2ca02c", "Amarillo": "#ffbf00", "Rojo": "#d62728"})
            ax.barh(orden["variable"], orden["psi"], color=colores)
            ax.axvline(UMBRALES["psi"][0], color="orange", ls="--")
            ax.axvline(UMBRALES["psi"][1], color="red", ls="--")
            ax.tick_params(axis="y", labelsize=7)
            ax.set_xlabel("PSI")
            st.pyplot(fig)
            plt.close(fig)

        if res["alertas"]:
            with st.expander(f"Alertas críticas ({len(res['alertas'])})"):
                for alerta in res["alertas"]:
                    st.write("🔴", alerta["mensaje"])

    with tab2:
        variable = st.selectbox("Variable", list(detalle["variable"].unique()))
        periodo2 = st.selectbox("Periodo", periodos, index=len(periodos) - 1, key="p2")
        ref, actual_df = res["referencia"], res["actual"]
        grupo = actual_df[actual_df["periodo"] == periodo2]
        fig, ax = plt.subplots(figsize=(9, 4))
        if variable in VARIABLES_NUMERICAS_MONITOREO:
            lo, hi = ref[variable].quantile([0.01, 0.99])
            rango = (lo, hi) if hi > lo else None
            ax.hist(ref[variable].dropna(), bins=30, range=rango, alpha=0.55, density=True, label="Histórico")
            ax.hist(grupo[variable].dropna(), bins=30, range=rango, alpha=0.55, density=True, label=f"Actual ({periodo2})")
        else:
            p, q = _proporciones_categoricas(ref[variable], grupo[variable])
            categorias = sorted(set(ref[variable].dropna().astype(str)) | set(grupo[variable].dropna().astype(str)))
            x = np.arange(len(categorias))
            ax.bar(x - 0.2, p, 0.4, label="Histórico")
            ax.bar(x + 0.2, q, 0.4, label=f"Actual ({periodo2})")
            ax.set_xticks(x, categorias)
        ax.set_title(variable)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    with tab3:
        st.subheader("Evolución del drift (PSI) en el tiempo")
        seleccion = st.multiselect(
            "Variables", list(detalle["variable"].unique()),
            default=list(res["tendencias"]["variable"].head(4)),
        )
        if seleccion:
            pivote = detalle[detalle["variable"].isin(seleccion)].pivot(index="periodo", columns="variable", values="psi")
            st.line_chart(pivote.reindex(periodos))
        st.subheader("Tendencias y cambios abruptos")
        st.dataframe(res["tendencias"].round(4), width="stretch")
        st.subheader("Resumen del drift por periodo")
        st.bar_chart(resumen.set_index("periodo")[["variables_rojo", "variables_amarillo"]])

    with tab4:
        st.subheader("Predicciones vs. realidad por periodo")
        st.dataframe(_redondear(resumen, 4), width="stretch")
        st.line_chart(resumen.set_index("periodo")[["tasa_riesgo_predicha", "tasa_incumplimiento_real"]])

    with tab5:
        columnas = [fe.COLUMNA_FECHA, fe.TARGET, COLUMNA_PREDICCION, COLUMNA_PROBABILIDAD]
        st.dataframe(res["datos"][columnas + [c for c in res["datos"].columns if c not in columnas]].head(500))
        st.download_button("Descargar tabla completa (CSV)", res["datos"].to_csv(index=False), "datos_con_pronosticos.csv")


def _ejecutando_en_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx(suppress_warning=True) is not None
    except Exception:  # streamlit no instalado o fuera de su runtime
        return False


if _ejecutando_en_streamlit():  # pragma: no cover
    app()
elif __name__ == "__main__":
    main()

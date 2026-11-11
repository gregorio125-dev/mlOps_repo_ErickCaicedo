"""
ft_engineering.py
=================
Primer componente del flujo de creación del modelo (ingeniería de características).

Responsabilidades
-----------------
1. Cargar la base de datos (``Base_de_datos.csv``).
2. Limpiar los datos aplicando las reglas descubiertas en ``comprension_eda.ipynb``.
3. Crear variables derivadas (ratios financieros, flags de mora, etc.).
4. Construir el ``ColumnTransformer`` con tres ramas:
   numeric / categoric / categoric ordinales.
5. Generar los conjuntos de entrenamiento y evaluación (train / test).

Variable objetivo: ``Pago_atiempo`` (1 = pagó a tiempo, 0 = no pagó a tiempo).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

# ---------------------------------------------------------------------------
# Rutas y constantes generales
# ---------------------------------------------------------------------------
DIRECTORIO_PIPELINE = Path(__file__).resolve().parent
RAIZ_PROYECTO = DIRECTORIO_PIPELINE.parent
RUTA_DATOS = RAIZ_PROYECTO / "Base_de_datos.csv"
DIRECTORIO_ARTEFACTOS = DIRECTORIO_PIPELINE / "artifacts"
RUTA_MODELO = DIRECTORIO_ARTEFACTOS / "best_model.joblib"
RUTA_INFO_MODELO = DIRECTORIO_ARTEFACTOS / "best_model_info.json"

TARGET = "Pago_atiempo"
COLUMNA_FECHA = "fecha_prestamo"
RANDOM_STATE = 42

# 'puntaje' se excluye: tiene fuga de información (data leakage).
# En el EDA se demuestra que por sí sola separa perfectamente el objetivo (AUC = 1.0).
COLUMNAS_EXCLUIDAS = ["puntaje"]

# ---------------------------------------------------------------------------
# Definición de variables por tipo
# ---------------------------------------------------------------------------
VARIABLES_NUMERICAS_BASE = [
    "capital_prestado",
    "plazo_meses",
    "edad_cliente",
    "salario_cliente",
    "total_otros_prestamos",
    "cuota_pactada",
    "puntaje_datacredito",
    "cant_creditosvigentes",
    "huella_consulta",
    "saldo_mora",
    "saldo_total",
    "saldo_principal",
    "saldo_mora_codeudor",
    "creditos_sectorFinanciero",
    "creditos_sectorCooperativo",
    "creditos_sectorReal",
    "promedio_ingresos_datacredito",
]
VARIABLES_DERIVADAS = [
    "cuota_sobre_salario",
    "capital_sobre_salario",
    "deuda_sobre_salario",
    "salario_sobre_ingreso_datacredito",
    "total_creditos_sectores",
    "mora_total",
    "tiene_mora",
]
VARIABLES_NUMERICAS = VARIABLES_NUMERICAS_BASE + VARIABLES_DERIVADAS
VARIABLES_CATEGORICAS = ["tipo_credito", "tipo_laboral"]  # nominales
VARIABLES_ORDINALES = ["tendencia_ingresos"]  # ordinal: Decreciente < Estable < Creciente
ORDEN_TENDENCIA = ["Decreciente", "Estable", "Creciente"]
FEATURES = VARIABLES_NUMERICAS + VARIABLES_CATEGORICAS + VARIABLES_ORDINALES

# Valores que representan "nulo" en la fuente. Se unifican todos a NaN.
VALORES_NULOS = ["", " ", "NA", "N/A", "NaN", "nan", "NULL", "null", "None", "none", "?", "-"]

# Reglas de validación (rango permitido). Valores fuera del rango se consideran inválidos -> NaN.
# Se definieron en el EDA al revisar outliers y valores centinela (p. ej. edad = 122).
LIMITES_VALIDOS: dict[str, tuple[float | None, float | None]] = {
    "edad_cliente": (18, 100),
    "salario_cliente": (1, 1e8),
    "total_otros_prestamos": (0, 1e9),
    "puntaje_datacredito": (150, 950),
    "capital_prestado": (1, None),
    "plazo_meses": (1, 120),
    "cuota_pactada": (1, None),
}


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------
def cargar_datos(ruta: str | Path | None = None) -> pd.DataFrame:
    """Carga la base de datos desde el CSV (por defecto ``Base_de_datos.csv``)."""
    ruta = Path(ruta) if ruta else RUTA_DATOS
    return pd.read_csv(ruta)


# ---------------------------------------------------------------------------
# Limpieza
# ---------------------------------------------------------------------------
def aplicar_limites(df: pd.DataFrame, limites: dict | None = None) -> pd.DataFrame:
    """Convierte a NaN los valores numéricos fuera de los rangos válidos."""
    df = df.copy()
    for columna, (minimo, maximo) in (limites or LIMITES_VALIDOS).items():
        if columna not in df.columns:
            continue
        invalido = pd.Series(False, index=df.index)
        if minimo is not None:
            invalido |= df[columna] < minimo
        if maximo is not None:
            invalido |= df[columna] > maximo
        df.loc[invalido, columna] = np.nan
    return df


def validar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Reporta cuántos registros violan cada regla de validación (para el EDA y el monitoreo)."""
    filas = []
    for columna, (minimo, maximo) in LIMITES_VALIDOS.items():
        if columna not in df.columns:
            continue
        serie = pd.to_numeric(df[columna], errors="coerce")
        invalido = pd.Series(False, index=df.index)
        if minimo is not None:
            invalido |= serie < minimo
        if maximo is not None:
            invalido |= serie > maximo
        filas.append(
            {
                "variable": columna,
                "minimo_valido": minimo,
                "maximo_valido": maximo,
                "registros_invalidos": int(invalido.sum()),
                "porcentaje": round(100 * invalido.mean(), 3),
            }
        )
    if "tendencia_ingresos" in df.columns:
        validos = df["tendencia_ingresos"].astype("string").str.strip().str.capitalize()
        invalido = validos.notna() & ~validos.isin(ORDEN_TENDENCIA)
        filas.append(
            {
                "variable": "tendencia_ingresos",
                "minimo_valido": None,
                "maximo_valido": None,
                "registros_invalidos": int(invalido.sum()),
                "porcentaje": round(100 * invalido.mean(), 3),
            }
        )
    return pd.DataFrame(filas)


def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica las reglas de limpieza definidas en el EDA.

    - Unifica los valores nulos a ``np.nan``.
    - Convierte cada columna a su tipo correcto.
    - Marca como nulos los valores fuera de rango.
    - Corrige ``tendencia_ingresos`` (contiene números que no son categorías).
    - ``saldo_principal`` vacío con ``saldo_total`` = 0 significa "sin deuda" -> 0.
    """
    df = df.copy()

    # 1) Unificar nulos en columnas de texto
    for columna in df.select_dtypes(include=["object", "string"]).columns:
        df[columna] = df[columna].replace(VALORES_NULOS, np.nan)

    # 2) Tipos correctos
    for columna in VARIABLES_NUMERICAS_BASE:
        if columna in df.columns:
            df[columna] = pd.to_numeric(df[columna], errors="coerce")
    if COLUMNA_FECHA in df.columns:
        df[COLUMNA_FECHA] = pd.to_datetime(df[COLUMNA_FECHA], errors="coerce")
    if "tipo_credito" in df.columns:
        codigo = pd.to_numeric(df["tipo_credito"], errors="coerce")
        df["tipo_credito"] = pd.Series(
            [str(int(v)) if pd.notna(v) else np.nan for v in codigo], index=df.index, dtype=object
        )
    if "tipo_laboral" in df.columns:
        df["tipo_laboral"] = df["tipo_laboral"].astype("string").str.strip().str.capitalize()
        df["tipo_laboral"] = df["tipo_laboral"].astype(object).where(df["tipo_laboral"].notna(), np.nan)
    if "tendencia_ingresos" in df.columns:
        texto = df["tendencia_ingresos"].astype("string").str.strip().str.capitalize()
        texto = texto.where(texto.isin(ORDEN_TENDENCIA))  # números u otros textos -> nulo
        df["tendencia_ingresos"] = texto.astype(object).where(texto.notna(), np.nan)

    # 3) Valores fuera de rango -> nulo
    df = aplicar_limites(df)

    # 4) Nulos estructurales: sin saldo total => sin saldo principal
    if {"saldo_total", "saldo_principal"}.issubset(df.columns):
        sin_deuda = (df["saldo_total"] == 0) & df["saldo_principal"].isna()
        df.loc[sin_deuda, "saldo_principal"] = 0.0
    return df


# ---------------------------------------------------------------------------
# Variables derivadas
# ---------------------------------------------------------------------------
def _dividir(numerador: pd.Series, denominador: pd.Series) -> pd.Series:
    """División segura: evita infinitos cuando el denominador es 0."""
    return (numerador / denominador.where(denominador != 0)).replace([np.inf, -np.inf], np.nan)


def crear_variables_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    """Crea ratios financieros y banderas a partir de las variables limpias."""
    df = df.copy()
    nan = pd.Series(np.nan, index=df.index)

    def col(nombre: str) -> pd.Series:
        return df[nombre] if nombre in df.columns else nan

    df["cuota_sobre_salario"] = _dividir(col("cuota_pactada"), col("salario_cliente"))
    df["capital_sobre_salario"] = _dividir(col("capital_prestado"), col("salario_cliente"))
    df["deuda_sobre_salario"] = _dividir(col("total_otros_prestamos"), col("salario_cliente"))
    df["salario_sobre_ingreso_datacredito"] = _dividir(
        col("salario_cliente"), col("promedio_ingresos_datacredito")
    )
    df["total_creditos_sectores"] = (
        col("creditos_sectorFinanciero").fillna(0)
        + col("creditos_sectorCooperativo").fillna(0)
        + col("creditos_sectorReal").fillna(0)
    )
    df["mora_total"] = col("saldo_mora").fillna(0) + col("saldo_mora_codeudor").fillna(0)
    df["tiene_mora"] = (df["mora_total"] > 0).astype(int)
    return df


def preparar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza + variables derivadas. Devuelve solo las columnas predictoras (``FEATURES``).

    Es el primer paso del pipeline del modelo: así el modelo guardado recibe datos
    "crudos" (como llegan de la base o de la API) y aplica todo el tratamiento.
    """
    df = crear_variables_derivadas(limpiar_datos(df))
    for columna in FEATURES:
        if columna not in df.columns:
            df[columna] = np.nan
    return df[FEATURES]


# ---------------------------------------------------------------------------
# Preprocesamiento (ColumnTransformer)
# ---------------------------------------------------------------------------
def construir_preprocesador() -> ColumnTransformer:
    """ColumnTransformer con las ramas numeric, categoric y categoric ordinales."""
    numerico = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorico = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist", min_frequency=30, sparse_output=False
                ),
            ),
        ]
    )
    ordinal = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(
                    categories=[ORDEN_TENDENCIA],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numerico, VARIABLES_NUMERICAS),
            ("categoric", categorico, VARIABLES_CATEGORICAS),
            ("categoric ordinales", ordinal, VARIABLES_ORDINALES),
        ],
        verbose_feature_names_out=False,
    )


def construir_pipeline_features() -> Pipeline:
    """Pipeline completo de features: datos crudos -> matriz numérica lista para el modelo."""
    return Pipeline(
        [
            ("features", FunctionTransformer(preparar_features)),
            ("preprocesador", construir_preprocesador()),
        ]
    )


# ---------------------------------------------------------------------------
# Datasets de entrenamiento y evaluación
# ---------------------------------------------------------------------------
def generar_datasets(
    test_size: float = 0.2, random_state: int = RANDOM_STATE, ruta: str | Path | None = None
):
    """Divide la base en entrenamiento y evaluación (estratificado por el objetivo).

    Devuelve ``X_train, X_test, y_train, y_test`` con las columnas *crudas*;
    el pipeline del modelo se encarga de transformarlas.
    """
    df = cargar_datos(ruta)
    df = df.dropna(subset=[TARGET])
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def generar_datasets_transformados(test_size: float = 0.2, random_state: int = RANDOM_STATE):
    """Devuelve los conjuntos ya transformados (DataFrames) y el pipeline de features ajustado."""
    X_train, X_test, y_train, y_test = generar_datasets(test_size, random_state)
    pipeline = construir_pipeline_features()
    nombres = None
    Xt_train = pipeline.fit_transform(X_train)
    Xt_test = pipeline.transform(X_test)
    nombres = pipeline.named_steps["preprocesador"].get_feature_names_out()
    Xt_train = pd.DataFrame(Xt_train, columns=nombres, index=X_train.index)
    Xt_test = pd.DataFrame(Xt_test, columns=nombres, index=X_test.index)
    return Xt_train, Xt_test, y_train, y_test, pipeline


def main() -> None:
    """Ejecuta el componente y muestra un resumen de los datasets resultantes."""
    Xt_train, Xt_test, y_train, y_test, _ = generar_datasets_transformados()
    print(f"Train: {Xt_train.shape} | Test: {Xt_test.shape}")
    print(f"Variables resultantes: {Xt_train.shape[1]}")
    print("Distribución del objetivo (train):")
    print(y_train.value_counts(normalize=True).round(4).to_string())
    print("Distribución del objetivo (test):")
    print(y_test.value_counts(normalize=True).round(4).to_string())


if __name__ == "__main__":
    main()

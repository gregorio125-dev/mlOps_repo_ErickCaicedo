"""
model_training_evaluation.py
============================
Entrena y evalúa varios modelos supervisados y selecciona el mejor según tres criterios:

* Performance  : capacidad predictiva en el conjunto de prueba (ROC-AUC).
* Consistency  : estabilidad entre folds de validación cruzada (desviación estándar baja).
* Scalability  : tiempo de entrenamiento (más rápido = mejor).

Salidas (carpeta ``mlops_pipeline/artifacts``):
    best_model.joblib        -> Pipeline completo (features + preprocesador + modelo)
    best_model_info.json     -> metadatos del modelo seleccionado
    model_comparison.csv     -> tabla resumen de evaluación
    figures/*.png            -> gráficos comparativos
"""
from __future__ import annotations

import json
import time
import warnings

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

import ft_engineering as fe

warnings.filterwarnings("ignore")

DIRECTORIO_FIGURAS = fe.DIRECTORIO_ARTEFACTOS / "figures"
UMBRAL_DECISION = 0.5
N_FOLDS = 5
# Pesos para elegir el mejor modelo (suman 1)
PESOS_SELECCION = {"performance": 0.60, "consistency": 0.25, "scalability": 0.15}


# ---------------------------------------------------------------------------
# Funciones reutilizables
# ---------------------------------------------------------------------------
def obtener_modelos() -> dict:
    """Catálogo de modelos candidatos. ``class_weight`` compensa el desbalance (~5 % de clase 0)."""
    rs = fe.RANDOM_STATE
    return {
        "Regresion Logistica": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=rs
        ),
        "Arbol de Decision": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=30, class_weight="balanced", random_state=rs
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=rs,
        ),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.05,
            max_iter=200,
            class_weight="balanced",
            random_state=rs,
        ),
    }


def build_model(estimador) -> Pipeline:
    """Arma el pipeline completo: features -> preprocesador (ColumnTransformer) -> modelo."""
    pipeline_features = fe.construir_pipeline_features()
    return Pipeline(pipeline_features.steps + [("modelo", estimador)])


def summarize_classification(y_true, y_pred, y_proba, umbral: float = UMBRAL_DECISION) -> dict:
    """Calcula las métricas de clasificación.

    ``y_proba`` es la probabilidad de la clase 1 (paga a tiempo). La clase de interés para el
    negocio es la 0 (no paga a tiempo), por eso se reportan precision / recall / F1 de la clase 0.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_clase0": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_clase0": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_clase0": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc_clase0": average_precision_score(1 - y_true, 1 - y_proba),
    }


def entrenar_y_evaluar(nombre, estimador, X_train, y_train, X_test, y_test) -> tuple[dict, Pipeline]:
    """Entrena un modelo, lo evalúa en train/test y mide consistencia (CV) y escalabilidad."""
    modelo = build_model(estimador)

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=fe.RANDOM_STATE)
    scores_cv = cross_val_score(modelo, X_train, y_train, cv=cv, scoring="roc_auc")

    inicio = time.perf_counter()
    modelo.fit(X_train, y_train)
    tiempo_entrenamiento = time.perf_counter() - inicio

    inicio = time.perf_counter()
    proba_test = modelo.predict_proba(X_test)[:, 1]
    tiempo_prediccion_1000 = (time.perf_counter() - inicio) / len(X_test) * 1000
    pred_test = (proba_test >= UMBRAL_DECISION).astype(int)
    proba_train = modelo.predict_proba(X_train)[:, 1]

    metricas = summarize_classification(y_test, pred_test, proba_test)
    fila = {"modelo": nombre}
    fila.update({f"{k}_test": v for k, v in metricas.items()})
    fila["roc_auc_train"] = roc_auc_score(y_train, proba_train)
    fila["roc_auc_cv_media"] = scores_cv.mean()
    fila["roc_auc_cv_std"] = scores_cv.std()
    fila["brecha_train_test"] = fila["roc_auc_train"] - fila["roc_auc_test"]
    fila["tiempo_entrenamiento_s"] = tiempo_entrenamiento
    fila["tiempo_prediccion_1000_s"] = tiempo_prediccion_1000
    return fila, modelo


def _normalizar(serie: pd.Series) -> pd.Series:
    """Min-max a [0, 1]; si todos son iguales devuelve 1."""
    rango = serie.max() - serie.min()
    return pd.Series(1.0, index=serie.index) if rango == 0 else (serie - serie.min()) / rango


def calcular_puntaje_seleccion(resumen: pd.DataFrame) -> pd.DataFrame:
    """Agrega puntajes de performance, consistency y scalability y un puntaje final ponderado."""
    resumen = resumen.copy()
    resumen["score_performance"] = _normalizar(resumen["roc_auc_test"])
    # consistencia: poca variación entre folds y poca brecha train-test
    inestabilidad = resumen["roc_auc_cv_std"] + resumen["brecha_train_test"].clip(lower=0)
    resumen["score_consistency"] = 1 - _normalizar(inestabilidad)
    resumen["score_scalability"] = 1 - _normalizar(np.log1p(resumen["tiempo_entrenamiento_s"]))
    resumen["score_final"] = (
        PESOS_SELECCION["performance"] * resumen["score_performance"]
        + PESOS_SELECCION["consistency"] * resumen["score_consistency"]
        + PESOS_SELECCION["scalability"] * resumen["score_scalability"]
    )
    return resumen.sort_values("score_final", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------
def graficar_comparacion_metricas(resumen: pd.DataFrame, ruta) -> None:
    metricas = ["roc_auc_test", "pr_auc_clase0_test", "recall_clase0_test", "f1_clase0_test"]
    datos = resumen.set_index("modelo")[metricas]
    ax = datos.plot(kind="bar", figsize=(11, 5), width=0.8)
    ax.set_title("Comparación de modelos en el conjunto de prueba")
    ax.set_ylabel("Valor de la métrica")
    ax.set_xlabel("")
    plt.xticks(rotation=15)
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(ruta, dpi=130)
    plt.close()


def graficar_curvas(modelos: dict, X_test, y_test, ruta) -> None:
    fig, ejes = plt.subplots(1, 2, figsize=(13, 5))
    for nombre, modelo in modelos.items():
        proba = modelo.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, proba)
        ejes[0].plot(fpr, tpr, label=f"{nombre} (AUC={roc_auc_score(y_test, proba):.3f})")
        precision, recall, _ = precision_recall_curve(1 - y_test, 1 - proba)
        ejes[1].plot(recall, precision, label=nombre)
    ejes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    ejes[0].set(title="Curva ROC", xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos")
    ejes[1].set(title="Curva Precision-Recall (clase 0: no paga)", xlabel="Recall", ylabel="Precision")
    ejes[0].legend(fontsize=8)
    ejes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(ruta, dpi=130)
    plt.close()


def graficar_tiempos_y_consistencia(resumen: pd.DataFrame, ruta) -> None:
    fig, ejes = plt.subplots(1, 2, figsize=(12, 4.5))
    ejes[0].bar(resumen["modelo"], resumen["tiempo_entrenamiento_s"], color="#4c78a8")
    ejes[0].set(title="Scalability: tiempo de entrenamiento (s)")
    ejes[0].tick_params(axis="x", rotation=20)
    ejes[1].bar(
        resumen["modelo"], resumen["roc_auc_cv_media"], yerr=resumen["roc_auc_cv_std"],
        color="#f58518", capsize=5,
    )
    ejes[1].set(title=f"Consistency: ROC-AUC medio en CV ({N_FOLDS} folds) ± desv. estándar")
    ejes[1].tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(ruta, dpi=130)
    plt.close()


def graficar_matriz_confusion(modelo, X_test, y_test, nombre, ruta) -> None:
    pred = (modelo.predict_proba(X_test)[:, 1] >= UMBRAL_DECISION).astype(int)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["No paga (0)", "Paga (1)"], cmap="Blues", ax=ax
    )
    ax.set_title(f"Matriz de confusión - {nombre}")
    plt.tight_layout()
    plt.savefig(ruta, dpi=130)
    plt.close()


# ---------------------------------------------------------------------------
# Flujo principal
# ---------------------------------------------------------------------------
def main() -> pd.DataFrame:
    fe.DIRECTORIO_ARTEFACTOS.mkdir(exist_ok=True)
    DIRECTORIO_FIGURAS.mkdir(exist_ok=True)

    X_train, X_test, y_train, y_test = fe.generar_datasets()
    print(f"Entrenamiento: {X_train.shape} | Prueba: {X_test.shape}")

    filas, modelos_entrenados = [], {}
    for nombre, estimador in obtener_modelos().items():
        print(f"-> Entrenando {nombre} ...")
        fila, modelo = entrenar_y_evaluar(nombre, estimador, X_train, y_train, X_test, y_test)
        filas.append(fila)
        modelos_entrenados[nombre] = modelo

    resumen = calcular_puntaje_seleccion(pd.DataFrame(filas))
    resumen.round(4).to_csv(fe.DIRECTORIO_ARTEFACTOS / "model_comparison.csv", index=False)
    print("\n=== Tabla resumen de evaluación ===")
    columnas = [
        "modelo", "roc_auc_test", "pr_auc_clase0_test", "recall_clase0_test", "f1_clase0_test",
        "roc_auc_cv_media", "roc_auc_cv_std", "tiempo_entrenamiento_s", "score_final",
    ]
    print(resumen[columnas].round(4).to_string(index=False))

    mejor_nombre = resumen.loc[0, "modelo"]
    mejor_modelo = modelos_entrenados[mejor_nombre]
    print(f"\n*** Mejor modelo: {mejor_nombre} ***")

    graficar_comparacion_metricas(resumen, DIRECTORIO_FIGURAS / "comparacion_metricas.png")
    graficar_curvas(modelos_entrenados, X_test, y_test, DIRECTORIO_FIGURAS / "curvas_roc_pr.png")
    graficar_tiempos_y_consistencia(resumen, DIRECTORIO_FIGURAS / "tiempo_y_consistencia.png")
    graficar_matriz_confusion(
        mejor_modelo, X_test, y_test, mejor_nombre, DIRECTORIO_FIGURAS / "matriz_confusion_mejor.png"
    )

    joblib.dump(mejor_modelo, fe.RUTA_MODELO)
    info = {
        "modelo": mejor_nombre,
        "umbral_decision": UMBRAL_DECISION,
        "variable_objetivo": fe.TARGET,
        "metricas_test": {
            k.replace("_test", ""): round(float(resumen.loc[0, k]), 4)
            for k in resumen.columns
            if k.endswith("_test") and not k.startswith("brecha")
        },
        "features": fe.FEATURES,
        "columnas_excluidas": fe.COLUMNAS_EXCLUIDAS,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    fe.RUTA_INFO_MODELO.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Modelo guardado en: {fe.RUTA_MODELO}")
    return resumen


if __name__ == "__main__":
    main()

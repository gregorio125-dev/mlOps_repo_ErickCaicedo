import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import ft_engineering as fe
import model_training_evaluation as mt


def test_summarize_classification_devuelve_metricas_validas():
    y = np.array([1, 1, 1, 0, 0, 1, 0, 1])
    proba = np.array([0.9, 0.8, 0.7, 0.2, 0.4, 0.6, 0.6, 0.95])
    metricas = mt.summarize_classification(y, (proba >= 0.5).astype(int), proba)
    for clave in ["accuracy", "recall_clase0", "f1_clase0", "roc_auc", "pr_auc_clase0", "mcc"]:
        assert clave in metricas
        assert -1 <= metricas[clave] <= 1
    assert metricas["roc_auc"] > 0.8


def test_catalogo_de_modelos():
    modelos = mt.obtener_modelos()
    assert len(modelos) >= 3
    assert "Regresion Logistica" in modelos


def test_build_model_entrena_y_predice(datos_crudos):
    muestra = datos_crudos.sample(1500, random_state=3)
    X, y = muestra.drop(columns=[fe.TARGET]), muestra[fe.TARGET].astype(int)
    modelo = mt.build_model(LogisticRegression(max_iter=300, class_weight="balanced"))
    modelo.fit(X, y)
    proba = modelo.predict_proba(X)[:, 1]
    assert proba.shape == (1500,)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_seleccion_prefiere_mejor_balance():
    resumen = pd.DataFrame(
        {
            "modelo": ["A", "B"],
            "roc_auc_test": [0.80, 0.60],
            "roc_auc_cv_std": [0.01, 0.05],
            "brecha_train_test": [0.02, 0.20],
            "tiempo_entrenamiento_s": [1.0, 10.0],
        }
    )
    ordenado = mt.calcular_puntaje_seleccion(resumen)
    assert ordenado.loc[0, "modelo"] == "A"
    assert ordenado.loc[0, "score_final"] > ordenado.loc[1, "score_final"]


def test_normalizar_valores_iguales():
    assert (mt._normalizar(pd.Series([2.0, 2.0])) == 1.0).all()


def test_main_completo_con_un_modelo(tmp_path, monkeypatch):
    """Ejecuta todo el flujo con un solo modelo rápido y artefactos en una carpeta temporal."""
    monkeypatch.setattr(fe, "DIRECTORIO_ARTEFACTOS", tmp_path)
    monkeypatch.setattr(fe, "RUTA_MODELO", tmp_path / "best_model.joblib")
    monkeypatch.setattr(fe, "RUTA_INFO_MODELO", tmp_path / "best_model_info.json")
    monkeypatch.setattr(mt, "DIRECTORIO_FIGURAS", tmp_path / "figures")
    monkeypatch.setattr(
        mt, "obtener_modelos",
        lambda: {"Regresion Logistica": LogisticRegression(max_iter=300, class_weight="balanced")},
    )
    resumen = mt.main()
    assert len(resumen) == 1
    assert (tmp_path / "best_model.joblib").exists()
    assert (tmp_path / "model_comparison.csv").exists()
    assert (tmp_path / "figures" / "curvas_roc_pr.png").exists()
    info = json.loads((tmp_path / "best_model_info.json").read_text(encoding="utf-8"))
    assert info["modelo"] == "Regresion Logistica"

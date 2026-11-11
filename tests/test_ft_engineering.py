import numpy as np
import pandas as pd

import ft_engineering as fe


def _fila(**cambios):
    base = {
        "tipo_credito": 4, "fecha_prestamo": "2025-01-07 14:40:36", "capital_prestado": 1_000_000.0,
        "plazo_meses": 12, "edad_cliente": 30, "tipo_laboral": "Empleado", "salario_cliente": 2_000_000,
        "total_otros_prestamos": 500_000, "cuota_pactada": 100_000, "puntaje": 95.2,
        "puntaje_datacredito": 800.0, "cant_creditosvigentes": 2, "huella_consulta": 3,
        "saldo_mora": 0.0, "saldo_total": 0.0, "saldo_principal": np.nan, "saldo_mora_codeudor": 0.0,
        "creditos_sectorFinanciero": 1, "creditos_sectorCooperativo": 0, "creditos_sectorReal": 1,
        "promedio_ingresos_datacredito": 1_500_000.0, "tendencia_ingresos": "Creciente",
    }
    base.update(cambios)
    return pd.DataFrame([base])


def test_cargar_datos_tiene_objetivo(datos_crudos):
    assert fe.TARGET in datos_crudos.columns
    assert len(datos_crudos) > 10_000


def test_edad_centinela_pasa_a_nulo():
    limpio = fe.limpiar_datos(_fila(edad_cliente=122))
    assert limpio["edad_cliente"].isna().all()


def test_salario_imposible_pasa_a_nulo():
    assert fe.limpiar_datos(_fila(salario_cliente=0))["salario_cliente"].isna().all()
    assert fe.limpiar_datos(_fila(salario_cliente=22_000_000_000))["salario_cliente"].isna().all()


def test_tendencia_invalida_pasa_a_nulo():
    assert fe.limpiar_datos(_fila(tendencia_ingresos="8315"))["tendencia_ingresos"].isna().all()
    assert fe.limpiar_datos(_fila(tendencia_ingresos=" creciente "))["tendencia_ingresos"].iloc[0] == "Creciente"


def test_tokens_de_nulo_se_unifican():
    limpio = fe.limpiar_datos(_fila(tipo_laboral="N/A"))
    assert limpio["tipo_laboral"].isna().all()


def test_saldo_principal_estructural_se_rellena_con_cero():
    limpio = fe.limpiar_datos(_fila(saldo_total=0.0, saldo_principal=np.nan))
    assert limpio["saldo_principal"].iloc[0] == 0.0


def test_tipo_credito_queda_como_texto():
    assert fe.limpiar_datos(_fila(tipo_credito=9))["tipo_credito"].iloc[0] == "9"


def test_variables_derivadas_y_division_segura():
    df = fe.limpiar_datos(_fila(salario_cliente=0))
    derivadas = fe.crear_variables_derivadas(df)
    assert np.isnan(derivadas["cuota_sobre_salario"].iloc[0])  # salario inválido -> NaN, no infinito
    normal = fe.crear_variables_derivadas(fe.limpiar_datos(_fila()))
    assert normal["cuota_sobre_salario"].iloc[0] == 0.05
    assert normal["total_creditos_sectores"].iloc[0] == 2
    assert normal["tiene_mora"].iloc[0] == 0


def test_mora_activa_la_bandera():
    derivadas = fe.crear_variables_derivadas(fe.limpiar_datos(_fila(saldo_mora=500.0)))
    assert derivadas["tiene_mora"].iloc[0] == 1
    assert derivadas["mora_total"].iloc[0] == 500.0


def test_preparar_features_devuelve_solo_predictoras_y_completa_faltantes():
    resultado = fe.preparar_features(_fila().drop(columns=["edad_cliente", "puntaje"]))
    assert list(resultado.columns) == fe.FEATURES
    assert "puntaje" not in resultado.columns  # excluida por fuga de datos
    assert resultado["edad_cliente"].isna().all()


def test_validar_datos_despues_de_limpiar_no_tiene_invalidos(datos_crudos):
    antes = fe.validar_datos(datos_crudos)["registros_invalidos"].sum()
    despues = fe.validar_datos(fe.limpiar_datos(datos_crudos))["registros_invalidos"].sum()
    assert antes > 0
    assert despues == 0


def test_generar_datasets_estratificado():
    X_train, X_test, y_train, y_test = fe.generar_datasets()
    assert len(X_train) + len(X_test) == len(fe.cargar_datos())
    assert abs(y_train.mean() - y_test.mean()) < 0.01
    assert fe.TARGET not in X_train.columns


def test_pipeline_de_features_no_deja_nulos(datos_crudos):
    pipeline = fe.construir_pipeline_features()
    muestra = datos_crudos.sample(500, random_state=1).drop(columns=[fe.TARGET])
    matriz = pipeline.fit_transform(muestra)
    assert not np.isnan(matriz).any()
    assert matriz.shape[0] == 500


def test_datasets_transformados():
    Xt_train, Xt_test, y_train, y_test, _ = fe.generar_datasets_transformados()
    assert Xt_train.shape[1] == Xt_test.shape[1]
    assert not Xt_train.isna().any().any()
    assert len(Xt_train) == len(y_train)


def test_main_imprime_resumen(capsys):
    fe.main()
    assert "Train:" in capsys.readouterr().out

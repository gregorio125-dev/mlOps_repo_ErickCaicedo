# Predicción de pago a tiempo de créditos (MLOps pipeline)

Proyecto final de **Machine Learning** – Universidad Católica Luis Amigo.
Modelo supervisado que predice si un cliente **pagará a tiempo** su crédito (`Pago_atiempo`), con pipeline reproducible, API, monitoreo de *data drift* y contenedor Docker.

## 1. Caso de negocio

Una entidad financiera otorga créditos de consumo. Cada crédito que no se paga a tiempo genera costos de cobranza y pérdidas. El objetivo es **anticipar el incumplimiento al momento de la aprobación** para decidir con mejor criterio (aprobar, ajustar monto/plazo, pedir garantías).

* **Tipo de problema:** clasificación binaria (1 = paga a tiempo, 0 = no paga a tiempo).
* **Datos:** 10.763 créditos (nov-2024 a abr-2026), 22 variables + objetivo.
* **Desbalance:** solo 4,7 % de los créditos son de la clase 0 (no pagan). Por eso la métrica principal **no** es la exactitud.

## 2. Estructura del repositorio

```
├── mlops_pipeline/
│   ├── Cargar_datos.ipynb            # carga del CSV
│   ├── comprension_eda.ipynb         # EDA, calidad de datos y reglas de validación
│   ├── ft_engineering.py             # limpieza, variables derivadas, ColumnTransformer, train/test
│   ├── model_training_evaluation.py  # entrena, compara y selecciona el mejor modelo
│   ├── model_deploy.py               # API FastAPI (/predict, /predict_csv, /health)
│   └── model_monitoring.py           # data drift (KS, PSI, Jensen-Shannon, Chi2) + app Streamlit
├── Base_de_datos.csv
├── requirements.txt
├── .gitignore
├── setup.bat
└── readme.md
```

Archivos adicionales que pide el enunciado o que dan soporte a la calidad: `Dockerfile`, `.dockerignore`, `Jenkinsfile`, `sonar-project.properties`, `.github/workflows/sonarcloud.yml` y la carpeta `tests/`.
`mlops_pipeline/artifacts/` guarda lo que genera el entrenamiento (modelo, tablas y gráficos).

## 3. Cómo ejecutarlo

**Windows:** doble clic en `setup.bat` (crea el entorno virtual, instala dependencias y entrena el modelo).

**Manual (cualquier sistema):**
```bash
python -m venv venv
venv\Scripts\activate          # Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
python mlops_pipeline/ft_engineering.py                 # genera train/test
python mlops_pipeline/model_training_evaluation.py      # entrena y guarda el mejor modelo
python mlops_pipeline/model_monitoring.py               # monitoreo por consola
streamlit run mlops_pipeline/model_monitoring.py        # tablero de monitoreo
uvicorn --app-dir mlops_pipeline model_deploy:app --port 8000   # API en http://localhost:8000/docs
python -m pytest tests --cov=mlops_pipeline             # pruebas y cobertura
```

## 4. Proceso y principales hallazgos

1. **Calidad de datos (EDA).** Se encontraron valores centinela (edad 122, 144 casos), salarios imposibles (hasta 22 mil millones), números dentro de `tendencia_ingresos` (que debe ser Creciente/Estable/Decreciente) y puntajes de la central fuera de rango. Se corrigen con reglas explícitas (`ft_engineering.limpiar_datos`).
2. **Fuga de datos.** La variable `puntaje` separa perfectamente a quienes pagan de quienes no (**AUC = 1,0** por sí sola): se calcula con información posterior al pago. **Se excluye del modelo**; incluirla daría un modelo "perfecto" pero inútil en producción.
3. **Señal débil pero real.** Las demás variables aportan poco individualmente (mejor AUC univariado: puntaje DataCrédito ≈ 0,62; luego consultas recientes, edad, monto y plazo). Los clientes que no pagan tienen menor puntaje, más consultas recientes, son más jóvenes y piden montos mayores. Estar en mora en el momento del desembolso baja el pago a tiempo del 95 % al 64 % (pocos casos: 55).
4. **Ingeniería de características.** Pipeline `ColumnTransformer` con tres ramas: numéricas (imputación por mediana + indicador de nulo + escalado), categóricas nominales (imputación + OneHot) y ordinales (imputación + OrdinalEncoder). Variables derivadas: cuota/salario, capital/salario, deuda/salario, salario/ingreso reportado, total de créditos, mora total y bandera de mora.
5. **Modelos comparados** (5-fold CV + prueba 20 %): Regresión Logística, Árbol de Decisión, Random Forest y Hist Gradient Boosting, elegidos por *performance* (60 %), *consistency* (25 %) y *scalability* (15 %).

| Modelo | ROC-AUC test | ROC-AUC CV (media ± std) | Recall clase 0 | Tiempo entrenamiento |
|---|---|---|---|---|
| **Regresión Logística (seleccionado)** | **0,695** | 0,663 ± 0,013 | 0,62 | 0,13 s |
| Random Forest | 0,656 | 0,672 ± 0,019 | 0,05 | 3,4 s |
| Hist Gradient Boosting | 0,630 | 0,648 ± 0,022 | 0,36 | 1,4 s |
| Árbol de Decisión | 0,598 | 0,579 ± 0,027 | 0,53 | 0,17 s |

   La regresión logística gana por rendimiento, estabilidad y velocidad; Random Forest sobreajusta (AUC train = 1,0). **Limitación:** con umbral 0,5 detecta ~62 % de los casos de no pago pero con baja precisión (~8 %), así que es una herramienta de **priorización de riesgo**, no de rechazo automático. El umbral puede ajustarse según el costo de cada error.
6. **Monitoreo.** Se compara el histórico (primer 50 % de los créditos) contra periodos posteriores con KS, PSI, Jensen-Shannon y Chi-cuadrado. Se detecta **drift creciente** en monto, plazo, salario y deuda: el número de variables en estado crítico sube con el tiempo y el sistema recomienda reentrenar. Los periodos se agrupan hasta tener ≥ 300 registros porque con muestras pequeñas el PSI se infla por ruido.

## 5. API

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"registros":[{"tipo_credito":4,"capital_prestado":1852560,"plazo_meses":12,"edad_cliente":32,
        "tipo_laboral":"Empleado","salario_cliente":3500000,"cuota_pactada":128650,
        "puntaje_datacredito":795,"tendencia_ingresos":"Creciente"}]}'
```
Respuesta: predicción (0/1), etiqueta y probabilidades. También acepta CSV en `/predict_csv` (hasta 10.000 filas por solicitud) y expone `/health`.

## 6. Docker

```bash
docker build -t creditos-api .
docker run -p 8000:8000 creditos-api
```
La imagen incluye código, dependencias, Uvicorn y entrena el modelo durante la construcción.

## 7. Calidad (SonarCloud) y CI/CD (Jenkins)

* **SonarCloud:** importar el repositorio en https://sonarcloud.io, copiar organización y *project key* en `sonar-project.properties`, y crear el secreto `SONAR_TOKEN` en GitHub. El workflow ejecuta pruebas, cobertura y análisis en cada push.
* **Jenkins:** `Jenkinsfile` clona el repo, valida la estructura de carpetas, instala dependencias, corre pruebas, (opcional) SonarCloud, construye la imagen en `master`/`certification` y notifica al terminar.

## 8. Ramas

`developer` (desarrollo) → `certification` (validación) → `master` (producción).

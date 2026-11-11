@echo off
REM ============================================================
REM setup.bat - Prepara el entorno del proyecto en Windows
REM   1) crea el entorno virtual  2) instala requirements.txt
REM   3) registra el kernel de Jupyter  4) entrena el modelo
REM ============================================================
cd /d "%~dp0"

if not exist venv (
    echo [1/4] Creando entorno virtual...
    python -m venv venv
    if errorlevel 1 goto :error
) else (
    echo [1/4] El entorno virtual ya existe.
)

call venv\Scripts\activate.bat

echo [2/4] Instalando dependencias...
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 goto :error

echo [3/4] Registrando kernel de Jupyter...
python -m ipykernel install --user --name mlops_creditos --display-name "Python (mlops_creditos)"

echo [4/4] Entrenando el modelo (genera mlops_pipeline\artifacts)...
python mlops_pipeline\model_training_evaluation.py
if errorlevel 1 goto :error

echo.
echo ============================================================
echo  Listo. Activa el entorno con:  venv\Scripts\activate.bat
echo  API:        uvicorn --app-dir mlops_pipeline model_deploy:app --port 8000
echo  Monitoreo:  streamlit run mlops_pipeline\model_monitoring.py
echo  Pruebas:    python -m pytest tests
echo ============================================================
goto :eof

:error
echo.
echo ERROR: el proceso se detuvo. Revisa el mensaje anterior.
exit /b 1

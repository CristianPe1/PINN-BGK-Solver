@echo off
:: Script batch para ejecutar múltiples modelos en Windows
setlocal enabledelayedexpansion

:: Configuración global
set PYTHON=python
set SRC_DIR=src
set CONFIG_DIR=config

:: Crear directorio para logs
if not exist logs mkdir logs

:: Mostrar menú
echo =====================================
echo PINN-BGK - Ejecución de Modelos
echo =====================================
echo.
echo 1. Entrenar todos los modelos secuencialmente
echo 2. Entrenar modelo de Burgers
echo 3. Entrenar modelo de Kovasznay
echo 4. Entrenar modelo de Taylor-Green
echo 5. Entrenar modelo de Cavity Flow
echo 6. Generar datos para todos los problemas
echo 7. Evaluar modelo reciente
echo 8. Ejecutar experimento con diferentes batch sizes
echo.
echo X. Salir
echo.
set /p OPTION="Seleccione una opción (1-8, X): "

:: Procesar opción seleccionada
if "%OPTION%"=="1" goto train_all
if "%OPTION%"=="2" goto train_burgers
if "%OPTION%"=="3" goto train_kovasznay
if "%OPTION%"=="4" goto train_taylor_green
if "%OPTION%"=="5" goto train_cavity_flow
if "%OPTION%"=="6" goto generate_all
if "%OPTION%"=="7" goto evaluate_model
if "%OPTION%"=="8" goto experiment_batch
if /i "%OPTION%"=="X" goto end

echo Opción inválida. Intente nuevamente.
goto end

:train_all
echo Entrenando todos los modelos secuencialmente...
call :train_burgers
call :train_kovasznay
call :train_taylor_green
call :train_cavity_flow
echo.
echo Todos los modelos han sido entrenados.
goto end

:train_burgers
echo.
echo Entrenando modelo para ecuación de Burgers...
:: Crear configuración temporal
set CONFIG_FILE=%CONFIG_DIR%\burgers_config.yaml
copy %CONFIG_DIR%\config.yaml %CONFIG_FILE%
echo selected_model: "pinn_v1" >> %CONFIG_FILE%
%PYTHON% %SRC_DIR%\main.py --mode train --config %CONFIG_FILE% > logs\burgers_train.log 2>&1
if %ERRORLEVEL% NEQ 0 echo Error durante el entrenamiento. Revise logs\burgers_train.log
exit /b

:train_kovasznay
echo.
echo Entrenando modelo para flujo de Kovasznay...
:: Crear configuración temporal
set CONFIG_FILE=%CONFIG_DIR%\kovasznay_config.yaml
copy %CONFIG_DIR%\config.yaml %CONFIG_FILE%
echo selected_model: "kovasznay" >> %CONFIG_FILE%
%PYTHON% %SRC_DIR%\main.py --mode train --config %CONFIG_FILE% > logs\kovasznay_train.log 2>&1
if %ERRORLEVEL% NEQ 0 echo Error durante el entrenamiento. Revise logs\kovasznay_train.log
exit /b

:train_taylor_green
echo.
echo Entrenando modelo para vórtice de Taylor-Green...
:: Crear configuración temporal
set CONFIG_FILE=%CONFIG_DIR%\taylor_green_config.yaml
copy %CONFIG_DIR%\config.yaml %CONFIG_FILE%
echo selected_model: "taylor_green" >> %CONFIG_FILE%
%PYTHON% %SRC_DIR%\main.py --mode train --config %CONFIG_FILE% > logs\taylor_green_train.log 2>&1
if %ERRORLEVEL% NEQ 0 echo Error durante el entrenamiento. Revise logs\taylor_green_train.log
exit /b

:train_cavity_flow
echo.
echo Entrenando modelo para flujo en cavidad...
:: Crear configuración temporal
set CONFIG_FILE=%CONFIG_DIR%\cavity_flow_config.yaml
copy %CONFIG_DIR%\config.yaml %CONFIG_FILE%
echo selected_model: "cavity_flow" >> %CONFIG_FILE%
%PYTHON% %SRC_DIR%\main.py --mode train --config %CONFIG_FILE% > logs\cavity_flow_train.log 2>&1
if %ERRORLEVEL% NEQ 0 echo Error durante el entrenamiento. Revise logs\cavity_flow_train.log
exit /b

:generate_all
echo Generando datos para todos los problemas...
:: Generar datos para Burgers
echo - Burgers...
%PYTHON% %SRC_DIR%\main.py --mode generate --config %CONFIG_DIR%\config.yaml > logs\burgers_generate.log 2>&1
:: Modificar la configuración para generar datos de Kovasznay
:: (repite para otros tipos)
echo.
echo Datos generados. Revise la carpeta data\synthetic para los resultados.
goto end

:evaluate_model
echo Evaluando modelo más reciente...
%PYTHON% %SRC_DIR%\main.py --mode evaluate > logs\evaluate.log 2>&1
echo.
echo Evaluación completada. Revise los resultados en la carpeta de salida.
goto end

:experiment_batch
echo Ejecutando experimento con diferentes batch sizes...
for %%b in (16 32 64 128 256) do (
    echo - Entrenando con batch_size=%%b
    set CONFIG_FILE=%CONFIG_DIR%\batch_%%b.yaml
    copy %CONFIG_DIR%\config.yaml !CONFIG_FILE!
    echo selected_model: "pinn_v1" >> !CONFIG_FILE!
    echo training: >> !CONFIG_FILE!
    echo   batch_size: %%b >> !CONFIG_FILE!
    %PYTHON% %SRC_DIR%\main.py --mode train --config !CONFIG_FILE! > logs\batch_%%b_train.log 2>&1
)
echo.
echo Experimento completado. Revise los logs para resultados.
goto end

:end
echo.
echo Programa finalizado.
endlocal

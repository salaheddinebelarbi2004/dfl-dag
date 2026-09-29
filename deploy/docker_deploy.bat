@echo off
REM Docker Deployment Script for Decentralized P2P Federated Learning
REM ================================================================

setlocal enabledelayedexpansion

cd /d "%~dp0.."
set "COMPOSE_FILE=deploy\docker-compose.yml"

echo.
echo ========================================
echo Docker P2P Federated Learning Deployment
echo ========================================
echo.

REM Check if Docker is available
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker not found. Install Docker Desktop for Windows first.
    echo         https://docs.docker.com/desktop/install/windows-install/
    pause
    exit /b 1
)
echo [OK] Docker found

REM Check if Docker Compose is available
docker compose version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker Compose not found.
    pause
    exit /b 1
)
echo [OK] Docker Compose found

REM Check for required files
if not exist "model.pt" (
    if exist "model.safetensors" (
        echo [OK] Using model.safetensors
    ) else (
        echo [WARNING] Neither model.pt nor model.safetensors found
        echo          Training may fail without a model file
    )
) else (
    echo [OK] model.pt found
)

if not exist "real_federated_data\client_0.npz" (
    echo [ERROR] real_federated_data directory not found or missing client files
    pause
    exit /b 1
)
echo [OK] real_federated_data found

echo.
echo ========================================
echo Building Docker image...
echo ========================================
echo.

docker compose -f %COMPOSE_FILE% build

if %errorlevel% neq 0 (
    echo [ERROR] Build failed
    pause
    exit /b 1
)
echo [OK] Image built

echo.
echo ========================================
echo Starting 12 client containers...
echo ========================================
echo.

docker compose -f %COMPOSE_FILE% up -d

if %errorlevel% neq 0 (
    echo [ERROR] Failed to start containers
    pause
    exit /b 1
)

echo.
echo [OK] All containers started
echo.
echo Monitoring logs (press Ctrl+C to stop watching):
echo ========================================
echo.

timeout /t 5 /nobreak

:monitor
cls
echo.
echo ========================================
echo Container Status
echo ========================================
echo.
docker compose -f %COMPOSE_FILE% ps
echo.
echo ========================================
echo Recent Logs (client-0, client-4, client-8)
echo ========================================
echo.
echo --- client-0 (Coordinator) ---
docker compose -f %COMPOSE_FILE% logs --tail=5 client-0 2>nul
echo.
echo --- client-4 (Shard 1 Chief) ---
docker compose -f %COMPOSE_FILE% logs --tail=5 client-4 2>nul
echo.
echo --- client-8 (Shard 2 Chief) ---
docker compose -f %COMPOSE_FILE% logs --tail=5 client-8 2>nul
echo.
echo ========================================
echo Commands:
echo   docker compose -f %COMPOSE_FILE% logs -f   Live logs
echo   docker compose -f %COMPOSE_FILE% ps        Container status
echo   docker_deploy stop       Stop all containers
echo   docker_deploy results    View results
echo ========================================
echo.

:menu
echo.
echo Select action:
echo   1. Refresh status
echo   2. View all logs (follow)
echo   3. Stop all containers
echo   4. View results
echo   5. Exit
echo.
choice /c 12345 /n /m "Enter choice (1-5): "
if errorlevel 5 exit /b 0
if errorlevel 4 goto show_results
if errorlevel 3 goto stop_all
if errorlevel 2 goto follow_logs
if errorlevel 1 goto monitor

:follow_logs
docker compose -f %COMPOSE_FILE% logs -f
goto monitor

:stop_all
echo.
echo Stopping all containers...
docker compose -f %COMPOSE_FILE% down
echo [OK] Containers stopped
pause
exit /b 0

:show_results
echo.
echo ========================================
echo Training Results
echo ========================================
echo.
if exist "results\results_client_0.json" (
    for /l %%i in (0,1,11) do (
        if exist "results\results_client_%%i.json" (
            echo --- client-%%i ---
            docker run --rm -v "!cd!\results:/results" alpine cat /results/results_client_%%i.json 2>nul | python -c "import sys,json; d=json.load(sys.stdin); print(f'  Final Acc: {d.get(\"final_test_accuracy\", \"N/A\"):.4f}, Rounds: {len(d.get(\"rounds\",[]))}')" 2>nul
        )
    )
) else (
    echo Results not yet available. Training may still be in progress.
)
echo.
pause
goto monitor

@echo off
REM Batch script to start Flowr P2P Nodes
REM Usage: start_nodes.bat

setlocal enabledelayedexpansion

echo.
echo ========================================
echo Flowr P2P Network Startup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python not found. Please install Python first.
    pause
    exit /b 1
)

REM Start Node 1 - Port 5000
start "Node-1 (5000)" cmd /k "python p2p_node.py 127.0.0.1 5000"
timeout /t 2 /nobreak

REM Start Node 2 - Port 5001
start "Node-2 (5001)" cmd /k "python p2p_node.py 127.0.0.1 5001"
timeout /t 2 /nobreak

REM Start Node 3 - Port 5002
start "Node-3 (5002)" cmd /k "python p2p_node.py 127.0.0.1 5002"
timeout /t 2 /nobreak

echo.
echo ========================================
echo All nodes started!
echo ========================================
echo.
echo Next steps:
echo 1. Connect nodes using: connect 127.0.0.1 5000
echo 2. Send tasks: task sum [1,2,3]
echo 3. Check results: result ^<task_id^>
echo 4. View peers: peers
echo.
echo ========================================
echo.

pause

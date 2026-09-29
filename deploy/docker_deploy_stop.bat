@echo off
REM Stop all Docker P2P containers
cd /d "%~dp0.."
echo Stopping all containers...
docker compose -f deploy\docker-compose.yml down
echo [OK] Containers stopped

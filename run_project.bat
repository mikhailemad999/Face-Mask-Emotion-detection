@echo off
setlocal enabledelayedexpansion
title FaceGuard.AI — Project Launcher
color 0A

echo.
echo  ╔══════════════════════════════════════════════════════════════╗
echo  ║       FaceGuard.AI — Face Mask ^& Emotion Detection         ║
echo  ║                  Project Launcher v1.0                      ║
echo  ╚══════════════════════════════════════════════════════════════╝
echo.

:: ── Get script directory (handles paths with spaces and &) ─────────────────
set "PROJECT_ROOT=%~dp0"
:: Remove trailing backslash
if "%PROJECT_ROOT:~-1%"=="\" set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

:: ── Step 0: Check prerequisites ────────────────────────────────────────────
echo [STEP 0] Checking prerequisites...

where docker >nul 2>&1
if %errorlevel% neq 0 (
    echo   [ERROR] Docker is not installed or not in PATH.
    echo   Please install Docker Desktop: https://www.docker.com/products/docker-desktop/
    goto :error_exit
)

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo   [ERROR] Python is not installed or not in PATH.
    echo   Please install Python 3.11+
    goto :error_exit
)

where node >nul 2>&1
if %errorlevel% neq 0 (
    echo   [ERROR] Node.js is not installed or not in PATH.
    echo   Please install Node.js v18+
    goto :error_exit
)

echo   [OK] Docker, Python, Node.js found.
echo.

:: ── Step 1: Copy .env if not present ───────────────────────────────────────
echo [STEP 1] Checking environment configuration...
if not exist "%PROJECT_ROOT%\.env" (
    echo   Copying .env.example to .env...
    copy "%PROJECT_ROOT%\.env.example" "%PROJECT_ROOT%\.env" >nul
    echo   [OK] .env created from template.
) else (
    echo   [OK] .env already exists.
)
echo.

:: ── Step 2: Start database containers ──────────────────────────────────────
echo [STEP 2] Starting database services (Docker Compose)...
pushd "%PROJECT_ROOT%"
docker compose up -d db mongo redis
popd

if %errorlevel% neq 0 (
    echo   [ERROR] Failed to start Docker containers.
    echo   Make sure Docker Desktop is running.
    goto :error_exit
)
echo   [OK] SQL Server, MongoDB, Redis containers are running.
echo.

:: ── Step 3: Wait for databases to be ready ─────────────────────────────────
echo [STEP 3] Waiting for databases to initialize...
echo   Giving SQL Server 8 seconds to start up...
ping -n 9 127.0.0.1 >nul
echo   [OK] Database startup wait complete.
echo.

:: ── Step 4: Install backend dependencies + migrations ──────────────────────
echo [STEP 4] Setting up Django backend...

:: Check if requirements are already installed by testing a key package
python -c "import django" >nul 2>&1
if %errorlevel% neq 0 (
    echo   Installing Python dependencies...
    pip install -r "%PROJECT_ROOT%\project\backend\requirements.txt" --quiet
) else (
    echo   [OK] Python dependencies already installed.
)

:: Run migrations & model registration
echo   Applying database migrations ^& registering models...
pushd "%PROJECT_ROOT%\project\backend"
python manage.py makemigrations detection --noinput >nul 2>&1
python manage.py migrate --noinput >nul 2>&1
python manage.py register_models >nul 2>&1
popd
echo   [OK] Migrations applied and ML models registered.
echo.

:: ── Step 5: Create and launch backend helper script ────────────────────────
echo [STEP 5] Starting Django backend server (port 8000)...

:: Write a temporary helper script that the start command can safely call
:: This avoids the & path issue with cmd /c nested quoting
> "%PROJECT_ROOT%\project\backend\.start_backend.cmd" (
    echo @echo off
    echo pushd "%%~dp0"
    echo python manage.py runserver 8000
    echo popd
)
start "FaceGuard - Django Backend" /min /d "%PROJECT_ROOT%\project\backend" .start_backend.cmd
echo   [OK] Backend started in background window.
echo.

:: ── Step 6: Install frontend dependencies (if needed) ──────────────────────
echo [STEP 6] Setting up React frontend...
pushd "%PROJECT_ROOT%\project\frontend"

if not exist "node_modules" (
    echo   Installing Node.js dependencies...
    call npm install --silent 2>nul
) else (
    echo   [OK] Node.js dependencies already installed.
)
popd
echo.

:: ── Step 7: Create and launch frontend helper script ───────────────────────
echo [STEP 7] Starting Vite frontend server (port 5173)...

:: Write a temporary helper script — uses node directly to avoid npm & path bug
> "%PROJECT_ROOT%\project\frontend\.start_frontend.cmd" (
    echo @echo off
    echo pushd "%%~dp0"
    echo node node_modules\vite\bin\vite.js
    echo popd
)
start "FaceGuard - Vite Frontend" /min /d "%PROJECT_ROOT%\project\frontend" .start_frontend.cmd
echo   [OK] Frontend started in background window.
echo.

:: ── Step 8: Wait for frontend to be ready ──────────────────────────────────
echo [STEP 8] Waiting for frontend to be ready...
echo.
set "RETRIES=0"
:wait_loop
set /a RETRIES+=1
if %RETRIES% gtr 30 (
    echo   [WARNING] Frontend took too long. Opening browser anyway...
    goto :open_browser
)
powershell -Command "try { $r = Invoke-WebRequest -Uri http://localhost:5173/ -UseBasicParsing -TimeoutSec 2; if ($r.StatusCode -eq 200) { exit 0 } } catch { exit 1 }" >nul 2>&1
if %errorlevel% neq 0 (
    <nul set /p =.
    ping -n 3 127.0.0.1 >nul
    goto :wait_loop
)
echo.

:open_browser
echo.
echo  ╔══════════════════════════════════════════════════════════════╗
echo  ║                  ALL SERVICES RUNNING!                      ║
echo  ╠══════════════════════════════════════════════════════════════╣
echo  ║                                                              ║
echo  ║  Frontend:  http://localhost:5173/                           ║
echo  ║  Backend:   http://localhost:8000/api/                       ║
echo  ║  Swagger:   http://localhost:8000/api/docs/                  ║
echo  ║  Admin:     http://localhost:8000/admin/                     ║
echo  ║                                                              ║
echo  ║  Databases:                                                  ║
echo  ║    SQL Server:  localhost:1433                                ║
echo  ║    MongoDB:     localhost:27017                               ║
echo  ║    Redis:       localhost:6379                                ║
echo  ║                                                              ║
echo  ╠══════════════════════════════════════════════════════════════╣
echo  ║  Press any key to STOP all services and exit.               ║
echo  ╚══════════════════════════════════════════════════════════════╝
echo.

:: Open the browser
start http://localhost:5173/

:: Wait for user to press a key to stop
pause >nul

:: ── Cleanup: Stop all services ─────────────────────────────────────────────
echo.
echo [CLEANUP] Stopping all services...

:: Kill the background server windows
taskkill /fi "WINDOWTITLE eq FaceGuard - Django Backend*" /f >nul 2>&1
taskkill /fi "WINDOWTITLE eq FaceGuard - Vite Frontend*" /f >nul 2>&1

:: Also kill any lingering python/node processes on those ports
for /f "tokens=5" %%p in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING 2^>nul') do (
    taskkill /pid %%p /f >nul 2>&1
)
for /f "tokens=5" %%p in ('netstat -ano ^| findstr :5173 ^| findstr LISTENING 2^>nul') do (
    taskkill /pid %%p /f >nul 2>&1
)

:: Stop Docker containers
pushd "%PROJECT_ROOT%"
docker compose stop db mongo redis >nul 2>&1
popd

:: Clean up helper scripts
del "%PROJECT_ROOT%\project\backend\.start_backend.cmd" >nul 2>&1
del "%PROJECT_ROOT%\project\frontend\.start_frontend.cmd" >nul 2>&1

echo [OK] All services stopped.
echo.
echo Goodbye!
ping -n 4 127.0.0.1 >nul
exit /b 0

:error_exit
echo.
echo [FAILED] Project could not start. Please fix the errors above.
pause
exit /b 1

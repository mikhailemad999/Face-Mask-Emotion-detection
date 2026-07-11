@echo off
title FaceGuard.AI — Stop All Services
color 0C

echo.
echo  Stopping FaceGuard.AI services...
echo  ===================================
echo.

:: Kill background server windows
echo [1/3] Stopping Django backend...
taskkill /fi "WINDOWTITLE eq FaceGuard - Django Backend*" /f >nul 2>&1
for /f "tokens=5" %%p in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING 2^>nul') do (
    taskkill /pid %%p /f >nul 2>&1
)
echo   [OK] Backend stopped.

echo [2/3] Stopping Vite frontend...
taskkill /fi "WINDOWTITLE eq FaceGuard - Vite Frontend*" /f >nul 2>&1
for /f "tokens=5" %%p in ('netstat -ano ^| findstr :5173 ^| findstr LISTENING 2^>nul') do (
    taskkill /pid %%p /f >nul 2>&1
)
echo   [OK] Frontend stopped.

echo [3/3] Stopping Docker containers...
cd /d "%~dp0"
docker compose stop db mongo redis >nul 2>&1
echo   [OK] Database containers stopped.

echo.
echo  All services stopped successfully.
echo.
ping -n 4 127.0.0.1 >nul

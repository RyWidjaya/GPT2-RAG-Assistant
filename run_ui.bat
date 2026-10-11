@echo off
chcp 65001 >nul
title Document Assistant - Local GPT-2 Workbench
cd /d "%~dp0"
echo ========================================================
echo   Starting Document Assistant (Local GPT-2 Workbench)...
echo ========================================================
python run_ui.py
if errorlevel 1 (
    echo.
    echo [ERROR] An error occurred while launching the application.
    echo Please ensure dependencies (fastapi, uvicorn, torch) are installed.
    echo.
)
pause

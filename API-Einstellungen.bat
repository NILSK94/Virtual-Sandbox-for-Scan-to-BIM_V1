@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0API-Einstellungen.ps1"
if errorlevel 1 pause

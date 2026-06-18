@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
  echo [エラー] セットアップが未実行です。先に setup.bat を実行してください。
  pause
  exit /b 1
)

echo 音声文字起こしツールを起動します...
echo ブラウザで http://127.0.0.1:5000 を開いてください。
echo （終了するにはこのウィンドウで Ctrl + C を押してください）
echo.

REM 既定ブラウザで自動的に開く
start "" http://127.0.0.1:5000

call venv\Scripts\python.exe app.py
pause

@echo off
chcp 932 >nul
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" goto err_setup

echo 音声文字起こしツールを起動します...
echo ブラウザで http://127.0.0.1:5000 を開いてください。
echo 終了するにはこのウィンドウで Ctrl + C を押してください。
echo.

rem Open the default browser automatically.
start "" http://127.0.0.1:5000

call venv\Scripts\python.exe app.py
pause
exit /b 0

:err_setup
echo [エラー] セットアップが未実行です。先に setup.bat を実行してください。
pause
exit /b 1

@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo  音声文字起こしツール  初回セットアップ
echo ============================================
echo.

rem Decide what to do with venv:
rem   - no venv               -> create
rem   - venv but pip unusable -> broken, recreate
rem   - venv and pip OK        -> reuse (avoid re-downloading PyTorch etc.)
if not exist "venv\Scripts\python.exe" goto create
call venv\Scripts\python.exe -m pip --version >nul 2>&1
if errorlevel 1 goto recreate
echo [1/4] 仮想環境は正常なため、再利用します。
goto pipupgrade

:recreate
echo [1/4] 仮想環境が不完全なため、削除してクリーンに作り直します...
rmdir /s /q "venv"
goto makevenv

:create
echo [1/4] 仮想環境を作成します...

:makevenv
python -m venv venv
if errorlevel 1 goto err_venv
rem Make sure pip exists inside the venv.
call venv\Scripts\python.exe -m ensurepip --upgrade

:pipupgrade
echo [2/4] pip を更新します...
call venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto err_pip

echo [3/4] PyTorch CPU版 をインストールします...
call venv\Scripts\python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto err_torch

echo [4/4] 必要なライブラリをインストールします...
call venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto err_lib

echo.
echo セットアップが完了しました。 start.bat を実行してツールを起動してください。
pause
exit /b 0

:err_venv
echo [エラー] 仮想環境の作成に失敗しました。Python 3.9 以上がインストールされているか確認してください。
pause
exit /b 1

:err_pip
echo [エラー] pip の更新に失敗しました。
pause
exit /b 1

:err_torch
echo [エラー] PyTorch のインストールに失敗しました。
pause
exit /b 1

:err_lib
echo [エラー] ライブラリのインストールに失敗しました。
pause
exit /b 1

@echo off
chcp 932 >nul
cd /d "%~dp0"
echo ============================================
echo  音声文字起こしツール  初回セットアップ
echo ============================================
echo.

if exist "venv\Scripts\python.exe" (
  echo [1/4] 仮想環境(venv) は既に存在するため、作成をスキップします。
) else (
  echo [1/4] 仮想環境(venv) を作成します...
  python -m venv venv
  if errorlevel 1 (
    echo [エラー] 仮想環境の作成に失敗しました。Python 3.9 以上がインストールされているか確認してください。
    pause
    exit /b 1
  )
)

echo [2/4] pip を更新します...
call venv\Scripts\python.exe -m pip install --upgrade pip

echo [3/4] PyTorch(CPU版) をインストールします...
call venv\Scripts\python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
  echo [エラー] PyTorch のインストールに失敗しました。
  pause
  exit /b 1
)

echo [4/4] 必要なライブラリをインストールします...
call venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
  echo [エラー] ライブラリのインストールに失敗しました。
  pause
  exit /b 1
)

echo.
echo セットアップが完了しました。 start.bat を実行してツールを起動してください。
pause
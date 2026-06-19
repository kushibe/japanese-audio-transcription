@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ============================================
echo  音声文字起こしツール  初回セットアップ
echo ============================================
echo.

REM 既存の venv の状態を確認する。
REM ・venv が無い → 新規作成
REM ・venv はあるが pip が無い（不完全な状態）→「No module named pip」対策として作り直す
REM ・venv があり pip も使える → そのまま再利用（PyTorch 等の再ダウンロードを避ける）
set NEED_CREATE=0
if not exist "venv\Scripts\python.exe" (
  set NEED_CREATE=1
) else (
  call venv\Scripts\python.exe -m pip --version >nul 2>&1
  if errorlevel 1 set NEED_CREATE=1
)

if "%NEED_CREATE%"=="1" (
  if exist "venv" (
    echo [1/4] 仮想環境(venv) が不完全なため、削除してクリーンに作り直します...
    rmdir /s /q "venv"
  ) else (
    echo [1/4] 仮想環境(venv) を作成します...
  )
  python -m venv venv
  if errorlevel 1 (
    echo [エラー] 仮想環境の作成に失敗しました。Python 3.9 以上がインストールされているか確認してください。
    pause
    exit /b 1
  )
  REM venv 内に pip が無い場合に備えて ensurepip で確実に用意する。
  call venv\Scripts\python.exe -m ensurepip --upgrade
) else (
  echo [1/4] 仮想環境(venv) は正常なため、再利用します。
)

echo [2/4] pip を更新します...
call venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 (
  echo [エラー] pip の更新に失敗しました。
  pause
  exit /b 1
)

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

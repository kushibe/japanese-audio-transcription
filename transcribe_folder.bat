@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" (
  echo [エラー] セットアップが未実行です。先に setup.bat を実行してください。
  pause
  exit /b 1
)

echo ============================================
echo  バッチ文字起こし（CLIモード／話者分離あり）
echo ============================================
echo  INPUT フォルダ内の .wav をまとめて文字起こしし、
echo  結果を OUTPUT フォルダに書き出します。
echo  既定で話者分離を行います（話者の人数は自動推定）。
echo  字幕(.srt)は出力しません。
echo.

rem cli.py の日本語出力が文字化けしないよう UTF-8 モードで実行する。
rem 既定で話者分離(--diarize)を行い、字幕(.srt)は出力しない(--no-srt)。
rem 追加の引数（例: --num-speakers 2 や --model medium）はそのまま cli.py へ渡される。
set PYTHONUTF8=1
call venv\Scripts\python.exe cli.py --diarize --no-srt %*

echo.
pause

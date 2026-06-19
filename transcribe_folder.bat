@echo off
chcp 932 >nul
cd /d "%~dp0"

if not exist "venv\Scripts\python.exe" goto err_setup

echo ============================================
echo  バッチ文字起こし（CLIモード／話者分離あり）
echo ============================================
echo  INPUT フォルダ内の .wav をまとめて文字起こしし、
echo  結果を OUTPUT フォルダに書き出します。
echo  既定で話者分離を行います。話者の人数は自動推定します。
echo  字幕ファイル .srt は出力しません。
echo.

rem Run cli.py in UTF-8 mode so Japanese output is not garbled.
rem Default: diarization on (--diarize), no subtitle file (--no-srt).
rem Extra args (e.g. --num-speakers 2 / --model medium) are passed through.
set PYTHONUTF8=1
call venv\Scripts\python.exe cli.py --diarize --no-srt %*

echo.
pause
exit /b 0

:err_setup
echo [エラー] セットアップが未実行です。先に setup.bat を実行してください。
pause
exit /b 1

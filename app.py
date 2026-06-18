# -*- coding: utf-8 -*-
"""
ローカル日本語音声文字起こしツール - バックエンド

アーキテクチャ:
  - faster-whisper (Whisper) によるローカル文字起こし。
    日本語を高精度で認識。音声データは外部に送信されず、すべてこのPC内で処理される。
  - SpeechBrain (ECAPA-TDNN) による話者分離（任意）。
  - Flask による軽量Webサーバ。文字起こしはバックグラウンドのジョブとして実行し、
    フロントエンド(HTML/JS)は進捗(残り時間)をポーリングで取得する。
"""
import os
import time
import uuid
import tempfile
import threading
import traceback
from collections import deque

from flask import Flask, request, jsonify, send_from_directory

# 文字起こしの中核処理は CLI と共通化するため transcribe_core に集約している。
# （HF のシンボリックリンク無効化など faster_whisper 読み込み前の設定も同モジュールが行う）
import transcribe_core as core
from transcribe_core import (
    BASE_DIR, DEFAULT_MODEL, ALLOWED_EXT,
    load_model_with_fallback, is_gpu_error, get_model,
    transcribe_collect, diarize_into, build_result,
)

# LLM(Bedrock)による問い合わせ分析。設定の読み書きと分析実行を提供する。
import llm_analyze

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 最大1GB


# ---------------------------------------------------------------------------
# ジョブ管理（バックグラウンド処理 + 進捗ポーリング）
# ---------------------------------------------------------------------------
_jobs = {}
_jobs_lock = threading.Lock()


def _new_job():
    job_id = uuid.uuid4().hex
    job = {
        "id": job_id,
        "status": "queued",          # queued/loading_model/transcribing/diarizing/done/error
        "phase_label": "待機中",
        "progress": 0.0,             # 現フェーズの進捗 0..1
        "processed_sec": 0.0,
        "duration": 0.0,
        "elapsed": 0.0,
        "eta_sec": None,             # 推定残り秒
        "partial_text": "",          # 文字起こし途中経過（リアルタイム表示用）
        "partial_count": 0,
        "error": None,
        "result": None,
        "created": time.time(),
    }
    with _jobs_lock:
        _jobs[job_id] = job
    return job


def _transcribe_with_progress(model, path, job):
    """モデルで文字起こしを実行し、ジョブの進捗(残り時間)を更新しながら
    (info, セグメント一覧, 全文) を返す。

    実際の認識処理は core.transcribe_collect が行い、ここでは1セグメントごとの
    コールバックで Web UI 向けの進捗(残り時間/途中経過)を計算・更新する。"""
    state = {"start": time.time(), "hist": deque(), "n": 0, "parts": []}

    # 残り時間は「直近の処理速度」で推定する（時間窓ベースの移動平均）。
    # 基準点を最初のセグメント以降に置くことで、冒頭の無音スキップによる
    # 速度の過大評価（ETA が序盤で増加する現象）を避ける。
    WINDOW = 20.0      # 直近何秒分の実時間で速度を測るか

    def _on_segment(seg, processed, duration):
        if not job["duration"]:
            job["duration"] = round(duration, 2)
        state["n"] += 1
        state["parts"].append(seg["text"])

        now = time.time()
        elapsed = now - state["start"]
        job["processed_sec"] = round(processed, 1)
        job["elapsed"] = round(elapsed, 1)
        job["partial_text"] = "\n".join(state["parts"])
        job["partial_count"] = state["n"]

        # 時間窓内の (処理済み音声秒 / 実時間) を直近速度とする。
        # 基準点(hist[0])は最初のセグメントなので、最初の1区間だけは
        # 速度を確定できず eta は「推定中」のままにする。
        hist = state["hist"]
        hist.append((now, processed))
        while len(hist) > 2 and now - hist[0][0] > WINDOW:
            hist.popleft()
        if duration > 0 and processed > 0:
            job["progress"] = min(processed / duration, 0.999)
            if len(hist) >= 2:
                t_span = now - hist[0][0]
                p_span = processed - hist[0][1]
                # 測定窓が短すぎる序盤（ウォームアップ直後の高速バースト）は
                # 速度が不正確なので ETA を出さない（「推定中」のまま）
                if t_span >= 3.0 and p_span > 0:
                    rate = p_span / t_span
                    # 異常値ガード: 終盤に窓内の前進量が小さくなると速度が
                    # ほぼ0になり ETA が暴騰するため、累積平均速度の50%を下限にする。
                    # （結果として ETA は累積推定の2倍以内に収まる）
                    if elapsed > 0:
                        rate = max(rate, 0.5 * (processed / elapsed))
                    job["eta_sec"] = max(0, int(round((duration - processed) / rate)))

    return transcribe_collect(model, path, on_segment=_on_segment)


def _run_job(job, path, filename, model_size, diarize, num_speakers):
    try:
        # --- 1. モデル読み込み ---
        job["status"] = "loading_model"
        job["phase_label"] = "モデル読み込み中"
        model = load_model_with_fallback(model_size)

        # --- 2. 文字起こし（残り時間つき） ---
        job["status"] = "transcribing"
        job["phase_label"] = "文字起こし中"
        try:
            info, seg_list, full_parts = _transcribe_with_progress(model, path, job)
        except RuntimeError as e:
            if is_gpu_error(e):
                app.logger.warning("GPU実行失敗 -> CPUで再試行: %s", e)
                model = get_model(model_size, force_cpu=True)
                job["progress"] = 0.0
                info, seg_list, full_parts = _transcribe_with_progress(model, path, job)
            else:
                raise

        # --- 3. 話者分離（任意） ---
        num_detected = 0
        if diarize and seg_list:
            job["status"] = "diarizing"
            job["phase_label"] = "話者分離中"
            job["progress"] = 0.0
            job["eta_sec"] = None

            d_start = time.time()

            def _cb(done, total):
                if total:
                    job["progress"] = min(done / total, 0.999)
                    el = time.time() - d_start
                    if done > 0:
                        speed = done / el
                        job["eta_sec"] = max(0, int(round((total - done) / speed))) if speed > 0 else None

            num_detected = diarize_into(path, seg_list, num_speakers=num_speakers, progress_cb=_cb)

        # --- 4. 出力生成 ---
        job["result"] = build_result(
            filename, info, seg_list, full_parts, model_size,
            diarized=diarize, num_detected=num_detected,
        )
        job["progress"] = 1.0
        job["eta_sec"] = 0
        job["status"] = "done"
        job["phase_label"] = "完了"

    except Exception as e:  # noqa: BLE001
        app.logger.error("ジョブ失敗: %s\n%s", e, traceback.format_exc())
        job["status"] = "error"
        job["error"] = f"{e}"
    finally:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# ルーティング
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "ファイルがアップロードされていません。"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "ファイル名が空です。"}), 400
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"対応していない形式です（{ext}）。.wav を指定してください。"}), 400

    model_size = request.form.get("model", DEFAULT_MODEL)
    diarize = request.form.get("diarize", "false").lower() in ("1", "true", "on", "yes")
    num_speakers_raw = request.form.get("num_speakers", "").strip()
    num_speakers = None
    if num_speakers_raw and num_speakers_raw.isdigit() and int(num_speakers_raw) >= 1:
        num_speakers = int(num_speakers_raw)

    # 一時ファイルに保存
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    f.save(tmp_path)

    job = _new_job()
    t = threading.Thread(
        target=_run_job,
        args=(job, tmp_path, f.filename, model_size, diarize, num_speakers),
        daemon=True,
    )
    t.start()
    return jsonify({"job_id": job["id"]})


@app.route("/progress/<job_id>")
def progress(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "ジョブが見つかりません。"}), 404

    payload = {
        "status": job["status"],
        "phase_label": job["phase_label"],
        "progress": round(job["progress"], 4),
        "processed_sec": job["processed_sec"],
        "duration": job["duration"],
        "elapsed": job["elapsed"],
        "eta_sec": job["eta_sec"],
        "partial_text": job["partial_text"],
        "partial_count": job["partial_count"],
        "error": job["error"],
    }
    if job["status"] == "done":
        payload["result"] = job["result"]
        # 結果を返したらメモリから解放
        with _jobs_lock:
            _jobs.pop(job_id, None)
    return jsonify(payload)


# ---------------------------------------------------------------------------
# LLM(問い合わせ分析) 関連
# ---------------------------------------------------------------------------
def _mask_secret(value: str) -> str:
    """秘密情報を画面表示用にマスクする（末尾4文字のみ残す）。"""
    if not value:
        return ""
    if len(value) <= 4:
        return "*" * len(value)
    return "*" * (len(value) - 4) + value[-4:]


@app.route("/config", methods=["GET"])
def get_config():
    """LLM(Bedrock)設定とカテゴリ一覧を返す。認証情報はマスクして返す。"""
    cfg = llm_analyze.load_bedrock_config()
    return jsonify({
        "bedrock": {
            # 秘密情報はマスク表示。設定済みかどうかも併せて返す。
            "aws_access_key": _mask_secret(cfg["aws_access_key"]),
            "aws_secret_key": _mask_secret(cfg["aws_secret_key"]),
            "aws_session_token": _mask_secret(cfg["aws_session_token"]),
            "aws_region": cfg["aws_region"],
            "model_id": cfg["model_id"],
            "max_tokens": cfg["max_tokens"],
            "has_access_key": bool(cfg["aws_access_key"]),
            "has_secret_key": bool(cfg["aws_secret_key"]),
            "has_session_token": bool(cfg["aws_session_token"]),
        },
        "categories": llm_analyze.load_categories(),
    })


@app.route("/config", methods=["POST"])
def update_config():
    """LLM(Bedrock)設定とカテゴリ一覧を保存する。

    認証情報フィールドは「空文字なら変更なし（既存値を維持）」とし、
    マスク済みの値がそのまま送り返されても上書きしないようにする。
    """
    data = request.get_json(silent=True) or {}
    bedrock_in = data.get("bedrock") or {}
    current = llm_analyze.load_bedrock_config()

    update = {}
    # 秘密情報: 空 or マスク文字列(末尾4文字以外が*)の場合は既存値を維持
    for key in ("aws_access_key", "aws_secret_key", "aws_session_token"):
        val = bedrock_in.get(key)
        if val is None:
            continue
        val = str(val)
        if val == "" or set(val[:-4]) == {"*"} and "*" in val:
            continue  # 未変更とみなす
        update[key] = val
    # 非秘密情報はそのまま反映
    for key in ("aws_region", "model_id", "max_tokens"):
        if key in bedrock_in and bedrock_in[key] is not None:
            update[key] = bedrock_in[key]

    saved = llm_analyze.save_bedrock_config({**current, **update})
    if "categories" in data:
        saved_categories = llm_analyze.save_categories(data["categories"])
    else:
        saved_categories = llm_analyze.load_categories()

    return jsonify({
        "ok": True,
        "bedrock": {
            "aws_access_key": _mask_secret(saved["aws_access_key"]),
            "aws_secret_key": _mask_secret(saved["aws_secret_key"]),
            "aws_session_token": _mask_secret(saved["aws_session_token"]),
            "aws_region": saved["aws_region"],
            "model_id": saved["model_id"],
            "max_tokens": saved["max_tokens"],
            "has_access_key": bool(saved["aws_access_key"]),
            "has_secret_key": bool(saved["aws_secret_key"]),
            "has_session_token": bool(saved["aws_session_token"]),
        },
        "categories": saved_categories,
    })


@app.route("/prompt", methods=["POST"])
def build_prompt_endpoint():
    """文字起こしを差し込んだLLM投入用プロンプト全文を返す。

    利用者が一般的なLLM（ChatGPT等）に自分で貼り付けて使えるように、
    設定済みのカテゴリ一覧を反映したプロンプト文をそのまま組み立てて返す。
    AWS認証情報は不要。
    """
    data = request.get_json(silent=True) or {}
    transcript = (data.get("transcript") or "").strip()
    if not transcript:
        return jsonify({"error": "文字起こしデータが空です。"}), 400
    prompt = llm_analyze.build_prompt(transcript)
    return jsonify({"ok": True, "prompt": prompt})


@app.route("/analyze", methods=["POST"])
def analyze():
    """話者分離テキスト等の文字起こしをLLMで整形し、JSONを返す。"""
    data = request.get_json(silent=True) or {}
    transcript = (data.get("transcript") or "").strip()
    if not transcript:
        return jsonify({"error": "文字起こしデータが空です。"}), 400
    try:
        result = llm_analyze.analyze_transcript(transcript)
        return jsonify({"ok": True, "result": result})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:  # noqa: BLE001
        app.logger.error("LLM分析に失敗: %s\n%s", e, traceback.format_exc())
        return jsonify({"error": f"LLM分析に失敗しました: {e}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, threaded=True)

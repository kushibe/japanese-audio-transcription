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

# HuggingFace のキャッシュがシンボリックリンクを作ろうとすると、Windows では
# 管理者権限/開発者モードが無い場合に [WinError 1314] で失敗する。
# シンボリックリンクを無効化（コピー方式）にして権限不要で動作させる。
# ※ faster_whisper(=huggingface_hub) のインポート前に設定する必要がある。
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from flask import Flask, request, jsonify, send_from_directory

from faster_whisper import WhisperModel

# ---------------------------------------------------------------------------
# 設定
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")

# 実行デバイス: 既定は "cpu"。どの環境でも確実に動作する。
#   NVIDIA GPU と CUDA/cuBLAS が正しく入っている環境では
#   WHISPER_DEVICE=cuda を指定すると高速化できる。
DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")
COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")

ALLOWED_EXT = {".wav"}

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 最大1GB

# ---------------------------------------------------------------------------
# モデルの遅延ロード（初回利用時に読み込み、以降は再利用）
# ---------------------------------------------------------------------------
_model_cache = {}
_model_lock = threading.Lock()


def get_model(model_size: str, force_cpu: bool = False) -> WhisperModel:
    """指定モデルを読み込んで返す（同一設定はキャッシュして再利用）。"""
    if force_cpu:
        device, compute = "cpu", "int8"
    else:
        device = DEVICE
        compute = COMPUTE_TYPE
        if compute == "auto" and device == "cpu":
            compute = "int8"

    key = (model_size, device, compute)
    with _model_lock:
        if key not in _model_cache:
            app.logger.info("Whisperモデルを読み込み中: %s (device=%s, compute=%s)",
                            model_size, device, compute)
            _model_cache[key] = WhisperModel(model_size, device=device, compute_type=compute)
            app.logger.info("モデル読み込み完了: %s", model_size)
        return _model_cache[key]


def _is_gpu_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ("cublas", "cuda", "cudnn", "gpu", "libcublas"))


def _format_timestamp(seconds: float, sep: str = ",") -> str:
    """秒数を HH:MM:SS,mmm 形式に変換。"""
    if seconds is None:
        seconds = 0.0
    ms = int(round(seconds * 1000.0))
    hours, ms = divmod(ms, 3600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{ms:03d}"


def _hhmmss(seconds: float) -> str:
    s = int(round(seconds or 0))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


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
    (info, セグメント一覧, 全文) を返す。"""
    segments, info = model.transcribe(
        path,
        language="ja",
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    duration = info.duration or 0.0
    job["duration"] = round(duration, 2)

    seg_list = []
    full_text_parts = []
    start = time.time()

    # 残り時間は「直近の処理速度」で推定する（時間窓ベースの移動平均）。
    # 基準点を最初のセグメント以降に置くことで、冒頭の無音スキップによる
    # 速度の過大評価（ETA が序盤で増加する現象）を避ける。
    WINDOW = 20.0      # 直近何秒分の実時間で速度を測るか
    hist = deque()     # (実時刻, 処理済み音声秒)
    for seg in segments:  # 遅延評価。ここで実際の計算が進む。
        text = seg.text.strip()
        seg_list.append({
            "id": seg.id,
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": text,
        })
        full_text_parts.append(text)

        now = time.time()
        processed = seg.end or 0.0
        elapsed = now - start
        job["processed_sec"] = round(processed, 1)
        job["elapsed"] = round(elapsed, 1)
        job["partial_text"] = "\n".join(full_text_parts)
        job["partial_count"] = len(seg_list)

        # 時間窓内の (処理済み音声秒 / 実時間) を直近速度とする。
        # 基準点(hist[0])は最初のセグメントなので、最初の1区間だけは
        # 速度を確定できず eta は「推定中」のままにする。
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
    return info, seg_list, full_text_parts


def _build_speaker_text(seg_list):
    """話者ラベル付きのタイムスタンプテキストを生成。
    連続する同一話者のセグメントは1段落にまとめる。"""
    lines = []
    cur = None
    buf = []
    start = 0.0
    for seg in seg_list:
        sp = seg.get("speaker", 1)
        if sp != cur:
            if buf:
                lines.append(f"[{_hhmmss(start)}] 話者{cur}: {''.join(buf)}")
            cur = sp
            buf = [seg["text"]]
            start = seg["start"]
        else:
            buf.append(seg["text"])
    if buf:
        lines.append(f"[{_hhmmss(start)}] 話者{cur}: {''.join(buf)}")
    return "\n".join(lines)


def _run_job(job, path, filename, model_size, diarize, num_speakers):
    try:
        # --- 1. モデル読み込み ---
        job["status"] = "loading_model"
        job["phase_label"] = "モデル読み込み中"
        try:
            model = get_model(model_size)
        except RuntimeError as e:
            if _is_gpu_error(e):
                app.logger.warning("GPU読み込み失敗 -> CPUにフォールバック: %s", e)
                model = get_model(model_size, force_cpu=True)
            else:
                raise

        # --- 2. 文字起こし（残り時間つき） ---
        job["status"] = "transcribing"
        job["phase_label"] = "文字起こし中"
        try:
            info, seg_list, full_parts = _transcribe_with_progress(model, path, job)
        except RuntimeError as e:
            if _is_gpu_error(e):
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
            from diarization import diarize_segments

            d_start = time.time()

            def _cb(done, total):
                if total:
                    job["progress"] = min(done / total, 0.999)
                    el = time.time() - d_start
                    if done > 0:
                        speed = done / el
                        job["eta_sec"] = max(0, int(round((total - done) / speed))) if speed > 0 else None

            labels = diarize_segments(path, seg_list, num_speakers=num_speakers, progress_cb=_cb)
            for seg, lab in zip(seg_list, labels):
                seg["speaker"] = lab
            num_detected = len(set(labels)) if labels else 0

        # --- 4. 出力生成 ---
        srt_lines = []
        for i, seg in enumerate(seg_list, start=1):
            spk = f"[話者{seg['speaker']}] " if "speaker" in seg else ""
            srt_lines.append(str(i))
            srt_lines.append(f"{_format_timestamp(seg['start'])} --> {_format_timestamp(seg['end'])}")
            srt_lines.append(spk + seg["text"])
            srt_lines.append("")
        srt_text = "\n".join(srt_lines)

        speaker_text = _build_speaker_text(seg_list) if diarize else ""

        job["result"] = {
            "filename": filename,
            "language": info.language,
            "language_probability": round(info.language_probability, 4),
            "duration": round(info.duration or 0.0, 2),
            "model": model_size,
            "diarized": bool(diarize),
            "num_speakers": num_detected,
            "text": "\n".join(full_parts),
            "segments": seg_list,
            "srt": srt_text,
            "speaker_text": speaker_text,
        }
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, threaded=True)

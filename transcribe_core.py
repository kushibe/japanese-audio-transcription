# -*- coding: utf-8 -*-
"""
文字起こしの共通コアモジュール

Web サーバ(app.py) と CLI(cli.py) の双方から利用する、文字起こしの中核処理を
ここに集約する。具体的には次を提供する。

  - Whisper モデルの読み込み（キャッシュ・GPU/CPU フォールバック）
  - セグメント単位の文字起こし（進捗コールバック対応）
  - 話者分離の呼び出し
  - 出力（SRT 字幕・話者別テキスト）の生成

音声データは外部に送信されず、すべてこの PC 内で処理される点は従来どおり。
"""
import os
import logging
import threading

# HuggingFace のキャッシュがシンボリックリンクを作ろうとすると、Windows では
# 管理者権限/開発者モードが無い場合に [WinError 1314] で失敗する。
# シンボリックリンクを無効化（コピー方式）にして権限不要で動作させる。
# ※ faster_whisper(=huggingface_hub) のインポート前に設定する必要がある。
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

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

logger = logging.getLogger("transcribe")

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
            logger.info("Whisperモデルを読み込み中: %s (device=%s, compute=%s)",
                        model_size, device, compute)
            _model_cache[key] = WhisperModel(model_size, device=device, compute_type=compute)
            logger.info("モデル読み込み完了: %s", model_size)
        return _model_cache[key]


def is_gpu_error(exc: Exception) -> bool:
    """例外メッセージが GPU(CUDA/cuBLAS) 由来かどうかを判定する。"""
    msg = str(exc).lower()
    return any(k in msg for k in ("cublas", "cuda", "cudnn", "gpu", "libcublas"))


def load_model_with_fallback(model_size: str) -> WhisperModel:
    """モデルを読み込む。GPU で失敗した場合は CPU にフォールバックする。"""
    try:
        return get_model(model_size)
    except RuntimeError as e:
        if is_gpu_error(e):
            logger.warning("GPU読み込み失敗 -> CPUにフォールバック: %s", e)
            return get_model(model_size, force_cpu=True)
        raise


# ---------------------------------------------------------------------------
# 出力フォーマット用ヘルパ
# ---------------------------------------------------------------------------
def format_timestamp(seconds: float, sep: str = ",") -> str:
    """秒数を HH:MM:SS,mmm 形式に変換。"""
    if seconds is None:
        seconds = 0.0
    ms = int(round(seconds * 1000.0))
    hours, ms = divmod(ms, 3600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{ms:03d}"


def hhmmss(seconds: float) -> str:
    s = int(round(seconds or 0))
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def build_speaker_text(seg_list):
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
                lines.append(f"[{hhmmss(start)}] 話者{cur}: {''.join(buf)}")
            cur = sp
            buf = [seg["text"]]
            start = seg["start"]
        else:
            buf.append(seg["text"])
    if buf:
        lines.append(f"[{hhmmss(start)}] 話者{cur}: {''.join(buf)}")
    return "\n".join(lines)


def build_srt(seg_list):
    """セグメント一覧から SRT 字幕テキストを生成。話者があればラベルを付与。"""
    lines = []
    for i, seg in enumerate(seg_list, start=1):
        spk = f"[話者{seg['speaker']}] " if "speaker" in seg else ""
        lines.append(str(i))
        lines.append(f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}")
        lines.append(spk + seg["text"])
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 文字起こし本体
# ---------------------------------------------------------------------------
def transcribe_collect(model, path, on_segment=None):
    """モデルで文字起こしを実行し、(info, セグメント一覧, 全文パーツ) を返す。

    faster-whisper はセグメントを遅延評価で返すため、ここを反復することで
    実際の認識処理が進む。on_segment(seg_dict, processed_sec, duration) が
    指定されていれば、1セグメントごとに呼び出して進捗を通知する。
    """
    segments, info = model.transcribe(
        path,
        language="ja",
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    duration = info.duration or 0.0

    seg_list = []
    full_text_parts = []
    for seg in segments:  # 遅延評価。ここで実際の計算が進む。
        text = seg.text.strip()
        d = {
            "id": seg.id,
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": text,
        }
        seg_list.append(d)
        full_text_parts.append(text)
        if on_segment:
            on_segment(d, seg.end or 0.0, duration)
    return info, seg_list, full_text_parts


def diarize_into(path, seg_list, num_speakers=None, progress_cb=None):
    """seg_list の各要素に話者番号("speaker")を書き込み、検出話者数を返す。"""
    from diarization import diarize_segments

    labels = diarize_segments(path, seg_list, num_speakers=num_speakers,
                              progress_cb=progress_cb)
    for seg, lab in zip(seg_list, labels):
        seg["speaker"] = lab
    return len(set(labels)) if labels else 0


def build_result(filename, info, seg_list, full_parts, model_size,
                 diarized, num_detected):
    """文字起こし結果を1つの dict にまとめる（Web/CLI 共通の成果物）。"""
    return {
        "filename": filename,
        "language": info.language,
        "language_probability": round(info.language_probability, 4),
        "duration": round(info.duration or 0.0, 2),
        "model": model_size,
        "diarized": bool(diarized),
        "num_speakers": num_detected,
        "text": "\n".join(full_parts),
        "segments": seg_list,
        "srt": build_srt(seg_list),
        "speaker_text": build_speaker_text(seg_list) if diarized else "",
    }

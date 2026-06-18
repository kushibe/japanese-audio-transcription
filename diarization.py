# -*- coding: utf-8 -*-
"""
話者分離（ダイアライゼーション）モジュール

方式:
  1. faster-whisper の音声デコーダで wav を 16kHz モノラルに変換（タイムライン保持）
  2. 各セグメント区間の音声から、SpeechBrain の ECAPA-TDNN モデルで
     話者埋め込みベクトルを計算
  3. 「信頼できる長いセグメント(既定 1.5 秒以上)」だけでクラスタリングして
     話者の基準(セントロイド)を作り、相槌などの短いセグメントは
     最も近い話者に割り当てる
     → 短い区間を無理に独立クラスタにしてしまう過分割を防ぐ

すべてローカルで動作し、外部送信や HuggingFace トークンは不要。
（ECAPA モデルは初回のみ公開リポジトリから自動ダウンロードされる）
"""
import os
import numpy as np

SR = 16000                       # 埋め込み計算用サンプリングレート
_RELIABLE_MIN = int(1.5 * SR)    # クラスタリングの基準にする最小長（約1.5秒）
_EMBED_MIN = int(0.4 * SR)       # 埋め込み計算に確保する最小長（約0.4秒）
_ABS_MIN = int(0.2 * SR)         # これ未満は埋め込み不可

# 自動推定時のコサイン距離しきい値（環境変数で調整可能）
_AUTO_THRESHOLD = float(os.environ.get("DIARIZE_THRESHOLD", "0.68"))

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.path.join(_BASE_DIR, "models", "ecapa")

_classifier = None


def _get_classifier():
    """SpeechBrain の ECAPA 話者埋め込みモデルを遅延ロード（CPU）。"""
    global _classifier
    if _classifier is None:
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy
        # Windows でシンボリックリンク作成権限が無くても動くよう COPY 戦略を使う
        _classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=_MODEL_DIR,
            run_opts={"device": "cpu"},
            local_strategy=LocalStrategy.COPY,
        )
    return _classifier


def _load_audio(path: str) -> np.ndarray:
    """16kHz モノラルの float32 波形を返す（タイムラインを保持）。"""
    from faster_whisper.audio import decode_audio
    return decode_audio(path, sampling_rate=SR)


def _embed(classifier, chunk: np.ndarray) -> np.ndarray:
    """1区間の波形から話者埋め込みベクトルを計算。"""
    import torch
    with torch.no_grad():
        t = torch.from_numpy(np.ascontiguousarray(chunk)).float().unsqueeze(0)
        emb = classifier.encode_batch(t)  # [1, 1, 192]
    return emb.reshape(-1).cpu().numpy()


def _l2norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _cos_dist(a, b):
    return 1.0 - float(np.dot(_l2norm(a), _l2norm(b)))


def _relabel_by_appearance(labels):
    """登場順に 話者1,2,... と番号を振り直す。"""
    mapping = {}
    out = []
    nxt = 1
    for l in labels:
        if l not in mapping:
            mapping[l] = nxt
            nxt += 1
        out.append(mapping[l])
    return out


def _cluster(mat: np.ndarray, num_speakers):
    """埋め込み行列をクラスタリングしてラベル配列を返す。

    num_speakers が指定されればその人数に、None なら距離しきい値で自動推定。
    """
    from sklearn.cluster import AgglomerativeClustering

    n = mat.shape[0]
    if n == 1:
        return [0]

    if num_speakers and int(num_speakers) >= 1:
        k = min(int(num_speakers), n)
        if k == 1:
            return [0] * n
        model = AgglomerativeClustering(
            n_clusters=k, metric="cosine", linkage="average"
        )
    else:
        model = AgglomerativeClustering(
            n_clusters=None, distance_threshold=_AUTO_THRESHOLD,
            metric="cosine", linkage="average",
        )
    return model.fit_predict(mat).tolist()


def diarize_segments(audio_path, segments, num_speakers=None, progress_cb=None):
    """各セグメントに話者番号(1始まり)を割り当てて返す。

    Args:
        audio_path:   解析対象の音声ファイル
        segments:     [{'start','end','text', ...}] の一覧
        num_speakers: 人数指定（None で自動推定）
        progress_cb:  progress_cb(done, total) で進捗を通知（任意）

    Returns:
        labels: segments と同順の話者番号リスト（1始まり）
    """
    total = len(segments)
    if total == 0:
        return []

    wav = _load_audio(audio_path)
    total_len = len(wav)
    classifier = _get_classifier()

    embeds = []       # 各セグメントの埋め込み（取得不能なら None）
    reliable = []     # クラスタリングの基準に使えるか（十分な長さ）
    for i, seg in enumerate(segments):
        s = int(seg["start"] * SR)
        e = int(seg["end"] * SR)
        raw_len = e - s
        # 埋め込み計算用に最小長だけは確保（隣の話者混入を避けるため広げ過ぎない）
        if raw_len < _EMBED_MIN:
            center = (s + e) // 2
            s = max(0, center - _EMBED_MIN // 2)
            e = min(total_len, s + _EMBED_MIN)
            s = max(0, e - _EMBED_MIN)
        chunk = wav[s:e]

        emb = None
        if len(chunk) >= _ABS_MIN:
            try:
                emb = _embed(classifier, chunk)
            except Exception:
                emb = None
        embeds.append(emb)
        reliable.append(raw_len >= _RELIABLE_MIN and emb is not None)

        if progress_cb:
            progress_cb(i + 1, total)

    valid = [v for v in embeds if v is not None]
    if not valid:
        return [1] * total

    mean_vec = np.mean(np.array(valid), axis=0)
    mat = np.array([v if v is not None else mean_vec for v in embeds])

    rel_idx = [i for i in range(total) if reliable[i]]

    # 信頼できる長いセグメントが2つ未満なら、全体をそのままクラスタリング
    if len(rel_idx) < 2:
        return _relabel_by_appearance(_cluster(mat, num_speakers))

    # 1. 長いセグメントだけで話者をクラスタリング
    rel_mat = mat[rel_idx]
    rel_labels = _cluster(rel_mat, num_speakers)

    # 2. 話者ごとのセントロイド（代表ベクトル）を作る
    uniq = sorted(set(rel_labels))
    centroids = {}
    for u in uniq:
        members = [rel_mat[j] for j, l in enumerate(rel_labels) if l == u]
        centroids[u] = _l2norm(np.mean(np.array(members), axis=0))

    # 3. 全セグメントを割り当て
    #    長いセグメントはクラスタ結果をそのまま、短いセグメントは最近傍の話者へ
    labels = [None] * total
    for k, i in enumerate(rel_idx):
        labels[i] = rel_labels[k]
    for i in range(total):
        if labels[i] is None:
            v = mat[i]
            labels[i] = min(uniq, key=lambda u: _cos_dist(v, centroids[u]))

    return _relabel_by_appearance(labels)

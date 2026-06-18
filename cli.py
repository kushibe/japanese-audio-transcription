# -*- coding: utf-8 -*-
"""
バッチ文字起こし CLI

所定のフォルダ（既定: INPUT）配下にある .wav ファイルを順番に文字起こしし、
結果を出力フォルダ（既定: OUTPUT）に書き出す。GUI を使わずに大量のファイルを
まとめて処理したいときに使う。文字起こしの中核は Web 版(app.py)と共通の
transcribe_core を利用するため、認識結果は GUI と同一。

使い方の例:
    python cli.py                          # INPUT 配下の .wav を OUTPUT へ
    python cli.py --diarize                # 話者分離つき
    python cli.py -i 録音 -o 結果 --model medium
    python cli.py --recursive              # サブフォルダも対象にする
    python cli.py --diarize --analyze      # 話者分離 + LLM(Bedrock)で整形JSON出力

出力（入力ファイル名を基準にする）:
    <name>.txt          … 全文テキスト
    <name>.srt          … 字幕(タイムスタンプ付き)
    <name>.speakers.txt … 話者別テキスト（--diarize 指定時のみ）
    <name>.analysis.json… LLM整形済みの問い合わせ分析JSON（--analyze 指定時のみ）

--analyze を使う場合は事前に config/bedrock_config.json へAWS認証情報・モデルIDを
設定しておくこと（GUI版の「⚙️ 設定」画面からも編集できる）。
"""
import os
import sys
import time
import json
import argparse

import transcribe_core as core

BASE_DIR = core.BASE_DIR


def _find_wav_files(input_dir, recursive):
    """入力フォルダ内の .wav ファイル一覧を返す（ソート済み）。"""
    files = []
    if recursive:
        for root, _dirs, names in os.walk(input_dir):
            for name in names:
                if os.path.splitext(name)[1].lower() in core.ALLOWED_EXT:
                    files.append(os.path.join(root, name))
    else:
        for name in os.listdir(input_dir):
            full = os.path.join(input_dir, name)
            if os.path.isfile(full) and os.path.splitext(name)[1].lower() in core.ALLOWED_EXT:
                files.append(full)
    return sorted(files)


def _progress_printer(label):
    """同一行を上書きしながら進捗を表示するコールバックを作る。

    返したコールバックには最後に見た音声長 `duration` を `_cb.duration` で
    保持させ、完了時に100%へ補正できるようにしている。
    """
    state = {"start": time.time(), "duration": 0.0}

    def _cb(done_or_seg, total_or_processed, duration=None):
        if duration is not None:
            # 文字起こし: on_segment(seg, processed_sec, duration)
            processed = total_or_processed
            state["duration"] = duration
            if duration > 0:
                pct = min(processed / duration, 1.0) * 100
                sys.stdout.write(
                    f"\r    {label}: {pct:5.1f}%  ({processed:6.1f}/{duration:.1f}秒)"
                )
                sys.stdout.flush()
        else:
            # 話者分離: progress_cb(done, total)
            done, total = done_or_seg, total_or_processed
            if total:
                pct = min(done / total, 1.0) * 100
                sys.stdout.write(f"\r    {label}: {pct:5.1f}%  ({done}/{total})")
                sys.stdout.flush()

    _cb.state = state
    return _cb


def _finish_transcribe_line(cb):
    """文字起こし完了時に進捗行を100%へ補正して確定する。

    VADで末尾の無音が除かれると最後のセグメントの終了時刻が総尺より手前に
    なり、processed/duration が100%に届かない。完了は確定しているので、
    総尺を分母・分子の両方に使って100%表示にしてから改行する。
    """
    dur = cb.state["duration"]
    if dur > 0:
        sys.stdout.write(f"\r    文字起こし: 100.0%  ({dur:6.1f}/{dur:.1f}秒)")
        sys.stdout.flush()
    print()  # 進捗行を改行で確定


def _transcribe_one(path, model_size, diarize, num_speakers):
    """1ファイルを文字起こしして結果 dict を返す。"""
    model = core.load_model_with_fallback(model_size)

    # --- 文字起こし（GPU失敗時はCPUで再試行） ---
    cb = _progress_printer("文字起こし")
    try:
        info, seg_list, full_parts = core.transcribe_collect(
            model, path, on_segment=cb)
    except RuntimeError as e:
        if core.is_gpu_error(e):
            print("\n    GPU実行失敗 -> CPUで再試行します。")
            model = core.get_model(model_size, force_cpu=True)
            cb = _progress_printer("文字起こし")
            info, seg_list, full_parts = core.transcribe_collect(
                model, path, on_segment=cb)
        else:
            raise
    _finish_transcribe_line(cb)

    # --- 話者分離（任意） ---
    num_detected = 0
    if diarize and seg_list:
        num_detected = core.diarize_into(
            path, seg_list, num_speakers=num_speakers,
            progress_cb=_progress_printer("話者分離"))
        print()

    return core.build_result(
        os.path.basename(path), info, seg_list, full_parts, model_size,
        diarized=diarize, num_detected=num_detected,
    )


def _write_outputs(result, out_dir, stem, write_srt=True):
    """結果をテキスト/字幕ファイルとして書き出し、書いたパス一覧を返す。"""
    os.makedirs(out_dir, exist_ok=True)
    written = []

    def _save(suffix, content):
        out_path = os.path.join(out_dir, stem + suffix)
        with open(out_path, "w", encoding="utf-8") as fp:
            fp.write(content)
        written.append(out_path)

    _save(".txt", result["text"])
    if write_srt:
        _save(".srt", result["srt"])
    if result.get("diarized") and result.get("speaker_text"):
        _save(".speakers.txt", result["speaker_text"])
    return written


def _analyze_one(result, out_dir, stem):
    """文字起こし結果をLLM(Bedrock)で整形し、<name>.analysis.json を書き出す。

    話者分離テキストがあればそれを、無ければ全文を入力に使う。
    書き出したパスを返す。
    """
    import llm_analyze

    transcript = result.get("speaker_text") or result.get("text") or ""
    data = llm_analyze.analyze_transcript(transcript)
    out_path = os.path.join(out_dir, stem + ".analysis.json")
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)
    return out_path


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="所定フォルダ内の .wav をまとめて文字起こしする CLI モード。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-i", "--input", default="INPUT",
                        help="入力フォルダ（既定: INPUT）")
    parser.add_argument("-o", "--output", default="OUTPUT",
                        help="出力フォルダ（既定: OUTPUT）")
    parser.add_argument("-m", "--model", default=core.DEFAULT_MODEL,
                        help=f"Whisperモデル（既定: {core.DEFAULT_MODEL}）")
    parser.add_argument("--diarize", action="store_true",
                        help="話者分離を行う")
    parser.add_argument("--num-speakers", type=int, default=None,
                        help="話者の人数（指定なしで自動推定）")
    parser.add_argument("-r", "--recursive", action="store_true",
                        help="サブフォルダ内の .wav も対象にする")
    parser.add_argument("--overwrite", action="store_true",
                        help="既に出力が存在する場合も上書きする（既定はスキップ）")
    parser.add_argument("--no-srt", action="store_true",
                        help="字幕(.srt)を出力しない")
    parser.add_argument("--analyze", action="store_true",
                        help="LLM(Bedrock)で問い合わせ分析JSON(.analysis.json)を出力する")
    args = parser.parse_args(argv)

    # --analyze 利用時は設定とライブラリの存在を起動時に確認する
    if args.analyze:
        import llm_analyze
        cfg = llm_analyze.load_bedrock_config()
        if not cfg.get("aws_access_key") or not cfg.get("aws_secret_key"):
            print("[エラー] --analyze にはAWS認証情報が必要です。")
            print(f"        {llm_analyze.BEDROCK_CONFIG_PATH} を設定するか、")
            print("        GUI版(app.py)の「⚙️ 設定」画面から入力してください。")
            return 2

    # 相対パスは実行場所ではなくスクリプト基準で解決し、bat 起動でも迷わないようにする。
    input_dir = args.input if os.path.isabs(args.input) else os.path.join(BASE_DIR, args.input)
    output_dir = args.output if os.path.isabs(args.output) else os.path.join(BASE_DIR, args.output)

    if not os.path.isdir(input_dir):
        # 初回利用の利便のため、空の入力フォルダを作って案内する。
        os.makedirs(input_dir, exist_ok=True)
        print(f"[案内] 入力フォルダを作成しました: {input_dir}")
        print("       ここに .wav ファイルを置いてから、もう一度実行してください。")
        return 0

    wav_files = _find_wav_files(input_dir, args.recursive)
    if not wav_files:
        print(f"[案内] {input_dir} に .wav ファイルが見つかりませんでした。")
        return 0

    print(f"対象: {len(wav_files)} ファイル / モデル: {args.model}"
          + ("（話者分離あり）" if args.diarize else ""))
    print(f"入力: {input_dir}")
    print(f"出力: {output_dir}")
    print("-" * 60)

    ok = 0
    failed = []
    for idx, path in enumerate(wav_files, start=1):
        rel = os.path.relpath(path, input_dir)
        stem = os.path.splitext(os.path.basename(path))[0]
        print(f"[{idx}/{len(wav_files)}] {rel}")

        # スキップ判定（--overwrite 未指定で出力が既にあれば飛ばす）
        # --analyze 指定時は .analysis.json も揃っている場合のみスキップする。
        existing = os.path.join(output_dir, stem + ".txt")
        analysis_exists = os.path.exists(os.path.join(output_dir, stem + ".analysis.json"))
        already_done = os.path.exists(existing) and (not args.analyze or analysis_exists)
        if not args.overwrite and already_done:
            print("    既に出力があるためスキップ（--overwrite で上書き可）")
            ok += 1
            continue

        t0 = time.time()
        try:
            result = _transcribe_one(path, args.model, args.diarize, args.num_speakers)
            written = _write_outputs(result, output_dir, stem, write_srt=not args.no_srt)
            if args.analyze:
                print("    LLMで整形JSONを生成中…")
                written.append(_analyze_one(result, output_dir, stem))
            dt = time.time() - t0
            print(f"    完了 ({dt:.1f}秒) -> " + ", ".join(os.path.basename(w) for w in written))
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"    [失敗] {e}")
            failed.append(rel)

    print("-" * 60)
    print(f"完了: {ok} 件" + (f" / 失敗: {len(failed)} 件" if failed else ""))
    for f in failed:
        print(f"  失敗: {f}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

# -*- coding: utf-8 -*-
"""
LLM(Amazon Bedrock)による問い合わせ分析モジュール

話者分離された文字起こしデータをコンタクトセンターの問い合わせ分析用の
構造化JSONへ整形する。Web版(app.py)とCLI版(cli.py)の双方から利用する
共通モジュールで、次を提供する。

  - 設定ファイル(config/配下)の読み書き
      * bedrock_config.json … AWS認証情報・リージョン・モデルID（GUIから編集）
      * categories.json     … プロンプトの「## 9. category」に埋め込む分類一覧
  - プロンプトの組み立て（カテゴリ一覧と文字起こしを差し込む）
  - Bedrock(Claude)の呼び出しと結果JSONのパース

LLM呼び出しには Anthropic 公式の Bedrock クライアント(AnthropicBedrock)を
利用する。認証情報(AWS Access Key / Secret Key / Session Token / Region)は
設定ファイルから読み込み、boto3 と同じ要領で受け渡す。
"""
import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(BASE_DIR, "config")
BEDROCK_CONFIG_PATH = os.path.join(CONFIG_DIR, "bedrock_config.json")
CATEGORIES_PATH = os.path.join(CONFIG_DIR, "categories.json")

# ---------------------------------------------------------------------------
# 既定値
# ---------------------------------------------------------------------------
# 認証情報は空で初期化し、利用前にGUI画面（または設定ファイル）から設定してもらう。
# model_id / region は環境に合わせて変更が必要なため、編集しやすいよう既定値を置く。
DEFAULT_BEDROCK_CONFIG = {
    "aws_access_key": "",
    "aws_secret_key": "",
    "aws_session_token": "",
    "aws_region": "ap-northeast-1",
    # Bedrock 上のモデルID（推論プロファイルID）。利用するアカウント/リージョンに
    # 合わせて変更すること。例: us.anthropic.claude-sonnet-4-5-20250929-v1:0
    "model_id": "apac.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "max_tokens": 4096,
}

# プロンプトの「## 9. category」に埋め込む既定の分類一覧。
DEFAULT_CATEGORIES = [
    "POS",
    "スマデバ",
    "ネットワーク",
    "認証",
    "ハードウェア",
    "釣銭機",
    "プリンタ",
    "マスタ登録",
    "売上処理",
    "システム作業",
    "その他",
]

# ---------------------------------------------------------------------------
# プロンプトテンプレート
# ---------------------------------------------------------------------------
# 「## 9. category」の分類一覧は設定ファイルから取得して {CATEGORIES} に、
# 文字起こしデータは {TRANSCRIPT} に差し込む。
PROMPT_TEMPLATE = """あなたはコンタクトセンターの問い合わせ分析担当者です。

音声文字起こしデータを読み取り、問い合わせ内容を構造化されたJSON形式に変換してください。

# 目的

後続工程で以下の分析を実施します。

* 問い合わせ件数分析
* 問い合わせカテゴリ分析
* 障害傾向分析
* FAQ候補抽出
* 問い合わせ削減施策の検討
* オペレータ対応内容分析

そのため、単なる要約ではなく、問い合わせの背景・課題・対応内容・結果が分かるレベルで整理してください。

# 抽出ルール

## 1. 問い合わせ単位で分割

1件の通話ログ内に複数の問い合わせが含まれる場合は、問い合わせごとに分割すること。

---

## 2. title

一覧表示用の短いタイトル。

20～40文字程度で作成すること。

例

* 1番レジの無線スキャナが反応しない
* 早朝システム作業後のレジ起動確認

---

## 3. summary

3～5文で要約すること。

以下を必ず含めること。

* 何が起きたか
* なぜ問い合わせたか
* オペレータが何をしたか
* 最終的な案内内容

---

## 4. background

問い合わせに至った背景を記載する。

例

* 保守窓口の受付開始前であった
* 作業完了後の確認指示があった
* 店舗側で対処方法が分からなかった

---

## 5. issue_detail

問い合わせ内容や症状を具体的に記載する。

店舗担当者が何に困っていたのか分かるように記載する。

---

## 6. actions_taken

オペレータが実施した対応内容を配列で記載する。

例

[
"状況ヒアリング",
"遠隔接続",
"レジ再起動"
]

---

## 7. result

今回の対応結果を記載する。

例

* レジ再起動を実施
* 作業完了を確認
* 一次対応のみ実施

---

## 8. next_action

今後必要な対応を記載する。

例

* 改善しない場合は保守窓口へ連絡
* センター側で継続確認

---

## 9. category

以下から最も近いものを選択する。

{CATEGORIES}

---

## 10. subcategory

問い合わせ内容を20～50文字程度で具体的に表現する。

例

* 無線スキャナが反応しない
* レジ起動後の稼働確認
* Wi-Fi接続状況の確認
* 売上データ送信エラー

---

## 11. issue_type

以下から選択する。

* 障害
* 操作方法
* 設定変更
* 確認依頼
* 作業報告
* 問い合わせ

---

## 12. business_impact

業務影響度を判定する。

以下から選択する。

* 高
* 中
* 低

判定基準

* 高：会計停止、営業停止、売上影響
* 中：一部業務へ影響
* 低：確認のみ、影響軽微

---

## 13. resolved

以下から選択する。

* true
* false
* unknown

判定基準

* true：通話中に解決した
* false：未解決のまま終了した
* unknown：解決可否が判断できない

---

## 14. keywords

分析用キーワードを3～10件抽出する。

製品名、機器名、エラー内容、業務名を優先すること。

---

## 15. hallucination禁止

会話に存在しない内容を補完しないこと。

推測で原因を書かないこと。

不明な場合は「不明」とする。

---

# 出力形式

{{
"inquiries": [
{{
"title": "",
"summary": "",
"background": "",
"issue_detail": "",
"actions_taken": [],
"result": "",
"next_action": "",
"category": "",
"subcategory": "",
"issue_type": "",
"business_impact": "",
"resolved": "",
"keywords": []
}}
]
}}

JSON以外は出力しない。

以下が文字起こしデータです。

{TRANSCRIPT}"""


# ---------------------------------------------------------------------------
# 設定ファイルの読み書き
# ---------------------------------------------------------------------------
def _ensure_config_dir():
    os.makedirs(CONFIG_DIR, exist_ok=True)


def load_bedrock_config() -> dict:
    """Bedrock設定を読み込む。未設定の項目は既定値で補完する。"""
    cfg = dict(DEFAULT_BEDROCK_CONFIG)
    if os.path.exists(BEDROCK_CONFIG_PATH):
        try:
            with open(BEDROCK_CONFIG_PATH, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            if isinstance(data, dict):
                cfg.update({k: data[k] for k in DEFAULT_BEDROCK_CONFIG if k in data})
        except (OSError, json.JSONDecodeError):
            pass
    # max_tokens は数値へ正規化
    try:
        cfg["max_tokens"] = int(cfg.get("max_tokens") or DEFAULT_BEDROCK_CONFIG["max_tokens"])
    except (TypeError, ValueError):
        cfg["max_tokens"] = DEFAULT_BEDROCK_CONFIG["max_tokens"]
    return cfg


def save_bedrock_config(data: dict) -> dict:
    """Bedrock設定を保存する（既知のキーのみ採用）。保存後の内容を返す。"""
    _ensure_config_dir()
    cfg = load_bedrock_config()
    for key in DEFAULT_BEDROCK_CONFIG:
        if key in data and data[key] is not None:
            cfg[key] = data[key]
    try:
        cfg["max_tokens"] = int(cfg["max_tokens"])
    except (TypeError, ValueError):
        cfg["max_tokens"] = DEFAULT_BEDROCK_CONFIG["max_tokens"]
    with open(BEDROCK_CONFIG_PATH, "w", encoding="utf-8") as fp:
        json.dump(cfg, fp, ensure_ascii=False, indent=2)
    return cfg


def load_categories() -> list:
    """カテゴリ一覧を読み込む。未作成なら既定値を返す。"""
    if os.path.exists(CATEGORIES_PATH):
        try:
            with open(CATEGORIES_PATH, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            if isinstance(data, list):
                cats = [str(c).strip() for c in data if str(c).strip()]
                if cats:
                    return cats
        except (OSError, json.JSONDecodeError):
            pass
    return list(DEFAULT_CATEGORIES)


def save_categories(categories) -> list:
    """カテゴリ一覧を保存する。リスト/改行区切り文字列の双方を受け付ける。"""
    _ensure_config_dir()
    if isinstance(categories, str):
        items = [c.strip() for c in categories.splitlines()]
    else:
        items = [str(c).strip() for c in (categories or [])]
    items = [c for c in items if c]
    if not items:
        items = list(DEFAULT_CATEGORIES)
    with open(CATEGORIES_PATH, "w", encoding="utf-8") as fp:
        json.dump(items, fp, ensure_ascii=False, indent=2)
    return items


# ---------------------------------------------------------------------------
# プロンプト組み立て
# ---------------------------------------------------------------------------
def build_prompt(transcript: str, categories=None) -> str:
    """文字起こしとカテゴリ一覧からLLMへ渡すプロンプトを生成する。"""
    if categories is None:
        categories = load_categories()
    cat_block = "\n".join(f"* {c}" for c in categories)
    return PROMPT_TEMPLATE.format(CATEGORIES=cat_block, TRANSCRIPT=transcript)


# ---------------------------------------------------------------------------
# Bedrock(Claude) 呼び出し
# ---------------------------------------------------------------------------
def _extract_json(text: str) -> dict:
    """LLM応答テキストからJSON部分を取り出してパースする。

    ```json ... ``` のコードフェンスや前後の余計な文字が混ざっても
    最初の '{' から最後の '}' までを対象にして解析する。
    """
    s = text.strip()
    if s.startswith("```"):
        # 先頭フェンス行（```や```json）と末尾フェンスを除去
        s = s.split("\n", 1)[1] if "\n" in s else s
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(s[start:end + 1])
        raise


def analyze_transcript(transcript: str, config=None, categories=None) -> dict:
    """文字起こしをBedrockで整形し、パース済みJSON(dict)を返す。

    認証情報・モデルIDは config（未指定なら設定ファイル）から取得する。
    """
    transcript = (transcript or "").strip()
    if not transcript:
        raise ValueError("文字起こしデータが空です。")

    cfg = config or load_bedrock_config()
    if not cfg.get("aws_access_key") or not cfg.get("aws_secret_key"):
        raise ValueError(
            "AWS認証情報が未設定です。設定画面でAWS Access Key / Secret Keyを入力してください。"
        )
    if not cfg.get("model_id"):
        raise ValueError("モデルIDが未設定です。設定画面でモデルIDを入力してください。")

    try:
        from anthropic import AnthropicBedrock
    except ImportError as e:
        raise RuntimeError(
            "anthropic(Bedrock対応版)がインストールされていません。"
            "`pip install \"anthropic[bedrock]\"` を実行してください。"
        ) from e

    client = AnthropicBedrock(
        aws_access_key=cfg["aws_access_key"],
        aws_secret_key=cfg["aws_secret_key"],
        aws_session_token=cfg.get("aws_session_token") or None,
        aws_region=cfg.get("aws_region") or DEFAULT_BEDROCK_CONFIG["aws_region"],
    )

    prompt = build_prompt(transcript, categories=categories)
    resp = client.messages.create(
        model=cfg["model_id"],
        max_tokens=int(cfg.get("max_tokens") or DEFAULT_BEDROCK_CONFIG["max_tokens"]),
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    return _extract_json(text)

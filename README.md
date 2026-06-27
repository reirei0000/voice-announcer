# Voice Announcer

VOICEVOX を使用した音声アナウンサーアプリケーション。
FastAPIをベースにしたWeb API経由、またはCSVで設定されたスケジュールに従って、指定したテキストをVOICEVOX音声で自動発声・再生します。

---

## 配布パッケージからの実行方法 (Windows/Linux)

GitHub Actions の Artifacts からダウンロードしたビルド済みのパッケージを使用して実行する手順です。

### 起動手順と自動セットアップ

ダウンロードした ZIP パッケージ（サイズ: 約50MB）を解凍すると、実行ファイルのみが入っています。

```text
Voice-Announcer/ (解凍したフォルダ)
  └── voice-announcer.exe (または Linux では拡張子なしの voice-announcer)
```

1. **実行ファイルの起動**
   * **Windows**: `voice-announcer.exe` をダブルクリックして実行します。
   * **Linux**: ターミナルで実行権限を付与して起動します。
     ```bash
     chmod +x ./voice-announcer
     ./voice-announcer
     ```

2. **自動セットアップ（初回起動時のみ）**
   初回起動時のみ、実行ファイルが音声モデルデータ（約1.3GB）と辞書データ（約100MB）が不足していることを検知し、**自動的にインターネットからデータをダウンロードしてセットアップします。**
   
   コンソール画面に以下のようなログが表示されます。**完了するまで約1〜2分かかりますので、画面を閉じずにお待ちください。**
   ```text
   ============================================================
   VOICEVOXのモデルデータおよび辞書データが見つかりません。
   初回起動時の自動セットアップを開始します（約1〜2分かかります）...
   ============================================================
   📥 セットアップツールをダウンロード中: ...
   📦 必要な音声モデル・辞書データをダウンロード中 (VOICEVOX 0.16.4) ...
   ✅ ダウンロードと展開が正常に完了しました！
   ```

3. **起動完了**
   ダウンロードが完了すると、自動的に以下のフォルダ構造が作成され、Uvicornサーバーが起動します。
   ```text
   Voice-Announcer/
     ├── voice-announcer.exe (または voice-announcer)
     └── example/
           └── python/
                 ├── dict/ (辞書データ)
                 └── models/ (音声モデルデータ)
   ```
   **2回目以降の起動時は、この `example` フォルダを参照するため、ダウンロード処理は走らず一瞬で起動します。**

> [!WARNING]
> 作成された `example` フォルダを移動・削除したり、実行ファイルだけを別のフォルダに移動させると、再度自動セットアップが走ってしまいます。配置はそのままでご使用ください。

---

## ローカル開発環境での起動手順

リポジトリをクローンして、Python スクリプトとして直接開発・実行する場合の手順です。

### 1. 必要要件
* Python 3.11 (推奨)
* 仮想環境 (venv) を使用することを推奨します。

### 2. セットアップ

```bash
# 仮想環境の作成と有効化
python -m venv venv
source venv/bin/activate  # Windows の場合は venv\Scripts\activate

# 依存ライブラリのインストール
pip install -r requirements.txt
```

### 3. VOICEVOX Core アセットのダウンロードと導入

#### Windows の場合
```powershell
Invoke-WebRequest https://github.com/VOICEVOX/voicevox_core/releases/download/0.16.4/download-windows-x64.exe -OutFile ./download.exe
"y" | ./download.exe -o ./example/python --exclude c-api
# pip で wheel と onnxruntime をインストール
pip install https://github.com/VOICEVOX/voicevox_core/releases/download/0.16.4/voicevox_core-0.16.4-cp310-abi3-win_amd64.whl
pip install onnxruntime
```

#### Linux の場合
```bash
curl -L -o ./download https://github.com/VOICEVOX/voicevox_core/releases/download/0.16.4/download-linux-x64
chmod +x ./download
echo y | ./download -o ./example/python --exclude c-api
# pip で wheel と onnxruntime をインストール
pip install https://github.com/VOICEVOX/voicevox_core/releases/download/0.16.4/voicevox_core-0.16.4-cp310-abi3-manylinux_2_34_x86_64.whl
pip install onnxruntime
```

### 4. アプリケーションの実行

```bash
python main.py
```
*(ローカルで直接実行する場合でも、`example/` フォルダがまだ作成されていなければ、自動的にセットアップが走ります)*

---

## 主な機能

* **テキスト読み上げ Web API**: `/speak` エンドポイントにリクエスト（POST/GET）を送ることで、指定したスピーカーIDで発声させることができます。
* **音声の重複再生防止 (排他ロック)**: 複数の読み上げ要求が同時に届いた際、スレッドロックを用いて順番に再生させます。これにより、C++ネイティブレイヤーの同時発声による強制終了（クラッシュ）を防ぎます。
* **スケジュール再生**: `data.csv` に設定されたスケジュールに従って、自動的に定期アナウンス（時報など）を実行します。

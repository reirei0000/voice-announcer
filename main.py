import os
import sys
import io
import csv
import logging
import multiprocessing
import threading  # ─── 追加：排他制御用 ───
import urllib.request
import subprocess
import stat
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
import uvicorn
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger

# VOICEVOX 新APIによるインポート
from voicevox_core.blocking import Onnxruntime, OpenJtalk, Synthesizer, VoiceModelFile

import sounddevice as sd
import soundfile as sf

# ---------------------------------------------------------
# 初期設定とパスの解決
# ---------------------------------------------------------
app = FastAPI()
scheduler = BackgroundScheduler()
scheduler.start()

# ─── 追加：同時発声によるC++ネイティブクラッシュを防ぐためのロック ───
speaker_lock = threading.Lock()

logging.basicConfig(format="[%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_resource_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

def get_external_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, relative_path)

def setup_voicevox_assets(base_dir, dict_dir, models_dir):
    print("=" * 60)
    print("VOICEVOXのモデルデータおよび辞書データが見つかりません。")
    print("初回起動時の自動セットアップを開始します（約1〜2分かかります）...")
    print("=" * 60)
    
    if sys.platform == "win32":
        downloader_url = "https://github.com/VOICEVOX/voicevox_core/releases/download/0.16.4/download-windows-x64.exe"
        downloader_path = os.path.join(base_dir, "download.exe")
    else:
        downloader_url = "https://github.com/VOICEVOX/voicevox_core/releases/download/0.16.4/download-linux-x64"
        downloader_path = os.path.join(base_dir, "download")
        
    try:
        # 1. ダウンローダーの取得
        print(f"📥 セットアップツールをダウンロード中: {downloader_url}")
        urllib.request.urlretrieve(downloader_url, downloader_path)
        
        if sys.platform != "win32":
            st = os.stat(downloader_path)
            os.chmod(downloader_path, st.st_mode | stat.S_IEXEC)
            
        # 2. ダウンローダーの実行
        print("📦 必要な音声モデル・辞書データをダウンロード中 (VOICEVOX 0.16.4) ...")
        
        output_dir = os.path.join(base_dir, "example/python")
        result = subprocess.run(
            [downloader_path, "-o", output_dir, "--exclude", "c-api"],
            input="y\n",
            text=True,
            capture_output=True,
            check=True
        )
        print("✅ ダウンロードと展開が正常に完了しました！")
        
    except Exception as e:
        print(f"❌ セットアップ中にエラーが発生しました: {e}")
        if 'result' in locals() and result.stderr:
            print(f"詳細エラー: {result.stderr}")
        sys.exit(1)
    finally:
        # クリーンアップ：一時的なダウンローダーファイルを削除
        if os.path.exists(downloader_path):
            try:
                os.remove(downloader_path)
            except Exception:
                pass

ONNXRUNTIME_PATH = get_resource_path("example/python/onnxruntime/lib/libvoicevox_onnxruntime.so.1.17.3")
if sys.platform == "win32":
    ONNXRUNTIME_PATH = get_resource_path("onnxruntime.dll")

DICT_DIR = Path(get_external_path("example/python/dict/open_jtalk_dic_utf_8-1.11"))
MODELS_DIR = Path(get_external_path("example/python/models/vvms"))

# 辞書フォルダとモデルフォルダが存在するかチェックし、なければ自動セットアップを実行
if not DICT_DIR.exists() or not MODELS_DIR.exists():
    if hasattr(sys, '_MEIPASS'):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    setup_voicevox_assets(base_dir, DICT_DIR, MODELS_DIR)

# ---------------------------------------------------------
# VOICEVOX 初期化
# ---------------------------------------------------------
logger.info(f"ONNX Runtime をロード中: {ONNXRUNTIME_PATH}")
onnxruntime = Onnxruntime.load_once(filename=str(ONNXRUNTIME_PATH))

logger.info(f"Synthesizer を初期化中 (辞書: {DICT_DIR})")
synthesizer = Synthesizer(
    onnxruntime,
    OpenJtalk(DICT_DIR),
    acceleration_mode="AUTO",
    cpu_num_threads=max(multiprocessing.cpu_count(), 2)
)

loaded_vvms = set()
LOAD_MODEL_IDS = [0, 2, 3, 8, 10, 14]
for m_id in LOAD_MODEL_IDS:
    vvm_path = MODELS_DIR / f"{m_id}.vvm"
    if vvm_path.exists():
        logger.info(f"音声モデルをロード中: {vvm_path.name}")
        with VoiceModelFile.open(vvm_path) as model:
            synthesizer.load_voice_model(model)
        loaded_vvms.add(m_id)

# ---------------------------------------------------------
# 音声合成・再生のコア機能（排他ロック対応版）
# ---------------------------------------------------------
def speak_text(text: str, speaker_id: int = 2, speed: float = 1.0, se: str = "", display_text: str = ""):
    current_time = datetime.now().strftime('%H:%M:%S')
    log_text = display_text if display_text else text
    
    print(f"[{current_time}] 📥 リクエスト受信 (待機列に入ります): {log_text}")

    # ─── 【重要】ここでロックを取得。先客がいたら、終わるまでこの行でスレッドが自動待機します ───
    with speaker_lock:
        active_time = datetime.now().strftime('%H:%M:%S')
        print(f"[{active_time}] 📢 画面表示: {log_text}")
        print(f"[{active_time}] 🗣️ 発声中(ID:{speaker_id}, 速度:{speed}): {text}")
        
        # 1. 効果音再生
        if se and os.path.exists(se):
            try:
                se_data, se_fs = sf.read(se)
                sd.play(se_data, se_fs)
                sd.wait()
            except Exception as e:
                print(f"効果音の再生失敗: {e}")

        try:
            if speaker_id not in loaded_vvms:
                vvm_path = MODELS_DIR / f"{speaker_id}.vvm"
                if vvm_path.exists():
                    with VoiceModelFile.open(vvm_path) as model:
                        synthesizer.load_voice_model(model)
                    loaded_vvms.add(speaker_id)
                else:
                    speaker_id = 2

            # 2. 音声合成と再生
            audio_query = synthesizer.create_audio_query(text, speaker_id)
            audio_query.speed_scale = speed
            wave_bytes = synthesizer.synthesis(audio_query, speaker_id)
            
            data, fs = sf.read(io.BytesIO(wave_bytes))
            sd.play(data, fs)
            sd.wait() # 再生が終わるまでしっかり待つ
        except Exception as e:
            print(f"音声合成/再生エラー: {e}")
            
    # with ブロックを抜けると、自動的に「鍵」が解放され、次の順番待ちリクエストが動き出します

# ---------------------------------------------------------
# REST API (即時発声用)
# ---------------------------------------------------------
class SpeakRequest(BaseModel):
    text: str
    speaker_id: int = 2
    speed: float = 1.0
    se: str = ""
    display_text: str = ""

@app.post("/api/speak")
def api_speak(req: SpeakRequest):
    speak_text(text=req.text, speaker_id=req.speaker_id, speed=req.speed, se=req.se, display_text=req.display_text)
    return {"status": "success", "message": f"発声完了: {req.text}"}

# ---------------------------------------------------------
# Web UI
# ---------------------------------------------------------
@app.get("/")
def index():
    jobs = scheduler.get_jobs()
    parsed_jobs = []
    for job in jobs:
        next_run = job.next_run_time.strftime("%H:%M:%S") if job.next_run_time else "--:--:--"
        text = job.args[0] if len(job.args) > 0 else ""
        speaker_id = job.args[1] if len(job.args) > 1 else 2
        speed = job.args[2] if len(job.args) > 2 else 1.0
        se = job.args[3] if len(job.args) > 3 else ""
        display_text = job.args[4] if len(job.args) > 4 else ""
        
        parsed_jobs.append({
            "time": next_run,
            "text": text,
            "speaker_id": speaker_id,
            "speed": speed,
            "se": se,
            "display_text": display_text if display_text else text
        })
    
    parsed_jobs.sort(key=lambda x: x["time"])
    
    table_rows = ""
    if not parsed_jobs:
        table_rows = "<tr><td colspan='5' style='text-align:center; color:#999; padding:20px;'>スケジュールされた時報はありません</td></tr>"
    else:
        for j in parsed_jobs:
            char_name = {2:"めたん", 3:"ずんだもん", 8:"つむぎ", 0:"NPC"}.get(j["speaker_id"], f"ID:{j['speaker_id']}")
            table_rows += f"""
            <tr>
                <td style="padding:12px; border-bottom:1px solid #eee; font-weight:bold; color:#e74c3c; font-family:monospace; font-size:16px;">{j['time']}</td>
                <td style="padding:12px; border-bottom:1px solid #eee;">{char_name} (x{j['speed']})</td>
                <td style="padding:12px; border-bottom:1px solid #eee; color:#666; font-size:13px;">{j['se'] if j['se'] else '-'}</td>
                <td style="padding:12px; border-bottom:1px solid #eee; color:#444;">{j['text']}</td>
                <td style="padding:12px; border-bottom:1px solid #eee; font-weight:bold; color:#2c3e50;">{j['display_text']}</td>
            </tr>
            """

    html = f"""
    <!DOCTYPE html>
    <html lang="ja">
    <head>
        <meta charset="UTF-8">
        <title>統合アナウンス・コントロールパネル</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, sans-serif; background: #eef2f3; padding: 20px; color: #333; }}
            .container-master {{ max-width: 1000px; margin: auto; }}
            .clock-card {{ background: #2c3e50; color: white; padding: 15px; border-radius: 12px; text-align: center; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }}
            #clock {{ font-size: 42px; font-weight: bold; font-family: 'Courier New', Courier, monospace; letter-spacing: 3px; margin-top: 5px; color: #2ecc71; }}
            .wrapper {{ display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 20px; }}
            .card {{ background: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); flex: 1; min-width: 300px; }}
            .card-full {{ background: white; padding: 25px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); margin-top: 20px; }}
            h2 {{ color: #34495e; font-size: 18px; border-bottom: 2px solid #eee; padding-bottom: 10px; margin-top: 0; }}
            label {{ display: block; margin: 12px 0 4px; font-weight: bold; font-size: 14px; }}
            input[type="text"], input[type="number"], select {{ width: 100%; padding: 10px; border: 1px solid #ccc; border-radius: 6px; box-sizing: border-box; }}
            input[type="file"] {{ display: block; width: 100%; margin: 15px 0; padding: 15px; border: 1px dashed #3498db; background: #fafafa; border-radius: 6px; }}
            button {{ width: 100%; background: #3498db; color: white; border: none; padding: 12px; border-radius: 6px; font-weight: bold; cursor: pointer; margin-top: 15px; font-size: 16px; }}
            button:hover {{ background: #2980b9; }}
            .btn-speak {{ background: #e74c3c; }}
            .btn-speak:hover {{ background: #c0392b; }}
            #status_msg {{ text-align: center; margin-top: 10px; font-weight: bold; height: 20px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
            th {{ background: #f8f9fa; text-align: left; padding: 12px; border-bottom: 2px solid #ddd; color: #555; }}
        </style>
    </head>
    <body>
        <div class="container-master">
            <div class="clock-card">
                <div style="font-size: 14px; text-transform: uppercase; letter-spacing: 1px; color: #bdc3c7;">現在時刻</div>
                <div id="clock">00:00:00</div>
            </div>
            <div class="wrapper">
                <div class="card">
                    <h2>💬 即時割り込みアナウンス</h2>
                    <label for="speak_text">発話文章 (text)</label>
                    <input type="text" id="speak_text" placeholder="例: ３時をお知らせします" required>
                    <label for="display_text">画面表示 (display_text) ※省略可</label>
                    <input type="text" id="display_text" placeholder="例: 【情報】３時をお知らせします">
                    <label for="speaker_id">声の種類 (speaker_id)</label>
                    <select id="speaker_id">
                        <option value="2">四国めたん (ノーマル)</option>
                        <option value="3">ずんだもん (ノーマル)</option>
                        <option value="8">春日部つむぎ (ノーマル)</option>
                        <option value="0">NPC (デフォルト話者)</option>
                    </select>
                    <label for="speed">発声速度 (speed)</label>
                    <input type="number" id="speed" value="1.0" step="0.1" min="0.5" max="2.0">
                    <button class="btn-speak" onclick="sendSpeakRequest()">🔊 今すぐ発声</button>
                    <div id="status_msg"></div>
                </div>
                <div class="card">
                    <h2>📅 スケジュール登録 (CSV)</h2>
                    <p style="font-size: 13px; color: #666; line-height: 1.4;">
                        必須ヘッダー: <code>enable, time, text</code><br>
                        任意ヘッダー: <code>speaker_id, speed, se, display_text, adjust_time</code><br>
                        <br>
                        <strong>time の指定方法:</strong><br>
                        ・年月日＋時間: <code>2026/06/28 15:30:00</code><br>
                        ・時間のみ（年月日省略）: <code>15:30:00</code> （現在時刻から見て未来の直近の日時が自動でセットされます）<br>
                        <br>
                        <strong>adjust_time の指定:</strong><br>
                        ・<code>1</code> (または省略・空欄): 指定時刻にピッタリ言い終わるように、発話時間などを自動計算して少し前にアナウンスを開始します（デフォルト）。<br>
                        ・<code>0</code>: 指定時刻通りにアナウンス処理（効果音や音声合成）を直接開始します（時間調整なし）。<br>
                        <br>
                        <strong>CSVの記述例:</strong>
                        <pre style="background: #f8f9fa; padding: 10px; border-radius: 6px; font-size: 11px; overflow-x: auto; border: 1px solid #ddd; margin-top: 5px; font-family: monospace; white-space: pre; line-height: 1.2;">enable,time,text,speaker_id,speed,se,display_text,adjust_time
1,2026/06/28 20:00:00,こんばんは,2,1.0,,【夜のアナウンス】こんばんは,
1,12:00:00,いま１２時です,3,1.0,,【時報】いま１２時です,0</pre>
                    </p>
                    <form action="/upload" method="post" enctype="multipart/form-data">
                        <input type="file" name="file" accept=".csv" required>
                        <button type="submit">📂 スケジュールを転送</button>
                    </form>
                </div>
            </div>
            <div class="card-full">
                <h2>📋 現在登録中の時報スケジュール一覧</h2>
                <table>
                    <thead>
                        <tr>
                            <th style="width: 12%;">発言時刻</th>
                            <th style="width: 18%;">声の種類 (速度)</th>
                            <th style="width: 15%;">効果音(SE)</th>
                            <th style="width: 30%;">発話文章 (text)</th>
                            <th style="width: 25%;">画面表示テキスト</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows}
                    </tbody>
                </table>
            </div>
        </div>
        <script>
            function updateClock() {{
                const now = new Date();
                const hours = String(now.getHours()).padStart(2, '0');
                const minutes = String(now.getMinutes()).padStart(2, '0');
                const seconds = String(now.getSeconds()).padStart(2, '0');
                document.getElementById('clock').innerText = `${{hours}}:${{minutes}}:${{seconds}}`;
            }}
            setInterval(updateClock, 1000);
            updateClock();

            async function sendSpeakRequest() {{
                const text = document.getElementById('speak_text').value;
                const display_text = document.getElementById('display_text').value;
                const speaker_id = parseInt(document.getElementById('speaker_id').value);
                const speed = parseFloat(document.getElementById('speed').value);
                const statusMsg = document.getElementById('status_msg');

                if (!text) {{
                    statusMsg.innerText = "発話文章を入力してください。";
                    statusMsg.style.color = "red";
                    return;
                }}

                statusMsg.innerText = "🔊 送信中...";
                statusMsg.style.color = "#2980b9";

                try {{
                    const response = await fetch('/api/speak', {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ text: text, display_text: display_text, speaker_id: speaker_id, speed: speed }})
                    }});
                    if (response.ok) {{
                        statusMsg.innerText = "✅ 発声完了！";
                        statusMsg.style.color = "#27ae60";
                        setTimeout(() => statusMsg.innerText = "", 3000);
                    }} else {{
                        statusMsg.innerText = "❌ エラーが発生しました。";
                        statusMsg.style.color = "red";
                    }}
                }} catch (error) {{
                    statusMsg.innerText = "❌ 通信に失敗しました。";
                    statusMsg.style.color = "red";
                }}
            }}
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    scheduler.remove_all_jobs()
    content = await file.read()
    decoded = content.decode('utf-8-sig').splitlines()
    reader = csv.DictReader(decoded)
    
    count = 0
    for row in reader:
        enable_val = str(row.get('enable', '1')).strip().upper()
        if enable_val in ['0', 'OFF', 'FALSE', '']:
            continue

        time_str = str(row.get('time', '')).strip()
        text_str = str(row.get('text', '')).strip()
        if not time_str or not text_str:
            continue

        speaker_val = str(row.get('speaker_id', 'default')).strip().lower()
        speaker_id = 2 if speaker_val == 'default' else int(speaker_val) if speaker_val.isdigit() else 2

        try:
            speed = float(row.get('speed', 1.0))
        except ValueError:
            speed = 1.0

        se_str = str(row.get('se', '')).strip()
        display_text_str = str(row.get('display_text', '')).strip()
        adjust_val = str(row.get('adjust_time', '')).strip().upper()
        # 空欄（省略時）または明示的に 1/ON/TRUE が指定されている場合は有効
        adjust_enabled = adjust_val == "" or adjust_val in ['1', 'ON', 'TRUE']

        t = None
        formats = [
            "%Y/%m/%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%H:%M:%S"
        ]
        for fmt in formats:
            try:
                parsed = datetime.strptime(time_str, fmt)
                if fmt == "%H:%M:%S":
                    now = datetime.now()
                    candidate = now.replace(hour=parsed.hour, minute=parsed.minute, second=parsed.second, microsecond=0)
                    if candidate <= now:
                        from datetime import timedelta
                        candidate += timedelta(days=1)
                    t = candidate
                else:
                    t = parsed
                break
            except ValueError:
                continue

        if t:
            if adjust_enabled:
                # SEファイルの長さを自動取得
                se_duration = 0.0
                if se_str and os.path.exists(se_str):
                    try:
                        info = sf.info(se_str)
                        se_duration = info.duration
                    except Exception:
                        pass
                
                synth_duration = 0.8
                # 発話時間（文字数 * 0.18 / speed）
                talk_duration = len(text_str) * 0.18 / speed
                
                offset = se_duration + synth_duration + talk_duration
                from datetime import timedelta
                t = t - timedelta(seconds=offset)

            try:
                scheduler.add_job(
                    speak_text,
                    DateTrigger(run_date=t),
                    args=[text_str, speaker_id, speed, se_str, display_text_str]
                )
                count += 1
            except Exception:
                pass

    return RedirectResponse(url="/", status_code=303)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

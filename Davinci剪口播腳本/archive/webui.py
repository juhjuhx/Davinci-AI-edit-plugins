"""
Smart A-Roll WebUI v3.0 — 單一服務架構
合併後端功能，支援佇列處理、SSE 即時進度推送
"""

import json
import logging
import os
import queue
import sys
import threading
import time
import uuid
import webbrowser
from pathlib import Path
from typing import Any, Dict, Optional

from flask import Flask, Response, jsonify, render_template_string, request

# 匯入核心引擎
sys.path.insert(0, r"D:\666\Davinci剪口播腳本")
from davinci_aroll_v2 import (
    AnalysisConfig,
    AnalysisQueue,
    ArollAnalyzer,
    Segment,
    export_edl,
    export_json,
    is_media_file,
    setup_logging,
)

# 匯入 AI 視頻編輯器
try:
    from ai_editor import get_ai_editor
    AI_EDITOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"AI 視頻編輯器不可用: {e}")
    AI_EDITOR_AVAILABLE = False

# 匯入綜合視頻編輯器
try:
    from comprehensive_editor import get_comprehensive_editor
    COMPREHENSIVE_EDITOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"綜合視頻編輯器不可用: {e}")
    COMPREHENSIVE_EDITOR_AVAILABLE = False

# ════════════════════════════════════════════════════════
#  設定
# ════════════════════════════════════════════════════════

PROJECT_DIR = r"D:\666\Davinci剪口播腳本"
DEFAULT_MODEL = r"D:\666\Davinci剪口播腳本\models\Gemma-4-E2B-Uncensored-HauhauCS-Aggressive-Q2_K_P.gguf"
RESULT_DIR = os.path.join(PROJECT_DIR, "results")

# 確保結果目錄存在
Path(RESULT_DIR).mkdir(parents=True, exist_ok=True)

# 設定日誌
logger = setup_logging(os.path.join(PROJECT_DIR, "logs"))

# ════════════════════════════════════════════════════════
#  Flask 應用
# ════════════════════════════════════════════════════════

app = Flask(__name__)

# 佇列管理器
analysis_queue = AnalysisQueue()

# SSE 進度推送
progress_queues: Dict[str, queue.Queue] = {}
progress_lock = threading.Lock()

# 分析執行緒
analysis_thread = None
analysis_running = False


def get_progress_queue(task_id: str) -> queue.Queue:
    """取得或建立進度佇列"""
    with progress_lock:
        if task_id not in progress_queues:
            progress_queues[task_id] = queue.Queue()
        return progress_queues[task_id]


def push_progress(task_id: str, data: Dict[str, Any]):
    """推送進度更新"""
    with progress_lock:
        if task_id in progress_queues:
            progress_queues[task_id].put(data)


def cleanup_progress_queue(task_id: str):
    """清理進度佇列"""
    with progress_lock:
        progress_queues.pop(task_id, None)


def analysis_worker():
    """分析工作執行緒"""
    global analysis_running
    analysis_running = True
    
    while analysis_running:
        # 處理下一個任務
        task = analysis_queue.process_next()
        if task is None:
            time.sleep(1)
            continue
        
        task_id = task["task_id"]
        video_path = task["video_path"]
        
        logger.info(f"開始處理任務: {task_id}, 檔案: {video_path}")
        
        # 推送進度更新
        def progress_cb(stage: str, pct: float):
            data = {
                "task_id": task_id,
                "status": "processing",
                "progress": int(pct * 100),
                "message": f"{stage} ({int(pct*100)}%)"
            }
            push_progress(task_id, data)
        
        try:
            # 更新任務狀態
            task_data = analysis_queue.get_task(task_id)
            if task_data:
                push_progress(task_id, {
                    "task_id": task_id,
                    "status": "processing",
                    "progress": 0,
                    "message": "開始分析..."
                })
            
            # 執行分析
            config = task_data.get("config") if task_data else None
            if isinstance(config, dict):
                config = AnalysisConfig.from_dict(config)
            
            analyzer = ArollAnalyzer(config=config, progress_cb=progress_cb)
            segments, fps = analyzer.analyze(video_path)
            
            # 推送完成
            push_progress(task_id, {
                "task_id": task_id,
                "status": "completed",
                "progress": 100,
                "message": "分析完成",
                "result": [s.to_dict() for s in segments],
                "fps": fps
            })
            
            logger.info(f"任務完成: {task_id}")
            
        except Exception as e:
            error_msg = str(e)
            if "0xc000001d" in error_msg:
                error_msg = "GPU 不支援此模型（GTX 1050 Ti 需要使用較小的模型或 CPU 模式）"
            elif "Could not find module" in error_msg:
                error_msg = "找不到 CUDA DLL，請確認已安裝 CUDA"
            
            logger.error(f"分析失敗: {error_msg}", exc_info=True)
            push_progress(task_id, {
                "task_id": task_id,
                "status": "failed",
                "progress": 0,
                "message": f"失敗: {error_msg}",
                "error": error_msg
            })
    
    analysis_running = False


# 啟動分析執行緒
analysis_thread = threading.Thread(target=analysis_worker, daemon=True)
analysis_thread.start()


# ════════════════════════════════════════════════════════
#  輸入驗證
# ════════════════════════════════════════════════════════

def validate_video_path(video_path: str) -> Optional[str]:
    """驗證影片路徑"""
    if not video_path:
        return "未指定影片路徑"
    
    if not isinstance(video_path, str):
        return "影片路徑必須是字串"
    
    # 清理路徑（移除不可見字符）
    video_path = video_path.strip()
    video_path = video_path.replace('\u200b', '')  # 移除零寬空格
    video_path = video_path.replace('\u200e', '')  # 移除 LTR 標記
    video_path = video_path.replace('\u200f', '')  # 移除 RTL 標記
    video_path = video_path.replace('\u202a', '')  # 移除 LTR 嵌入
    video_path = video_path.replace('\u202b', '')  # 移除 RTL 嵌入
    video_path = video_path.replace('\u202c', '')  # 移除 POP 格式
    video_path = video_path.replace('\u202d', '')  # 移除 LTR 覆蓋
    video_path = video_path.replace('\u202e', '')  # 移除 RTL 覆蓋
    
    # 除錯日誌
    logger.debug(f"驗證路徑: repr={repr(video_path)}")
    
    # 檢查路徑是否合法
    try:
        path = Path(video_path)
    except (ValueError, OSError) as e:
        return f"無效的路徑: {e}"
    
    # 檢查檔案是否存在
    if not path.is_file():
        # 嘗試列出父目錄的內容
        parent = path.parent
        if parent.exists():
            files = [f.name for f in parent.iterdir() if f.is_file()]
            logger.debug(f"父目錄存在，檔案: {files[:10]}")
        return f"檔案不存在: {video_path}"
    
    # 檢查是否為支援的格式
    if not is_media_file(video_path):
        return f"不支援的檔案格式: {path.suffix}"
    
    return None


def validate_config(config: Dict[str, Any]) -> Optional[str]:
    """驗證設定"""
    if not isinstance(config, dict):
        return "設定必須是字典"
    
    # 驗證數值範圍
    if "batch_size" in config:
        bs = config["batch_size"]
        if not isinstance(bs, (int, float)) or bs < 1 or bs > 32:
            return "batch_size 必須在 1-32 之間"
    
    if "pause_gap" in config:
        pg = config["pause_gap"]
        if not isinstance(pg, (int, float)) or pg < 0.1 or pg > 10:
            return "pause_gap 必須在 0.1-10 之間"
    
    if "confidence_threshold" in config:
        ct = config["confidence_threshold"]
        if not isinstance(ct, (int, float)) or ct < 0 or ct > 1:
            return "confidence_threshold 必須在 0-1 之間"
    
    return None


# ════════════════════════════════════════════════════════
#  HTML 模板
# ════════════════════════════════════════════════════════

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>智能剪口播 v3.0</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, 'Segoe UI', sans-serif; }
body { background: #111; display: flex; justify-content: center; padding: 20px; min-height: 100vh; }

.dv-panel {
  background: #1e1e1e; border-radius: 8px; overflow: hidden;
  border: 1px solid #333; max-width: 600px; width: 100%;
}

.dv-titlebar {
  background: #2a2a2a; padding: 10px 16px;
  display: flex; align-items: center; gap: 8px;
  border-bottom: 1px solid #3a3a3a;
}
.dv-dots { display: flex; gap: 5px; }
.dv-dot { width: 10px; height: 10px; border-radius: 50%; }
.dv-title { flex: 1; font-size: 13px; color: #aaa; text-align: center; }

.dv-section { padding: 12px 16px; border-bottom: 1px solid #2e2e2e; }
.dv-section-label {
  font-size: 10px; letter-spacing: 0.08em; text-transform: uppercase;
  color: #666; margin-bottom: 8px;
}

.dv-row { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
.dv-row:last-child { margin-bottom: 0; }
.dv-label { font-size: 12px; color: #888; min-width: 80px; }

.dv-input {
  flex: 1; background: #111; border: 1px solid #3a3a3a; border-radius: 4px;
  padding: 8px 10px; font-size: 12px; color: #ccc; outline: none;
}
.dv-input:focus { border-color: #5a8af5; }

.dv-btn {
  background: #333; border: 1px solid #444; border-radius: 4px;
  padding: 8px 14px; font-size: 12px; color: #ccc; cursor: pointer;
  transition: all 0.2s;
}
.dv-btn:hover { background: #444; border-color: #555; }
.dv-btn.primary {
  background: linear-gradient(135deg, #2d5be3, #4a7af5);
  border-color: #3a6bf0;
  color: #fff;
  font-weight: 600;
  padding: 14px 0;
  width: 100%;
  font-size: 16px;
  border-radius: 6px;
  letter-spacing: 0.05em;
  box-shadow: 0 2px 8px rgba(45, 91, 227, 0.3);
}
.dv-btn.primary:hover {
  background: linear-gradient(135deg, #3a6bf0, #5a8af5);
  box-shadow: 0 4px 12px rgba(45, 91, 227, 0.5);
  transform: translateY(-1px);
}
.dv-btn.primary:disabled {
  background: #222;
  border-color: #333;
  color: #555;
  cursor: not-allowed;
  box-shadow: none;
  transform: none;
}
.dv-btn.stop { background: #3d1a1a; border-color: #5a2a2a; color: #d06060; padding: 12px 0; width: 100%; font-size: 14px; }
.dv-btn.stop:hover { background: #4d2a2a; }
.dv-btn.action { flex: 1; background: #252525; border-color: #383838; padding: 10px 0; text-align: center; font-size: 13px; }

.dv-select {
  flex: 1; background: #111; border: 1px solid #3a3a3a; border-radius: 4px;
  padding: 8px 10px; font-size: 12px; color: #ccc; outline: none;
}

.dv-progress-bar { height: 5px; background: #2a2a2a; border-radius: 3px; overflow: hidden; margin-bottom: 6px; }
.dv-progress-fill {
  height: 100%; border-radius: 3px;
  background: linear-gradient(90deg, #2d5be3, #5a8af5);
  transition: width 0.3s ease;
}
.dv-status { font-size: 12px; color: #888; min-height: 18px; }

.backend-badge {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 12px; border-radius: 12px;
  background: #1a2a1a; border: 1px solid #2a4a2a;
  font-size: 11px; color: #5aaf5a;
}
.backend-badge.offline { background: #2a1a1a; border-color: #4a2a2a; color: #af5a5a; }
.backend-dot { width: 6px; height: 6px; border-radius: 50%; background: #5aaf5a; }
.backend-badge.offline .backend-dot { background: #af5a5a; }

.stat-row { display: flex; gap: 8px; }
.stat-card {
  flex: 1; background: #151515; border: 1px solid #2a2a2a; border-radius: 5px;
  padding: 10px; text-align: center;
}
.stat-num { font-size: 20px; font-weight: 600; }
.stat-lbl { font-size: 10px; color: #555; margin-top: 3px; }
.stat-green { color: #5aaf5a; }
.stat-red   { color: #e05555; }
.stat-blue  { color: #5a8af5; }

.log-box {
  background: #111; border: 1px solid #2a2a2a; border-radius: 5px;
  padding: 10px; max-height: 180px; overflow-y: auto; font-family: 'Consolas', monospace;
  font-size: 11px; color: #888; line-height: 1.6;
}
.log-box .error { color: #e05555; }
.log-box .success { color: #5aaf5a; }
.log-box .info { color: #5a8af5; }

.queue-list {
  background: #111; border: 1px solid #2a2a2a; border-radius: 5px;
  padding: 10px; max-height: 200px; overflow-y: auto;
}
.queue-item {
  display: flex; align-items: center; gap: 8px;
  padding: 6px 0; border-bottom: 1px solid #222;
}
.queue-item:last-child { border-bottom: none; }
.queue-status {
  width: 8px; height: 8px; border-radius: 50%;
}
.queue-status.pending { background: #888; }
.queue-status.processing { background: #5a8af5; animation: pulse 1s infinite; }
.queue-status.completed { background: #5aaf5a; }
.queue-status.failed { background: #e05555; }
.queue-name { flex: 1; font-size: 11px; color: #ccc; }
.queue-progress { font-size: 10px; color: #666; }

.segment-list {
  background: #111; border: 1px solid #2a2a2a; border-radius: 5px;
  padding: 10px; max-height: 300px; overflow-y: auto;
}
.segment-item {
  display: flex; align-items: center; gap: 8px;
  padding: 8px 0; border-bottom: 1px solid #222;
}
.segment-item:last-child { border-bottom: none; }
.segment-item input[type="checkbox"] {
  width: 16px; height: 16px; cursor: pointer;
}
.segment-item .segment-text {
  flex: 1; font-size: 11px; color: #ccc;
}
.segment-item .segment-flags {
  font-size: 10px; color: #e05555;
}
.segment-item .segment-time {
  font-size: 10px; color: #666;
  font-family: 'Consolas', monospace;
}

@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.5; } }
.animating .dv-progress-fill { animation: pulse 1.4s ease-in-out infinite; }

.stage-indicator {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 8px;
  margin-bottom: 10px;
  background: #151515;
  border-radius: 6px;
  border: 1px solid #2a2a2a;
}
.stage-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  opacity: 0.35;
  transition: all 0.3s ease;
}
.stage-item.active {
  opacity: 1;
}
.stage-item.active .stage-icon {
  transform: scale(1.15);
}
.stage-item.done {
  opacity: 0.7;
}
.stage-icon {
  font-size: 22px;
  line-height: 1;
  transition: transform 0.3s ease;
}
.stage-name {
  font-size: 10px;
  color: #666;
  white-space: nowrap;
}
.stage-item.active .stage-name {
  color: #5a8af5;
  font-weight: 600;
}
.stage-item.done .stage-name {
  color: #5aaf5a;
}
.stage-arrow {
  font-size: 12px;
  color: #444;
}
@keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
.stage-item.active .stage-icon::after {
  content: '';
  display: block;
  width: 24px;
  height: 24px;
  border: 2px solid #5a8af5;
  border-top-color: transparent;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  position: absolute;
  top: 0;
  left: 0;
}
.stage-icon {
  position: relative;
  display: inline-block;
}
</style>
</head>
<body>

<div class="dv-panel" id="panel">
  <div class="dv-titlebar">
    <div class="dv-dots">
      <div class="dv-dot" style="background:#e05555"></div>
      <div class="dv-dot" style="background:#e09a22"></div>
      <div class="dv-dot" style="background:#5aaf5a"></div>
    </div>
    <div class="dv-title">智能剪口播 v3.0 — AI A-Roll 分析</div>
  </div>

  <div class="dv-section">
    <div class="dv-section-label">檔案設定</div>
    <div class="dv-row">
      <span class="dv-label">影片路徑</span>
      <input class="dv-input" id="videoPath" placeholder="輸入影片/音訊路徑...">
      <button class="dv-btn" onclick="browseFile()">選擇</button>
    </div>
    <div class="dv-row">
      <span class="dv-label">LLM 模型</span>
      <input class="dv-input" id="llmPath" value="{{ default_model }}">
    </div>
    <div class="dv-row" style="margin-top:10px">
      <button class="dv-btn primary" id="startBtn" onclick="addToQueue()">▶  開始分析</button>
    </div>
  </div>

  <div class="dv-section">
    <div class="dv-section-label">模型參數</div>
    <div class="dv-row">
      <span class="dv-label">Whisper</span>
      <select class="dv-select" id="whisperSize">
        <option>tiny</option><option>base</option><option>small</option>
        <option>medium</option><option selected>large-v3</option>
      </select>
      <span class="dv-label" style="min-width:40px;text-align:right">語言</span>
      <select class="dv-select" style="max-width:70px" id="lang">
        <option selected>zh</option><option>en</option><option>ja</option><option>ko</option>
      </select>
    </div>
    <div class="dv-row">
      <span class="dv-label">停頓閾值</span>
      <input type="range" id="pauseGap" min="0.5" max="5.0" step="0.1" value="2.0" style="flex:1">
      <span class="dv-label" style="min-width:40px;text-align:right" id="pauseVal">2.0s</span>
    </div>
    <div class="dv-row">
      <span class="dv-label">信心度</span>
      <input type="range" id="confidence" min="0.1" max="0.9" step="0.05" value="0.2" style="flex:1">
      <span class="dv-label" style="min-width:40px;text-align:right" id="confVal">0.2</span>
    </div>
    <div class="dv-row">
      <span class="dv-label">LLM 批次</span>
      <input type="range" id="batchSize" min="2" max="12" step="1" value="8" style="flex:1">
      <span class="dv-label" style="min-width:40px;text-align:right" id="batchVal">8</span>
    </div>
  </div>

  <div class="dv-section">
    <div class="dv-section-label">分析佇列</div>
    <div class="queue-list" id="queueList">
      <div style="text-align:center;color:#555;font-size:11px;padding:20px;">佇列為空</div>
    </div>
  </div>

  <div class="dv-section">
    <div class="dv-section-label">分析進度</div>
    <div class="stage-indicator" id="stageIndicator" style="display:none">
      <div class="stage-item" id="stageAudio">
        <div class="stage-icon">🎵</div>
        <div class="stage-name">音訊提取</div>
      </div>
      <div class="stage-arrow">→</div>
      <div class="stage-item" id="stageWhisper">
        <div class="stage-icon">🎤</div>
        <div class="stage-name">語音轉錄</div>
      </div>
      <div class="stage-arrow">→</div>
      <div class="stage-item" id="stageFilter">
        <div class="stage-icon">🔍</div>
        <div class="stage-name">信心過濾</div>
      </div>
      <div class="stage-arrow">→</div>
      <div class="stage-item" id="stageRule">
        <div class="stage-icon">⚙️</div>
        <div class="stage-name">規則引擎</div>
      </div>
      <div class="stage-arrow">→</div>
      <div class="stage-item" id="stageLLM">
        <div class="stage-icon">🤖</div>
        <div class="stage-name">LLM 分析</div>
      </div>
    </div>
    <div class="dv-progress-bar" id="progressBar">
      <div class="dv-progress-fill" id="progressFill" style="width:0%"></div>
    </div>
    <div class="dv-status" id="statusLabel">就緒，等待分析...</div>
  </div>

  <div class="dv-section" id="resultSection" style="display:none">
    <div class="dv-section-label">分析結果</div>
    <div class="stat-row" style="margin-bottom:10px">
      <div class="stat-card"><div class="stat-num stat-blue" id="statTotal">—</div><div class="stat-lbl">總片段</div></div>
      <div class="stat-card"><div class="stat-num stat-green" id="statUsable">—</div><div class="stat-lbl">可用</div></div>
      <div class="stat-card"><div class="stat-num stat-red" id="statCut">—</div><div class="stat-lbl">需剪輯</div></div>
    </div>
    <div class="dv-row" style="gap:8px">
      <button class="dv-btn action" onclick="exportJSON()">💾 匯出 JSON</button>
      <button class="dv-btn action" onclick="exportEDL()">📋 匯出 EDL</button>
    </div>
  </div>

  <div class="dv-section" id="adjustSection" style="display:none">
    <div class="dv-section-label">手動調整</div>
    <div class="segment-list" id="segmentList"></div>
    <div class="dv-row" style="gap:8px;margin-top:10px">
      <button class="dv-btn action" onclick="selectAll()">全選可用</button>
      <button class="dv-btn action" onclick="deselectAll()">全選不可用</button>
      <button class="dv-btn action" onclick="applyChanges()">套用變更</button>
    </div>
  </div>

  <div class="dv-section" id="aiEditorSection">
    <div class="dv-section-label">AI 視頻編輯</div>
    <div class="dv-row">
      <span class="dv-label">靜音閾值</span>
      <input type="range" id="silenceThreshold" min="0.001" max="0.05" step="0.001" value="0.006" style="flex:1">
      <span class="dv-label" style="min-width:40px;text-align:right" id="thresholdVal">0.006</span>
    </div>
    <div class="dv-row" style="gap:8px;margin-top:10px">
      <button class="dv-btn action" id="detectSilenceBtn" onclick="detectSilence()">🔇 偵測靜音</button>
      <button class="dv-btn action" id="removeSilenceBtn" onclick="removeSilence()">✂️ 移除靜音</button>
    </div>
    <div id="silenceResult" style="display:none;margin-top:10px">
      <div class="dv-status" id="silenceStatus"></div>
    </div>
  </div>

  <div class="dv-section" id="videoToolsSection">
    <div class="dv-section-label">視頻工具</div>
    <div class="dv-row" style="gap:8px">
      <button class="dv-btn action" onclick="getVideoInfo()">📊 視頻資訊</button>
      <button class="dv-btn action" onclick="generateSRT()">📝 生成字幕</button>
      <button class="dv-btn action" onclick="compressVideo()">📦 壓縮視頻</button>
    </div>
    <div class="dv-row" style="gap:8px;margin-top:8px">
      <span class="dv-label">速度</span>
      <input type="range" id="speedFactor" min="0.5" max="3.0" step="0.1" value="1.0" style="flex:1">
      <span class="dv-label" style="min-width:40px;text-align:right" id="speedVal">1.0x</span>
      <button class="dv-btn action" onclick="changeSpeed()">⏩ 調整速度</button>
    </div>
    <div id="videoToolsResult" style="display:none;margin-top:10px">
      <div class="dv-status" id="videoToolsStatus"></div>
    </div>
  </div>

  <div class="dv-section">
    <div class="dv-section-label">執行日誌</div>
    <div class="log-box" id="logBox"></div>
  </div>
</div>

<script>
let currentTaskId = null;
let eventSource = null;
let queuePollInterval = null;

const STAGES = ['audio', 'whisper', 'filter', 'rule', 'llm'];
const STAGE_NAMES = {
  'audio': '音訊提取',
  'whisper': '語音轉錄', 
  'filter': '信心過濾',
  'rule': '規則引擎',
  'llm': 'LLM 分析'
};

function resetStageIndicator() {
  STAGES.forEach(s => {
    const el = document.getElementById('stage' + s.charAt(0).toUpperCase() + s.slice(1));
    if (el) {
      el.classList.remove('active', 'done');
    }
  });
}

function updateStageIndicator(message) {
  const indicator = document.getElementById('stageIndicator');
  indicator.style.display = 'flex';
  
  let foundActive = false;
  for (let i = STAGES.length - 1; i >= 0; i--) {
    const stage = STAGES[i];
    const stageName = STAGE_NAMES[stage];
    const el = document.getElementById('stage' + stage.charAt(0).toUpperCase() + stage.slice(1));
    if (!el) continue;
    
    if (message.includes(stageName)) {
      el.classList.add('active');
      el.classList.remove('done');
      foundActive = true;
    } else if (!foundActive) {
      el.classList.remove('active');
      el.classList.add('done');
    } else {
      el.classList.remove('active', 'done');
    }
  }
}

function log(msg, type='info') {
  const box = document.getElementById('logBox');
  const time = new Date().toLocaleTimeString();
  box.innerHTML += `<div class="${type}">[${time}] ${msg}</div>`;
  box.scrollTop = box.scrollHeight;
}

function browseFile() {
  fetch('/api/browse-file')
    .then(r => r.json())
    .then(data => {
      if (data.path) {
        document.getElementById('videoPath').value = data.path;
        log('已選擇: ' + data.path.split('\\').pop().split('/').pop(), 'info');
      }
    })
    .catch(e => {
      log('選擇檔案失敗: ' + e, 'error');
    });
}

function addToQueue() {
  const video = document.getElementById('videoPath').value.trim();
  if (!video) {
    log('請先輸入影片路徑', 'error');
    return;
  }

  const startBtn = document.getElementById('startBtn');
  startBtn.disabled = true;
  startBtn.textContent = '處理中...';

  const config = {
    llm_path: document.getElementById('llmPath').value,
    whisper_model: document.getElementById('whisperSize').value,
    batch_size: parseInt(document.getElementById('batchSize').value),
    pause_gap: parseFloat(document.getElementById('pauseGap').value),
    confidence_threshold: parseFloat(document.getElementById('confidence').value)
  };

  log('加入佇列: ' + video.split('\\').pop().split('/').pop(), 'info');

  fetch('/api/add-to-queue', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({video_path: video, config: config})
  })
  .then(r => r.json())
  .then(data => {
    if (data.task_id) {
      log('已加入佇列，等待分析...', 'success');
      document.getElementById('videoPath').value = '';
      refreshQueue();
    } else {
      log('加入失敗: ' + (data.error || ''), 'error');
      startBtn.disabled = false;
      startBtn.textContent = '▶  開始分析';
    }
  })
  .catch(e => {
    log('錯誤: ' + e, 'error');
    startBtn.disabled = false;
    startBtn.textContent = '▶  開始分析';
  });
}

function refreshQueue() {
  fetch('/api/queue')
    .then(r => r.json())
    .then(data => {
      const list = document.getElementById('queueList');
      if (data.tasks.length === 0) {
        list.innerHTML = '<div style="text-align:center;color:#555;font-size:11px;padding:20px;">佇列為空</div>';
        return;
      }

      list.innerHTML = data.tasks.map(t => {
        const name = t.video_path.split('\\').pop().split('/').pop();
        return `<div class="queue-item">
          <div class="queue-status ${t.status}"></div>
          <div class="queue-name">${name}</div>
          <div class="queue-progress">${t.message}</div>
        </div>`;
      }).join('');

      // 找到正在處理的任務
      const processing = data.tasks.find(t => t.status === 'processing');
      if (processing && currentTaskId !== processing.task_id) {
        currentTaskId = processing.task_id;
        startSSE(currentTaskId);
      }

      // 顯示最新結果
      const completed = data.tasks.filter(t => t.status === 'completed').pop();
      if (completed && completed.result) {
        showResults(completed.result);
      }
    });
}

function startSSE(taskId) {
  if (eventSource) {
    eventSource.close();
  }

  resetStageIndicator();
  document.getElementById('stageIndicator').style.display = 'flex';
  document.getElementById('resultSection').style.display = 'none';

  eventSource = new EventSource('/api/progress/' + taskId);
  
  eventSource.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    const progress = data.progress || 0;
    const msg = data.message || '';
    document.getElementById('progressFill').style.width = progress + '%';
    document.getElementById('statusLabel').textContent = progress + '% - ' + msg;
    document.getElementById('progressBar').classList.add('animating');
    
    updateStageIndicator(msg);
    log(msg, 'info');

    if (data.status === 'completed') {
      eventSource.close();
      eventSource = null;
      document.getElementById('progressBar').classList.remove('animating');
      STAGES.forEach(s => {
        const el = document.getElementById('stage' + s.charAt(0).toUpperCase() + s.slice(1));
        if (el) { el.classList.remove('active'); el.classList.add('done'); }
      });
      document.getElementById('startBtn').disabled = false;
      document.getElementById('startBtn').textContent = '▶  開始分析';
      log('分析完成!', 'success');
      if (data.result) {
        showResults(data.result);
      }
      refreshQueue();
    } else if (data.status === 'failed') {
      eventSource.close();
      eventSource = null;
      document.getElementById('progressBar').classList.remove('animating');
      document.getElementById('stageIndicator').style.display = 'none';
      document.getElementById('startBtn').disabled = false;
      document.getElementById('startBtn').textContent = '▶  開始分析';
      log('分析失敗: ' + (data.error || ''), 'error');
      refreshQueue();
    }
  };

  eventSource.onerror = function() {
    eventSource.close();
    eventSource = null;
  };
}

function showResults(segments) {
  const total = segments.length;
  const usable = segments.filter(s => s.usable).length;
  const cut = total - usable;
  const flagged = segments.filter(s => s.flags && s.flags.length > 0).length;

  document.getElementById('statTotal').textContent = total;
  document.getElementById('statUsable').textContent = usable;
  document.getElementById('statCut').textContent = cut;
  document.getElementById('resultSection').style.display = 'block';

  log(`結果: ${total} 片段, ${usable} 可用, ${cut} 需剪輯`, 'success');
  
  if (cut === 0 && flagged === 0) {
    log('所有片段都沒有問題，不需要剪輯', 'info');
  } else if (flagged > 0) {
    log(`發現 ${flagged} 個有問題的片段`, 'info');
  }
  
  // 顯示各片段狀態
  segments.forEach((s, i) => {
    const flags = s.flags || [];
    if (flags.length > 0) {
      log(`片段 ${i+1}: ${s.text.substring(0, 20)}... [${flags.join(', ')}]`, 'info');
    }
  });
  
  // 顯示手動調整介面
  showSegmentEditor(segments);
}

function showSegmentEditor(segments) {
  const section = document.getElementById('adjustSection');
  section.style.display = 'block';
  
  const list = document.getElementById('segmentList');
  list.innerHTML = segments.map((s, i) => {
    const flags = s.flags || [];
    const flagStr = flags.length > 0 ? `<span class="segment-flags">[${flags.join(', ')}]</span>` : '';
    const timeStr = `${s.start.toFixed(1)}s - ${s.end.toFixed(1)}s`;
    return `
      <div class="segment-item">
        <input type="checkbox" id="seg-${i}" ${s.usable ? 'checked' : ''} 
               onchange="toggleSegment(${i}, this.checked)">
        <span class="segment-time">${timeStr}</span>
        <span class="segment-text">${s.text.substring(0, 40)}${s.text.length > 40 ? '...' : ''}</span>
        ${flagStr}
      </div>
    `;
  }).join('');
}

function toggleSegment(index, usable) {
  // 更新片段狀態
  log(`片段 ${index+1} 標記為 ${usable ? '可用' : '不可用'}`, 'info');
}

function selectAll() {
  const checkboxes = document.querySelectorAll('#segmentList input[type="checkbox"]');
  checkboxes.forEach(cb => {
    cb.checked = true;
  });
  log('已全選為可用', 'info');
}

function deselectAll() {
  const checkboxes = document.querySelectorAll('#segmentList input[type="checkbox"]');
  checkboxes.forEach(cb => {
    cb.checked = false;
  });
  log('已全選為不可用', 'info');
}

function applyChanges() {
  const checkboxes = document.querySelectorAll('#segmentList input[type="checkbox"]');
  const changes = [];
  checkboxes.forEach((cb, i) => {
    changes.push({
      index: i,
      usable: cb.checked
    });
  });
  
  // 重新計算統計
  const usable = changes.filter(c => c.usable).length;
  const cut = changes.length - usable;
  
  document.getElementById('statUsable').textContent = usable;
  document.getElementById('statCut').textContent = cut;
  
  log(`已套用變更: ${usable} 可用, ${cut} 需剪輯`, 'success');
  
  // 重新匯出 EDL
  exportEDL();
}

function exportJSON() {
  fetch('/api/export-json')
    .then(r => r.json())
    .then(data => {
      if (data.path) {
        log('JSON 已匯出: ' + data.path, 'success');
      } else {
        log('匯出失敗: ' + (data.error || ''), 'error');
      }
    });
}

function exportEDL() {
  fetch('/api/export-edl')
    .then(r => r.json())
    .then(data => {
      if (data.path) {
        log('EDL 已匯出: ' + data.path, 'success');
      } else {
        log('匯出失敗: ' + (data.error || ''), 'error');
      }
    });
}

// Slider 值顯示
document.getElementById('pauseGap').oninput = function() {
  document.getElementById('pauseVal').textContent = this.value + 's';
};
document.getElementById('confidence').oninput = function() {
  document.getElementById('confVal').textContent = this.value;
};
document.getElementById('batchSize').oninput = function() {
  document.getElementById('batchVal').textContent = this.value;
};
document.getElementById('silenceThreshold').oninput = function() {
  document.getElementById('thresholdVal').textContent = this.value;
};

// AI 視頻編輯功能
function detectSilence() {
  const video = document.getElementById('videoPath').value.trim();
  if (!video) {
    log('請先輸入影片路徑', 'error');
    return;
  }

  const threshold = parseFloat(document.getElementById('silenceThreshold').value);
  const btn = document.getElementById('detectSilenceBtn');
  btn.disabled = true;
  btn.textContent = '分析中...';

  log('開始偵測靜音...', 'info');

  fetch('/api/detect-silence', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({video_path: video, threshold: threshold})
  })
  .then(r => r.json())
  .then(data => {
    btn.disabled = false;
    btn.textContent = '🔇 偵測靜音';
    
    if (data.success) {
      const resultDiv = document.getElementById('silenceResult');
      resultDiv.style.display = 'block';
      
      const status = document.getElementById('silenceStatus');
      status.textContent = `偵測到 ${data.segments.length} 個靜音片段，總計 ${data.total_silence.toFixed(1)} 秒`;
      
      log(`偵測到 ${data.segments.length} 個靜音片段`, 'success');
      data.segments.forEach((s, i) => {
        log(`靜音 ${i+1}: ${s.start}s - ${s.end}s (${s.duration}s)`, 'info');
      });
    } else {
      log('偵測失敗: ' + (data.error || ''), 'error');
    }
  })
  .catch(e => {
    btn.disabled = false;
    btn.textContent = '🔇 偵測靜音';
    log('錯誤: ' + e, 'error');
  });
}

function removeSilence() {
  const video = document.getElementById('videoPath').value.trim();
  if (!video) {
    log('請先輸入影片路徑', 'error');
    return;
  }

  const threshold = parseFloat(document.getElementById('silenceThreshold').value);
  const btn = document.getElementById('removeSilenceBtn');
  btn.disabled = true;
  btn.textContent = '處理中...';

  log('開始移除靜音...', 'info');

  fetch('/api/remove-silence', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({video_path: video, threshold: threshold})
  })
  .then(r => r.json())
  .then(data => {
    btn.disabled = false;
    btn.textContent = '✂️ 移除靜音';
    
    if (data.success) {
      const resultDiv = document.getElementById('silenceResult');
      resultDiv.style.display = 'block';
      
      const status = document.getElementById('silenceStatus');
      status.textContent = `靜音移除完成: ${data.output_path}`;
      
      log('靜音移除完成: ' + data.output_path, 'success');
    } else {
      log('移除失敗: ' + (data.error || ''), 'error');
    }
  })
  .catch(e => {
    btn.disabled = false;
    btn.textContent = '✂️ 移除靜音';
    log('錯誤: ' + e, 'error');
  });
}

// 初始化
refreshQueue();
queuePollInterval = setInterval(refreshQueue, 3000);

// 綜合視頻編輯功能
function getVideoInfo() {
  const video = document.getElementById('videoPath').value.trim();
  if (!video) {
    log('請先輸入影片路徑', 'error');
    return;
  }

  log('正在取得視頻資訊...', 'info');

  fetch('/api/video-info', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({video_path: video})
  })
  .then(r => r.json())
  .then(data => {
    if (data.success) {
      const info = data.info;
      const resultDiv = document.getElementById('videoToolsResult');
      resultDiv.style.display = 'block';
      
      const status = document.getElementById('videoToolsStatus');
      status.textContent = `時長: ${info.duration.toFixed(1)}s | FPS: ${info.fps} | 解析度: ${info.width}x${info.height}`;
      
      log(`視頻資訊: ${info.width}x${info.height}, ${info.fps}fps, ${info.duration.toFixed(1)}s`, 'success');
    } else {
      log('取得資訊失敗: ' + (data.error || ''), 'error');
    }
  })
  .catch(e => {
    log('錯誤: ' + e, 'error');
  });
}

function generateSRT() {
  const video = document.getElementById('videoPath').value.trim();
  if (!video) {
    log('請先輸入影片路徑', 'error');
    return;
  }

  log('正在生成字幕（這可能需要一些時間）...', 'info');

  fetch('/api/generate-srt', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({video_path: video, language: 'zh', model_size: 'base'})
  })
  .then(r => r.json())
  .then(data => {
    if (data.success) {
      const resultDiv = document.getElementById('videoToolsResult');
      resultDiv.style.display = 'block';
      
      const status = document.getElementById('videoToolsStatus');
      status.textContent = `字幕已生成: ${data.srt_path}`;
      
      log('字幕生成完成: ' + data.srt_path, 'success');
    } else {
      log('字幕生成失敗: ' + (data.error || ''), 'error');
    }
  })
  .catch(e => {
    log('錯誤: ' + e, 'error');
  });
}

function compressVideo() {
  const video = document.getElementById('videoPath').value.trim();
  if (!video) {
    log('請先輸入影片路徑', 'error');
    return;
  }

  log('正在壓縮視頻...', 'info');

  fetch('/api/compress-video', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({video_path: video, crf: 23, preset: 'medium'})
  })
  .then(r => r.json())
  .then(data => {
    if (data.success) {
      const resultDiv = document.getElementById('videoToolsResult');
      resultDiv.style.display = 'block';
      
      const status = document.getElementById('videoToolsStatus');
      status.textContent = `壓縮完成: ${data.output_path}`;
      
      log('視頻壓縮完成: ' + data.output_path, 'success');
    } else {
      log('壓縮失敗: ' + (data.error || ''), 'error');
    }
  })
  .catch(e => {
    log('錯誤: ' + e, 'error');
  });
}

function changeSpeed() {
  const video = document.getElementById('videoPath').value.trim();
  if (!video) {
    log('請先輸入影片路徑', 'error');
    return;
  }

  const speedFactor = parseFloat(document.getElementById('speedFactor').value);
  log(`正在調整視頻速度為 ${speedFactor}x...`, 'info');

  fetch('/api/change-speed', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({video_path: video, speed_factor: speedFactor})
  })
  .then(r => r.json())
  .then(data => {
    if (data.success) {
      const resultDiv = document.getElementById('videoToolsResult');
      resultDiv.style.display = 'block';
      
      const status = document.getElementById('videoToolsStatus');
      status.textContent = `速度調整完成: ${data.output_path}`;
      
      log('視頻速度調整完成: ' + data.output_path, 'success');
    } else {
      log('速度調整失敗: ' + (data.error || ''), 'error');
    }
  })
  .catch(e => {
    log('錯誤: ' + e, 'error');
  });
}

// Slider 值顯示更新
document.getElementById('speedFactor').oninput = function() {
  document.getElementById('speedVal').textContent = this.value + 'x';
};
</script>
</body>
</html>
"""


# ════════════════════════════════════════════════════════
#  API 路由
# ════════════════════════════════════════════════════════

@app.route('/')
def index():
    """首頁"""
    return render_template_string(HTML_TEMPLATE, default_model=DEFAULT_MODEL)


@app.route('/api/browse-file')
def browse_file():
    """選擇檔案"""
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        path = filedialog.askopenfilename(
            title="選擇影片/音訊",
            filetypes=[
                ("Media Files", "*.mp4 *.mov *.mkv *.avi *.mxf *.webm *.flv *.wmv *.mp3 *.wav *.flac *.aac *.ogg *.m4a *.wma *.opus"),
                ("Video Files", "*.mp4 *.mov *.mkv *.avi *.mxf *.webm *.flv *.wmv"),
                ("Audio Files", "*.mp3 *.wav *.flac *.aac *.ogg *.m4a *.wma *.opus"),
                ("All Files", "*.*")
            ]
        )
        root.destroy()
        
        return jsonify({"path": path or ""})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/api/add-to-queue', methods=['POST'])
def add_to_queue():
    """加入佇列"""
    data = request.json
    if not data:
        return jsonify({"error": "無效的請求"})
    
    video_path = data.get("video_path", "")
    config_data = data.get("config", {})
    
    # 清理路徑（移除不可見字符）
    if isinstance(video_path, str):
        video_path = video_path.strip()
        video_path = video_path.replace('\u200b', '')  # 移除零寬空格
        video_path = video_path.replace('\u200e', '')  # 移除 LTR 標記
        video_path = video_path.replace('\u200f', '')  # 移除 RTL 標記
        video_path = video_path.replace('\u202a', '')  # 移除 LTR 嵌入
        video_path = video_path.replace('\u202b', '')  # 移除 RTL 嵌入
        video_path = video_path.replace('\u202c', '')  # 移除 POP 格式
        video_path = video_path.replace('\u202d', '')  # 移除 LTR 覆蓋
        video_path = video_path.replace('\u202e', '')  # 移除 RTL 覆蓋
    
    # 除錯日誌
    logger.info(f"收到路徑: {repr(video_path)}")
    
    # 驗證影片路徑
    error = validate_video_path(video_path)
    if error:
        logger.warning(f"路徑驗證失敗: {error}")
        return jsonify({"error": error})
    
    # 驗證設定
    error = validate_config(config_data)
    if error:
        return jsonify({"error": error})
    
    # 建立設定（從 config.json 載入）
    config_path = os.path.join(PROJECT_DIR, "config.json")
    if os.path.exists(config_path):
        config = AnalysisConfig.load(config_path)
        logger.info(f"已載入設定: {config_path}")
        logger.info(f"信心度閾值: {config.confidence_threshold}, 停頓閾值: {config.pause_gap_sec}")
    else:
        config = AnalysisConfig()
        logger.warning(f"找不到設定檔: {config_path}，使用預設值")
    
    if "llm_path" in config_data:
        config.llm_model_path = config_data["llm_path"]
    if "whisper_model" in config_data:
        config.whisper_model_size = config_data["whisper_model"]
    if "batch_size" in config_data:
        config.llm_batch_size = int(config_data["batch_size"])
    if "pause_gap" in config_data:
        config.pause_gap_sec = float(config_data["pause_gap"])
    if "confidence_threshold" in config_data:
        config.confidence_threshold = float(config_data["confidence_threshold"])
    
    # 加入佇列
    task_id = analysis_queue.add_task(video_path, config)
    
    return jsonify({"task_id": task_id})


@app.route('/api/queue')
def get_queue():
    """取得佇列狀態"""
    tasks = analysis_queue.get_all_tasks()
    return jsonify({"tasks": tasks})


@app.route('/api/progress/<task_id>')
def stream_progress(task_id: str):
    """SSE 進度推送"""
    def generate():
        q = get_progress_queue(task_id)
        while True:
            try:
                data = q.get(timeout=30)
                yield f"data: {json.dumps(data)}\n\n"
                
                if data.get("status") in ("completed", "failed"):
                    break
            except queue.Empty:
                # 發送心跳
                yield f"data: {json.dumps({'heartbeat': True})}\n\n"
        
        cleanup_progress_queue(task_id)
    
    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/export-json')
def export_json_api():
    """匯出 JSON"""
    # 取得最新完成的任務
    tasks = analysis_queue.get_all_tasks()
    completed = [t for t in tasks if t["status"] == "completed"]
    
    if not completed:
        return jsonify({"error": "沒有分析結果"})
    
    latest = completed[-1]
    segments = latest.get("result", [])
    
    if not segments:
        return jsonify({"error": "沒有分析結果"})
    
    # 匯出
    path = os.path.join(RESULT_DIR, "last_result.json")
    try:
        # 原子寫入
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
        return jsonify({"path": path})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/api/export-edl')
def export_edl_api():
    """匯出 EDL"""
    # 取得最新完成的任務
    tasks = analysis_queue.get_all_tasks()
    completed = [t for t in tasks if t["status"] == "completed"]
    
    if not completed:
        return jsonify({"error": "沒有分析結果"})
    
    latest = completed[-1]
    segments_data = latest.get("result", [])
    
    if not segments_data:
        return jsonify({"error": "沒有分析結果"})
    
    # 轉換為 Segment 物件
    segments = [Segment.from_dict(s) for s in segments_data]
    
    # 取得幀率
    fps = latest.get("fps", 24.0)
    
    # 匯出
    path = os.path.join(RESULT_DIR, "last_result.edl")
    try:
        export_edl(segments, path, fps=fps)
        return jsonify({"path": path})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/api/update-segment', methods=['POST'])
def update_segment():
    """更新片段狀態（手動調整）"""
    data = request.json
    if not data:
        return jsonify({"error": "無效的請求"})
    
    task_id = data.get("task_id")
    segment_index = data.get("segment_index")
    usable = data.get("usable")
    
    # 更新佇列中的任務結果
    with analysis_queue._lock:
        for task in analysis_queue._queue:
            if task["task_id"] == task_id:
                if task["result"] and segment_index < len(task["result"]):
                    task["result"][segment_index]["usable"] = usable
                    return jsonify({"success": True})
    return jsonify({"error": "找不到任務或片段"})


# ════════════════════════════════════════════════════════
#  AI 視頻編輯功能
# ════════════════════════════════════════════════════════

@app.route('/api/detect-silence', methods=['POST'])
def detect_silence():
    """偵測靜音片段"""
    if not AI_EDITOR_AVAILABLE:
        return jsonify({"error": "AI 視頻編輯器不可用，請安裝 librosa"})
    
    data = request.json
    if not data:
        return jsonify({"error": "無效的請求"})
    
    video_path = data.get("video_path")
    threshold = data.get("threshold", 0.006)
    step = data.get("step", 0.5)
    
    if not video_path or not os.path.isfile(video_path):
        return jsonify({"error": "找不到影片檔案"})
    
    try:
        editor = get_ai_editor()
        silence_segments = editor.detect_silence(
            video_path,
            threshold=threshold,
            step=step
        )
        
        return jsonify({
            "success": True,
            "segments": [
                {
                    "start": s.start,
                    "end": s.end,
                    "duration": s.duration
                }
                for s in silence_segments
            ],
            "total_silence": sum(s.duration for s in silence_segments)
        })
    except Exception as e:
        logger.error(f"靜音偵測失敗: {e}", exc_info=True)
        return jsonify({"error": str(e)})


@app.route('/api/remove-silence', methods=['POST'])
def remove_silence():
    """移除靜音並輸出影片"""
    if not AI_EDITOR_AVAILABLE:
        return jsonify({"error": "AI 視頻編輯器不可用，請安裝 librosa"})
    
    data = request.json
    if not data:
        return jsonify({"error": "無效的請求"})
    
    video_path = data.get("video_path")
    threshold = data.get("threshold", 0.006)
    step = data.get("step", 0.5)
    scale = data.get("scale", "-2:720")
    
    if not video_path or not os.path.isfile(video_path):
        return jsonify({"error": "找不到影片檔案"})
    
    # 產生輸出路徑
    video_name = Path(video_path).stem
    output_path = os.path.join(RESULT_DIR, f"{video_name}_no_silence.mp4")
    
    try:
        editor = get_ai_editor()
        result_path = editor.remove_silence(
            video_path,
            output_path,
            threshold=threshold,
            step=step,
            scale=scale
        )
        
        return jsonify({
            "success": True,
            "output_path": result_path,
            "message": "靜音移除完成"
        })
    except Exception as e:
        logger.error(f"靜音移除失敗: {e}", exc_info=True)
        return jsonify({"error": str(e)})


@app.route('/api/audio-energy', methods=['POST'])
def get_audio_energy():
    """取得音訊能量分佈"""
    if not AI_EDITOR_AVAILABLE:
        return jsonify({"error": "AI 視頻編輯器不可用，請安裝 librosa"})
    
    data = request.json
    if not data:
        return jsonify({"error": "無效的請求"})
    
    video_path = data.get("video_path")
    step = data.get("step", 0.5)
    
    if not video_path or not os.path.isfile(video_path):
        return jsonify({"error": "找不到影片檔案"})
    
    try:
        editor = get_ai_editor()
        energy_data = editor.get_audio_energy(video_path, step=step)
        
        return jsonify({
            "success": True,
            "energy": energy_data
        })
    except Exception as e:
        logger.error(f"音訊能量分析失敗: {e}", exc_info=True)
        return jsonify({"error": str(e)})


# ════════════════════════════════════════════════════════
#  綜合視頻編輯功能
# ════════════════════════════════════════════════════════

@app.route('/api/video-info', methods=['POST'])
def get_video_info():
    """取得視頻資訊"""
    if not COMPREHENSIVE_EDITOR_AVAILABLE:
        return jsonify({"error": "綜合視頻編輯器不可用"})
    
    data = request.json
    if not data:
        return jsonify({"error": "無效的請求"})
    
    video_path = data.get("video_path")
    if not video_path or not os.path.isfile(video_path):
        return jsonify({"error": "找不到影片檔案"})
    
    try:
        editor = get_comprehensive_editor()
        info = editor.get_video_info(video_path)
        
        return jsonify({
            "success": True,
            "info": {
                "path": info.path,
                "duration": info.duration,
                "fps": info.fps,
                "width": info.width,
                "height": info.height,
                "codec": info.codec,
                "audio_codec": info.audio_codec,
                "file_size": info.file_size
            }
        })
    except Exception as e:
        logger.error(f"取得視頻資訊失敗: {e}", exc_info=True)
        return jsonify({"error": str(e)})


@app.route('/api/generate-srt', methods=['POST'])
def generate_srt():
    """生成 SRT 字幕"""
    if not COMPREHENSIVE_EDITOR_AVAILABLE:
        return jsonify({"error": "綜合視頻編輯器不可用"})
    
    data = request.json
    if not data:
        return jsonify({"error": "無效的請求"})
    
    video_path = data.get("video_path")
    language = data.get("language", "zh")
    model_size = data.get("model_size", "base")
    
    if not video_path or not os.path.isfile(video_path):
        return jsonify({"error": "找不到影片檔案"})
    
    # 產生輸出路徑
    video_name = Path(video_path).stem
    output_path = os.path.join(RESULT_DIR, f"{video_name}.srt")
    
    try:
        editor = get_comprehensive_editor()
        result_path = editor.generate_srt(
            video_path,
            output_path,
            language=language,
            model_size=model_size
        )
        
        return jsonify({
            "success": True,
            "srt_path": result_path,
            "message": "字幕生成完成"
        })
    except Exception as e:
        logger.error(f"生成字幕失敗: {e}", exc_info=True)
        return jsonify({"error": str(e)})


@app.route('/api/compress-video', methods=['POST'])
def compress_video():
    """壓縮視頻"""
    if not COMPREHENSIVE_EDITOR_AVAILABLE:
        return jsonify({"error": "綜合視頻編輯器不可用"})
    
    data = request.json
    if not data:
        return jsonify({"error": "無效的請求"})
    
    video_path = data.get("video_path")
    crf = data.get("crf", 23)
    preset = data.get("preset", "medium")
    scale = data.get("scale")
    
    if not video_path or not os.path.isfile(video_path):
        return jsonify({"error": "找不到影片檔案"})
    
    # 產生輸出路徑
    video_name = Path(video_path).stem
    output_path = os.path.join(RESULT_DIR, f"{video_name}_compressed.mp4")
    
    try:
        editor = get_comprehensive_editor()
        result_path = editor.compress_video(
            video_path,
            output_path,
            crf=crf,
            preset=preset,
            scale=scale
        )
        
        return jsonify({
            "success": True,
            "output_path": result_path,
            "message": "視頻壓縮完成"
        })
    except Exception as e:
        logger.error(f"壓縮視頻失敗: {e}", exc_info=True)
        return jsonify({"error": str(e)})


@app.route('/api/cut-video', methods=['POST'])
def cut_video():
    """切割視頻"""
    if not COMPREHENSIVE_EDITOR_AVAILABLE:
        return jsonify({"error": "綜合視頻編輯器不可用"})
    
    data = request.json
    if not data:
        return jsonify({"error": "無效的請求"})
    
    video_path = data.get("video_path")
    start_time = data.get("start_time")
    end_time = data.get("end_time")
    
    if not video_path or not os.path.isfile(video_path):
        return jsonify({"error": "找不到影片檔案"})
    
    if start_time is None or end_time is None:
        return jsonify({"error": "請指定開始和結束時間"})
    
    # 產生輸出路徑
    video_name = Path(video_path).stem
    output_path = os.path.join(RESULT_DIR, f"{video_name}_cut.mp4")
    
    try:
        editor = get_comprehensive_editor()
        result_path = editor.cut_video(
            video_path,
            output_path,
            start_time=start_time,
            end_time=end_time
        )
        
        return jsonify({
            "success": True,
            "output_path": result_path,
            "message": "視頻切割完成"
        })
    except Exception as e:
        logger.error(f"切割視頻失敗: {e}", exc_info=True)
        return jsonify({"error": str(e)})


@app.route('/api/concat-videos', methods=['POST'])
def concat_videos():
    """拼接多個視頻"""
    if not COMPREHENSIVE_EDITOR_AVAILABLE:
        return jsonify({"error": "綜合視頻編輯器不可用"})
    
    data = request.json
    if not data:
        return jsonify({"error": "無效的請求"})
    
    video_paths = data.get("video_paths")
    scale = data.get("scale")
    
    if not video_paths or len(video_paths) < 2:
        return jsonify({"error": "請提供至少兩個影片檔案"})
    
    # 檢查檔案是否存在
    for path in video_paths:
        if not os.path.isfile(path):
            return jsonify({"error": f"找不到影片檔案: {path}"})
    
    # 產生輸出路徑
    output_path = os.path.join(RESULT_DIR, "concat_output.mp4")
    
    try:
        editor = get_comprehensive_editor()
        result_path = editor.concat_videos(
            video_paths,
            output_path,
            scale=scale
        )
        
        return jsonify({
            "success": True,
            "output_path": result_path,
            "message": "視頻拼接完成"
        })
    except Exception as e:
        logger.error(f"拼接視頻失敗: {e}", exc_info=True)
        return jsonify({"error": str(e)})


@app.route('/api/change-speed', methods=['POST'])
def change_speed():
    """改變視頻速度"""
    if not COMPREHENSIVE_EDITOR_AVAILABLE:
        return jsonify({"error": "綜合視頻編輯器不可用"})
    
    data = request.json
    if not data:
        return jsonify({"error": "無效的請求"})
    
    video_path = data.get("video_path")
    speed_factor = data.get("speed_factor", 1.0)
    
    if not video_path or not os.path.isfile(video_path):
        return jsonify({"error": "找不到影片檔案"})
    
    if speed_factor <= 0:
        return jsonify({"error": "速度因子必須大於 0"})
    
    # 產生輸出路徑
    video_name = Path(video_path).stem
    output_path = os.path.join(RESULT_DIR, f"{video_name}_speed_{speed_factor}x.mp4")
    
    try:
        editor = get_comprehensive_editor()
        result_path = editor.change_speed(
            video_path,
            output_path,
            speed_factor=speed_factor
        )
        
        return jsonify({
            "success": True,
            "output_path": result_path,
            "message": f"視頻速度調整為 {speed_factor}x"
        })
    except Exception as e:
        logger.error(f"改變速度失敗: {e}", exc_info=True)
        return jsonify({"error": str(e)})


# ════════════════════════════════════════════════════════
#  啟動
# ════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 50)
    print("  Smart A-Roll WebUI v3.0")
    print("  http://127.0.0.1:8888")
    print("=" * 50)
    webbrowser.open("http://127.0.0.1:8888")
    app.run(host="127.0.0.1", port=8888, debug=False, threaded=True)

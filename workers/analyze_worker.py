"""
分析佇列 Worker
- 單執行緒背景處理分析任務
- SSE 推送進度給前端
- 任務狀態查詢
"""

import logging
import os
import queue
import sys
import threading
import time
import uuid
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional

# 將父目錄加入路徑
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.analyzer import Analyzer
from core.config import get_config
from core.edl import export_all
from core.llm import get_llm
from core.models import AnalysisResult, QueueTask

logger = logging.getLogger("smart_aroll.worker")


class AnalyzeWorker:
    """分析 worker (單執行緒佇列)"""

    def __init__(self, config=None):
        self.config = config or get_config()
        self.analyzer = Analyzer(self.config)

        # 嘗試載入 LLM
        if self.config.llm.get("enabled", False):
            llm = get_llm(self.config)
            if llm.is_ready:
                self.analyzer.set_llm(llm)
                logger.info("LLM 已掛載到 analyzer")
            else:
                logger.warning("LLM 載入失敗,使用純規則模式")

        self.queue: queue.Queue = queue.Queue()
        self.tasks: Dict[str, QueueTask] = {}
        self.results: Dict[str, AnalysisResult] = {}
        self.subscribers: Dict[str, List[Callable]] = {}
        self._lock = threading.Lock()
        self._thread = None
        self._stop = False
        self.current_task: Optional[QueueTask] = None

    def start(self):
        """啟動 worker 執行緒"""
        if self._thread and self._thread.is_alive():
            return
        self._stop = False
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="AnalyzeWorker")
        self._thread.start()
        logger.info("✓ AnalyzeWorker 已啟動")

    def stop(self):
        """停止 worker"""
        self._stop = True
        if self._thread:
            self._thread.join(timeout=5)

    def submit(self, video_path: str) -> str:
        """提交分析任務,回傳 task_id"""
        task_id = str(uuid.uuid4())[:8]
        task = QueueTask(id=task_id, video_path=video_path)
        with self._lock:
            self.tasks[task_id] = task
        self.queue.put(task_id)
        logger.info(f"任務已加入佇列: {task_id} ({video_path})")
        return task_id

    def get_task(self, task_id: str) -> Optional[QueueTask]:
        with self._lock:
            return self.tasks.get(task_id)

    def get_result(self, task_id: str) -> Optional[AnalysisResult]:
        with self._lock:
            return self.results.get(task_id)

    def get_status(self) -> Dict[str, Any]:
        """取得佇列狀態"""
        with self._lock:
            return {
                "queue_size": self.queue.qsize(),
                "current": asdict(self.current_task) if self.current_task else None,
                "tasks": [asdict(t) for t in self.tasks.values()],
            }

    def subscribe(self, task_id: str, callback: Callable):
        """訂閱任務進度"""
        with self._lock:
            self.subscribers.setdefault(task_id, []).append(callback)

    def _notify(self, task_id: str, event: str, data: Any):
        """通知訂閱者"""
        with self._lock:
            callbacks = list(self.subscribers.get(task_id, []))
        for cb in callbacks:
            try:
                cb(event, data)
            except Exception as e:
                logger.error(f"訂閱 callback 失敗: {e}", exc_info=True)

    def _run_loop(self):
        """主迴圈"""
        logger.info("Worker 開始處理任務")
        while not self._stop:
            try:
                task_id = self.queue.get(timeout=1.0)
            except queue.Empty:
                continue

            task = self.get_task(task_id)
            if task is None:
                continue

            with self._lock:
                self.current_task = task
                task.status = "running"
                task.started_at = time.time()
                task.stage = "starting"
                task.message = "準備開始..."

            self._notify(task_id, "started", {"video_path": task.video_path})

            try:
                # 進度回呼
                def progress_cb(stage, message, percent):
                    task.stage = stage
                    task.message = message
                    task.progress = percent
                    self._notify(
                        task_id,
                        "progress",
                        {
                            "stage": stage,
                            "message": message,
                            "percent": percent,
                        },
                    )

                result = self.analyzer.analyze(task.video_path, progress_callback=progress_cb)

                with self._lock:
                    self.results[task_id] = result
                    task.result = result
                    task.status = "done"
                    task.progress = 100
                    task.finished_at = time.time()
                    task.stage = "done"
                    task.message = "分析完成"

                # 自動匯出
                try:
                    output_dir = self.config.export.get("result_dir")
                    if output_dir:
                        paths = export_all(
                            result,
                            output_dir,
                            fps=result.fps,
                            track_index=self.config.export.get("dv_track_index", 1),
                        )
                        task.message = f"已輸出至 {output_dir}"
                        self._notify(task_id, "exported", paths)
                except Exception as e:
                    logger.error(f"自動匯出失敗: {e}")

                self._notify(task_id, "done", result.to_dict())

            except Exception as e:
                logger.error(f"任務 {task_id} 失敗: {e}", exc_info=True)
                task.status = "error"
                task.error = str(e)
                task.finished_at = time.time()
                self._notify(task_id, "error", {"error": str(e)})

            finally:
                with self._lock:
                    self.current_task = None
                self.queue.task_done()


# 全域單例
_worker: Optional[AnalyzeWorker] = None


def get_worker() -> AnalyzeWorker:
    global _worker
    if _worker is None:
        _worker = AnalyzeWorker()
        _worker.start()
    return _worker

import asyncio
import uuid
import time
import logging
from typing import Dict, Any, List, Set, Optional, Callable
from dataclasses import dataclass, field
from fastapi import WebSocket

logger = logging.getLogger("voicesolate.job_manager")

@dataclass
class JobRecord:
    job_id: str
    job_type: str
    params: Dict[str, Any]
    status: str = "queued"  # queued, running, completed, failed, cancelled
    progress: float = 0.0    # 0.0 to 100.0
    stage: str = "init"
    message: str = "Queued"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    logs: List[Dict[str, Any]] = field(default_factory=list)
    workers: Dict[str, Any] = field(default_factory=dict)
    _cancel_requested: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "status": self.status,
            "progress": round(self.progress, 1),
            "stage": self.stage,
            "message": self.message,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
            "result": self.result,
            "workers": self.workers,
            "logs_count": len(self.logs),
            "recent_logs": self.logs[-20:] if self.logs else []
        }

class JobManager:
    """
    Central background job coordinator and real-time WebSocket event broadcaster.
    """
    def __init__(self):
        self.jobs: Dict[str, JobRecord] = {}
        self.active_sockets: Set[WebSocket] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        self._loop = loop

    async def connect_socket(self, websocket: WebSocket):
        await websocket.accept()
        self.active_sockets.add(websocket)
        logger.info(f"WebSocket client connected. Total clients: {len(self.active_sockets)}")
        # Send initial state
        await websocket.send_json({
            "type": "connection_ack",
            "timestamp": time.time(),
            "active_jobs": [j.to_dict() for j in self.jobs.values() if j.status in ("queued", "running")]
        })

    def disconnect_socket(self, websocket: WebSocket):
        self.active_sockets.discard(websocket)
        logger.info(f"WebSocket client disconnected. Total clients: {len(self.active_sockets)}")

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcasts a JSON message to all active WebSocket clients."""
        if not self.active_sockets:
            return
        dead_sockets = set()
        for ws in self.active_sockets:
            try:
                await ws.send_json(message)
            except Exception:
                dead_sockets.add(ws)
        self.active_sockets.difference_update(dead_sockets)

    def sync_broadcast(self, message: Dict[str, Any]):
        """Thread-safe non-async call to broadcast via the running event loop."""
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self.broadcast(message), self._loop)

    def create_job(self, job_type: str, params: Dict[str, Any]) -> JobRecord:
        job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        job = JobRecord(job_id=job_id, job_type=job_type, params=params)
        self.jobs[job_id] = job
        self.sync_broadcast({
            "type": "job_created",
            "job": job.to_dict()
        })
        return job

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        return self.jobs.get(job_id)

    def cancel_job(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        if job and job.status in ("queued", "running"):
            job._cancel_requested = True
            job.status = "cancelled"
            job.message = "Cancelled by user"
            job.updated_at = time.time()
            self.sync_broadcast({
                "type": "job_update",
                "job": job.to_dict()
            })
            return True
        return False

    def is_cancelled(self, job_id: str) -> bool:
        job = self.jobs.get(job_id)
        return job._cancel_requested if job else False

    def update_job(
        self,
        job_id: str,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        stage: Optional[str] = None,
        status: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ):
        job = self.jobs.get(job_id)
        if not job:
            return
        job.updated_at = time.time()
        if progress is not None:
            job.progress = min(100.0, max(0.0, float(progress)))
        if message is not None:
            job.message = message
        if stage is not None:
            job.stage = stage
        if status is not None:
            job.status = status
        if result is not None:
            job.result = result
        if error is not None:
            job.error = error
            job.status = "failed"

        self.sync_broadcast({
            "type": "job_update",
            "job": job.to_dict()
        })

    def log(self, job_id: str, level: str, text: str):
        job = self.jobs.get(job_id)
        entry = {
            "timestamp": time.time(),
            "level": level.upper(),
            "text": text
        }
        if job:
            job.logs.append(entry)
        self.sync_broadcast({
            "type": "job_log",
            "job_id": job_id,
            "entry": entry
        })

    def update_worker_state(
        self,
        job_id: str,
        worker_id: str,
        state: str,
        chunk_start: float,
        chunk_end: float,
        snippet: str = "",
        confidence: Optional[float] = None
    ):
        """
        Emits worker divide-and-conquer telemetry for the search procedure radar.
        States: 'scanning', 'snapping', 'matched', 'idle', 'error'
        """
        job = self.jobs.get(job_id)
        worker_info = {
            "worker_id": worker_id,
            "state": state,
            "chunk_start": round(chunk_start, 2),
            "chunk_end": round(chunk_end, 2),
            "snippet": snippet[:80] if snippet else "",
            "confidence": round(confidence, 1) if confidence is not None else None,
            "updated_at": time.time()
        }
        if job:
            job.workers[worker_id] = worker_info

        self.sync_broadcast({
            "type": "worker_event",
            "job_id": job_id,
            "worker": worker_info
        })

    def notify_clip_discovered(self, job_id: str, clip_data: Dict[str, Any]):
        self.sync_broadcast({
            "type": "clip_discovered",
            "job_id": job_id,
            "clip": clip_data
        })

job_manager = JobManager()

"""
Watchdog — System health monitoring, heartbeat, auto-recovery.
"""
import asyncio
import os
import time
import psutil
from pathlib import Path

from src.utils import setup_logging, utc_now

logger = setup_logging()

HEARTBEAT_FILE = Path(os.getenv("HEARTBEAT_FILE", "data/heartbeat.txt"))
MEMORY_LIMIT_MB = int(os.getenv("MEMORY_LIMIT_MB", "512"))


class Watchdog:
    def __init__(self, notifier=None):
        self.notifier = notifier
        self.is_healthy: bool = True
        self.restart_count: int = 0
        self._last_heartbeat: float = time.time()
        self._start_time: float = time.time()
        self._error_count: int = 0
        self._last_errors: list = []
        self.component_status: dict = {
            "memory": True,
            "disk": True,
            "network": True,
        }

    def heartbeat(self):
        """Call every cycle to signal the bot is alive."""
        self._last_heartbeat = time.time()
        try:
            HEARTBEAT_FILE.parent.mkdir(parents=True, exist_ok=True)
            HEARTBEAT_FILE.write_text(str(time.time()))
        except Exception:
            pass

    def record_error(self, error_str: str):
        self._error_count += 1
        self._last_errors.append({
            "time": utc_now().isoformat(),
            "error": error_str[:200],
        })
        self._last_errors = self._last_errors[-10:]  # keep last 10

    def _check_memory(self) -> bool:
        try:
            process = psutil.Process(os.getpid())
            mem_mb = process.memory_info().rss / (1024 * 1024)
            ok = mem_mb < MEMORY_LIMIT_MB
            if not ok:
                import gc
                gc.collect()
                logger.warning("Memory high: %.0f MB — GC triggered", mem_mb)
            self.component_status["memory"] = True
            return mem_mb
        except Exception:
            return 0

    def _check_disk(self) -> bool:
        try:
            usage = psutil.disk_usage("/")
            free_pct = usage.free / usage.total
            self.component_status["disk"] = free_pct > 0.05
            if free_pct < 0.10:
                logger.warning("Disk space low: %.1f%% free", free_pct * 100)
            return free_pct
        except Exception:
            return 1.0

    def get_status(self) -> dict:
        mem_mb = self._check_memory()
        disk_free = self._check_disk()
        try:
            process = psutil.Process(os.getpid())
            cpu_pct = process.cpu_percent(interval=0.1)
        except Exception:
            cpu_pct = 0.0

        uptime = time.time() - self._start_time

        return {
            "is_healthy": self.is_healthy,
            "components": self.component_status.copy(),
            "restart_count": self.restart_count,
            "memory_mb": round(mem_mb, 1) if isinstance(mem_mb, float) else 0,
            "cpu_percent": round(cpu_pct, 1),
            "last_heartbeat": self._last_heartbeat,
            "uptime_seconds": uptime,
            "uptime_hours": round(uptime / 3600, 1),
            "error_count": self._error_count,
            "last_errors": self._last_errors[-3:],
        }

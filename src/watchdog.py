import asyncio
import os
import time
import psutil
from pathlib import Path
from typing import Optional

from src.utils import setup_logging, utc_now

logger = setup_logging()

HEARTBEAT_FILE = Path("data/heartbeat.txt")
HEARTBEAT_INTERVAL = 10
MAX_HEARTBEAT_AGE = 60
MEMORY_LIMIT_MB = 512


class Watchdog:
    def __init__(self, exchange, database, notifier=None):
        self.exchange = exchange
        self.db = database
        self.notifier = notifier
        self.is_healthy: bool = True
        self.last_heartbeat: float = time.time()
        self.restart_count: int = 0
        self.max_restarts: int = 10
        self.component_status: dict = {
            "exchange_api": True,
            "websocket_kline": True,
            "database": True,
            "memory": True,
            "disk": True,
        }
        self._running = False

    async def start(self):
        self._running = True
        logger.info("Watchdog started")
        asyncio.create_task(self._heartbeat_loop())
        asyncio.create_task(self._health_check_loop())

    async def stop(self):
        self._running = False

    def beat(self):
        self.last_heartbeat = time.time()
        try:
            HEARTBEAT_FILE.parent.mkdir(parents=True, exist_ok=True)
            HEARTBEAT_FILE.write_text(str(time.time()))
        except Exception:
            pass

    async def _heartbeat_loop(self):
        while self._running:
            self.beat()
            await asyncio.sleep(HEARTBEAT_INTERVAL)

    async def _health_check_loop(self):
        while self._running:
            await asyncio.sleep(30)
            try:
                await self._check_all()
            except Exception as e:
                logger.error("Health check error: %s", e)

    async def _check_all(self):
        await self._check_exchange_api()
        self._check_websockets()
        self._check_database()
        self._check_memory()
        self._check_disk()

        unhealthy = [k for k, v in self.component_status.items() if not v]
        if unhealthy:
            self.is_healthy = False
            logger.warning("Unhealthy components: %s", unhealthy)
            await self._attempt_recovery(unhealthy)
        else:
            self.is_healthy = True

    async def _check_exchange_api(self):
        try:
            price = await self.exchange.get_price("BTCUSDT")
            self.component_status["exchange_api"] = price > 0
        except Exception:
            self.component_status["exchange_api"] = False

    def _check_websockets(self):
        self.component_status["websocket_kline"] = self.exchange.is_ws_connected("kline")

    def _check_database(self):
        try:
            from sqlalchemy import text
            session = self.db.get_session()
            session.execute(text("SELECT 1"))
            session.close()
            self.component_status["database"] = True
        except Exception:
            self.component_status["database"] = False

    def _check_memory(self):
        try:
            process = psutil.Process(os.getpid())
            mem_mb = process.memory_info().rss / (1024 * 1024)
            self.component_status["memory"] = mem_mb < MEMORY_LIMIT_MB
            if mem_mb > MEMORY_LIMIT_MB * 0.8:
                logger.warning("Memory usage high: %.1f MB / %d MB limit", mem_mb, MEMORY_LIMIT_MB)
        except Exception:
            self.component_status["memory"] = True

    def _check_disk(self):
        try:
            usage = psutil.disk_usage("/")
            free_pct = usage.free / usage.total
            self.component_status["disk"] = free_pct > 0.05
            if free_pct < 0.10:
                logger.warning("Disk space low: %.1f%% free", free_pct * 100)
        except Exception:
            self.component_status["disk"] = True

    async def _attempt_recovery(self, unhealthy: list):
        for component in unhealthy:
            if self.restart_count >= self.max_restarts:
                logger.critical("Max restart attempts reached, manual intervention required")
                if self.notifier:
                    await self.notifier.send_alert(
                        "CRITICAL: Max restart attempts reached. Bot needs manual intervention."
                    )
                return

            self.db.save_watchdog_event({
                "timestamp": utc_now(),
                "event_type": f"RECOVERY_{component.upper()}",
                "details": f"Attempting recovery for {component}",
                "resolved": False,
            })

            recovered = False
            if component == "exchange_api":
                recovered = await self._recover_exchange_api()
            elif component.startswith("websocket"):
                recovered = await self._recover_websocket(component)
            elif component == "database":
                recovered = self._recover_database()
            elif component == "memory":
                recovered = self._recover_memory()

            if recovered:
                self.component_status[component] = True
                self.db.save_watchdog_event({
                    "timestamp": utc_now(),
                    "event_type": f"RECOVERED_{component.upper()}",
                    "details": f"Successfully recovered {component}",
                    "resolved": True,
                })
                logger.info("Recovered component: %s", component)
            else:
                self.restart_count += 1
                logger.error("Failed to recover %s (attempt %d/%d)",
                             component, self.restart_count, self.max_restarts)
                if self.notifier:
                    await self.notifier.send_alert(
                        f"Recovery failed for {component} (attempt {self.restart_count}/{self.max_restarts})"
                    )

    async def _recover_exchange_api(self) -> bool:
        for attempt in range(3):
            try:
                await asyncio.sleep(5 * (attempt + 1))
                await self.exchange._sync_time()
                price = await self.exchange.get_price("BTCUSDT")
                if price > 0:
                    self.exchange.circuit_breaker.record_success()
                    return True
            except Exception as e:
                logger.warning("Exchange API recovery attempt %d failed: %s", attempt + 1, e)
        return False

    async def _recover_websocket(self, component: str) -> bool:
        ws_name = component.replace("websocket_", "")
        try:
            ws = self.exchange.ws_connections.get(ws_name)
            if ws and not ws.closed:
                await ws.close()
            self.exchange.ws_connections.pop(ws_name, None)

            callback = self.exchange._ws_callbacks.get(ws_name)
            if callback:
                if ws_name == "kline":
                    watchlist = self.db.load_state("watchlist_symbols")
                    if watchlist:
                        import json
                        symbols = json.loads(watchlist)
                        await self.exchange.start_kline_stream(symbols, "1h", callback)
                elif ws_name == "ticker":
                    await self.exchange.start_ticker_stream(callback)
                await asyncio.sleep(5)
                return self.exchange.is_ws_connected(ws_name)
        except Exception as e:
            logger.error("WebSocket recovery failed for %s: %s", ws_name, e)
        return False

    def _recover_database(self) -> bool:
        try:
            from sqlalchemy import text
            self.db.engine.dispose()
            session = self.db.get_session()
            session.execute(text("SELECT 1"))
            session.close()
            return True
        except Exception as e:
            logger.error("Database recovery failed: %s", e)
            return False

    def _recover_memory(self) -> bool:
        import gc
        gc.collect()
        try:
            process = psutil.Process(os.getpid())
            mem_mb = process.memory_info().rss / (1024 * 1024)
            return mem_mb < MEMORY_LIMIT_MB
        except Exception:
            return True

    def get_status(self) -> dict:
        try:
            process = psutil.Process(os.getpid())
            mem_mb = process.memory_info().rss / (1024 * 1024)
            cpu_pct = process.cpu_percent(interval=0.1)
        except Exception:
            mem_mb = 0
            cpu_pct = 0

        return {
            "is_healthy": self.is_healthy,
            "components": self.component_status.copy(),
            "restart_count": self.restart_count,
            "memory_mb": round(mem_mb, 1),
            "cpu_percent": round(cpu_pct, 1),
            "last_heartbeat": self.last_heartbeat,
            "uptime_seconds": time.time() - self.last_heartbeat,
        }

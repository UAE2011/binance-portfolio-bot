import asyncio
import time
import json
import hashlib
import hmac
from typing import Optional, Callable
from urllib.parse import urlencode

import aiohttp
import numpy as np

from src.utils import setup_logging, utc_now, round_step_size, round_tick_size

logger = setup_logging()


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.state = "CLOSED"
        self.last_failure_time = 0.0

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning("Circuit breaker OPENED after %d failures", self.failure_count)

    def record_success(self):
        self.failure_count = 0
        self.state = "CLOSED"

    def can_execute(self) -> bool:
        if self.state == "CLOSED":
            return True
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        return True


class BinanceExchange:
    def __init__(self, api_key: str, api_secret: str, testnet: bool = True):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet

        if testnet:
            self.base_url = "https://testnet.binance.vision"
            self.ws_url = "wss://testnet.binance.vision/ws"
            self.stream_url = "wss://testnet.binance.vision/stream"
        else:
            self.base_url = "https://api.binance.com"
            self.ws_url = "wss://stream.binance.com:9443/ws"
            self.stream_url = "wss://stream.binance.com:9443/stream"

        self.session: Optional[aiohttp.ClientSession] = None
        self.ws_connections: dict = {}
        self.exchange_info: dict = {}
        self.symbol_filters: dict = {}
        self.time_offset: int = 0
        self.circuit_breaker = CircuitBreaker()
        self.used_weight = 0
        self.listen_key: Optional[str] = None
        self._ws_callbacks: dict = {}
        self._reconnect_tasks: dict = {}

    async def initialize(self):
        self.session = aiohttp.ClientSession()
        await self._sync_time()
        await self._load_exchange_info()
        logger.info("Exchange initialized (testnet=%s)", self.testnet)

    async def close(self):
        for name, ws in self.ws_connections.items():
            if ws and not ws.closed:
                await ws.close()
        for task in self._reconnect_tasks.values():
            task.cancel()
        if self.session and not self.session.closed:
            await self.session.close()

    def _sign(self, params: dict) -> dict:
        params["timestamp"] = int(time.time() * 1000) + self.time_offset
        query = urlencode(params)
        signature = hmac.new(
            self.api_secret.encode(), query.encode(), hashlib.sha256
        ).hexdigest()
        params["signature"] = signature
        return params

    def _headers(self) -> dict:
        return {"X-MBX-APIKEY": self.api_key}

    async def _request(self, method: str, path: str, params: dict = None,
                       signed: bool = False) -> dict:
        if not self.circuit_breaker.can_execute():
            logger.warning("Circuit breaker OPEN, skipping request to %s", path)
            return {}

        if params is None:
            params = {}
        if signed:
            params = self._sign(params)

        url = f"{self.base_url}{path}"
        backoff = 1
        for attempt in range(4):
            try:
                # Binance requires ALL params (including POST/DELETE) as query string
                query_url = f"{url}?{urlencode(params)}" if params else url
                async with self.session.request(
                    method, query_url,
                    headers=self._headers(), timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    weight = resp.headers.get("X-MBX-USED-WEIGHT-1M", "0")
                    self.used_weight = int(weight)
                    if self.used_weight > 960:
                        logger.warning("API weight at %d/1200, pausing 30s", self.used_weight)
                        await asyncio.sleep(30)

                    if resp.status == 200:
                        self.circuit_breaker.record_success()
                        return await resp.json()
                    elif resp.status in (429, 418):
                        retry_after = int(resp.headers.get("Retry-After", backoff))
                        logger.warning("Rate limited (HTTP %d), waiting %ds", resp.status, retry_after)
                        await asyncio.sleep(retry_after)
                        backoff = min(backoff * 2, 60)
                    else:
                        body = await resp.text()
                        logger.error("API error %d on %s: %s", resp.status, path, body)
                        self.circuit_breaker.record_failure()
                        if attempt < 3:
                            await asyncio.sleep(backoff)
                            backoff = min(backoff * 2, 30)
                        else:
                            return {}
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.error("Request error on %s (attempt %d): %s", path, attempt + 1, e)
                self.circuit_breaker.record_failure()
                if attempt < 3:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 30)
        return {}

    async def _sync_time(self):
        try:
            data = await self._request("GET", "/api/v3/time")
            if data:
                server_time = data["serverTime"]
                local_time = int(time.time() * 1000)
                self.time_offset = server_time - local_time
                logger.info("Time synced, offset: %dms", self.time_offset)
        except Exception as e:
            logger.error("Time sync failed: %s", e)

    async def _load_exchange_info(self):
        data = await self._request("GET", "/api/v3/exchangeInfo")
        if not data:
            logger.error("Failed to load exchange info")
            return
        self.exchange_info = data
        for sym_info in data.get("symbols", []):
            symbol = sym_info["symbol"]
            filters = {}
            for f in sym_info.get("filters", []):
                if f["filterType"] == "LOT_SIZE":
                    filters["step_size"] = float(f["stepSize"])
                    filters["min_qty"] = float(f["minQty"])
                    filters["max_qty"] = float(f["maxQty"])
                elif f["filterType"] == "PRICE_FILTER":
                    filters["tick_size"] = float(f["tickSize"])
                    filters["min_price"] = float(f["minPrice"])
                elif f["filterType"] == "NOTIONAL" or f["filterType"] == "MIN_NOTIONAL":
                    filters["min_notional"] = float(f.get("minNotional", f.get("notional", "10")))
            filters["base_asset"] = sym_info.get("baseAsset", "")
            filters["quote_asset"] = sym_info.get("quoteAsset", "")
            filters["status"] = sym_info.get("status", "")
            self.symbol_filters[symbol] = filters
        logger.info("Loaded exchange info for %d symbols", len(self.symbol_filters))

    def adjust_quantity(self, symbol: str, quantity: float) -> float:
        f = self.symbol_filters.get(symbol, {})
        step = f.get("step_size", 0.00001)
        return round_step_size(quantity, step)

    def adjust_price(self, symbol: str, price: float) -> float:
        f = self.symbol_filters.get(symbol, {})
        tick = f.get("tick_size", 0.01)
        return round_tick_size(price, tick)

    def get_min_notional(self, symbol: str) -> float:
        return self.symbol_filters.get(symbol, {}).get("min_notional", 10.0)

    async def get_account(self) -> dict:
        return await self._request("GET", "/api/v3/account", signed=True)

    async def get_balances(self) -> dict:
        account = await self.get_account()
        balances = {}
        for b in account.get("balances", []):
            free = float(b["free"])
            locked = float(b["locked"])
            if free > 0 or locked > 0:
                balances[b["asset"]] = {"free": free, "locked": locked, "total": free + locked}
        return balances

    async def get_usdt_balance(self) -> float:
        balances = await self.get_balances()
        return balances.get("USDT", {}).get("free", 0.0)

    async def get_klines(self, symbol: str, interval: str = "1d",
                         limit: int = 200) -> list:
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        data = await self._request("GET", "/api/v3/klines", params)
        if not data:
            return []
        result = []
        for k in data:
            result.append({
                "open_time": k[0], "open": float(k[1]), "high": float(k[2]),
                "low": float(k[3]), "close": float(k[4]), "volume": float(k[5]),
                "close_time": k[6], "quote_volume": float(k[7]),
                "trades": k[8], "taker_buy_base": float(k[9]),
                "taker_buy_quote": float(k[10]),
            })
        return result

    async def get_ticker_24h(self, symbol: str = None) -> dict:
        params = {}
        if symbol:
            params["symbol"] = symbol
        return await self._request("GET", "/api/v3/ticker/24hr", params)

    async def get_all_tickers(self) -> list:
        data = await self._request("GET", "/api/v3/ticker/price")
        return data if isinstance(data, list) else []

    async def get_price(self, symbol: str) -> float:
        data = await self._request("GET", "/api/v3/ticker/price", {"symbol": symbol})
        return float(data.get("price", 0))

    async def place_market_buy(self, symbol: str, quote_qty: float) -> dict:
        params = {
            "symbol": symbol, "side": "BUY", "type": "MARKET",
            "quoteOrderQty": f"{quote_qty:.2f}",
        }
        result = await self._request("POST", "/api/v3/order", params, signed=True)
        if result:
            logger.info("MARKET BUY %s for %.2f USDT — orderId: %s",
                        symbol, quote_qty, result.get("orderId"))
        return result

    async def place_market_sell(self, symbol: str, quantity: float) -> dict:
        qty = self.adjust_quantity(symbol, quantity)
        params = {
            "symbol": symbol, "side": "SELL", "type": "MARKET",
            "quantity": f"{qty}",
        }
        result = await self._request("POST", "/api/v3/order", params, signed=True)
        if result:
            logger.info("MARKET SELL %s qty=%.8f — orderId: %s",
                        symbol, qty, result.get("orderId"))
        return result

    async def place_limit_buy(self, symbol: str, quantity: float, price: float) -> dict:
        qty = self.adjust_quantity(symbol, quantity)
        px = self.adjust_price(symbol, price)
        params = {
            "symbol": symbol, "side": "BUY", "type": "LIMIT",
            "quantity": f"{qty}", "price": f"{px}", "timeInForce": "GTC",
        }
        return await self._request("POST", "/api/v3/order", params, signed=True)

    async def place_oco_sell(self, symbol: str, quantity: float,
                             take_profit_price: float, stop_price: float,
                             stop_limit_price: float) -> dict:
        qty = self.adjust_quantity(symbol, quantity)
        tp = self.adjust_price(symbol, take_profit_price)
        sp = self.adjust_price(symbol, stop_price)
        slp = self.adjust_price(symbol, stop_limit_price)
        params = {
            "symbol": symbol, "side": "SELL",
            "quantity": f"{qty}", "price": f"{tp}",
            "stopPrice": f"{sp}", "stopLimitPrice": f"{slp}",
            "stopLimitTimeInForce": "GTC",
        }
        return await self._request("POST", "/api/v3/order/oco", params, signed=True)

    async def cancel_all_orders(self, symbol: str) -> dict:
        params = {"symbol": symbol}
        return await self._request("DELETE", "/api/v3/openOrders", params, signed=True)

    async def get_open_orders(self, symbol: str = None) -> list:
        params = {}
        if symbol:
            params["symbol"] = symbol
        data = await self._request("GET", "/api/v3/openOrders", params, signed=True)
        return data if isinstance(data, list) else []

    async def create_listen_key(self) -> str:
        data = await self._request("POST", "/api/v3/userDataStream")
        self.listen_key = data.get("listenKey", "")
        return self.listen_key

    async def keepalive_listen_key(self):
        if self.listen_key:
            await self._request("PUT", "/api/v3/userDataStream",
                                {"listenKey": self.listen_key})

    async def start_kline_stream(self, symbols: list, interval: str,
                                 callback: Callable):
        streams = [f"{s.lower()}@kline_{interval}" for s in symbols]
        await self._start_combined_stream("kline", streams, callback)

    async def start_ticker_stream(self, callback: Callable):
        await self._start_single_stream("ticker", "!ticker@arr", callback)

    async def start_user_stream(self, callback: Callable):
        listen_key = await self.create_listen_key()
        if listen_key:
            await self._start_single_stream("user", listen_key, callback)
            asyncio.create_task(self._keepalive_loop())

    async def _keepalive_loop(self):
        while True:
            await asyncio.sleep(1800)
            try:
                await self.keepalive_listen_key()
            except Exception as e:
                logger.error("Listen key keepalive failed: %s", e)

    async def _start_single_stream(self, name: str, stream: str,
                                   callback: Callable):
        url = f"{self.ws_url}/{stream}"
        self._ws_callbacks[name] = callback
        asyncio.create_task(self._ws_connect(name, url, callback))

    async def _start_combined_stream(self, name: str, streams: list,
                                     callback: Callable):
        stream_str = "/".join(streams)
        url = f"{self.stream_url}?streams={stream_str}"
        self._ws_callbacks[name] = callback
        asyncio.create_task(self._ws_connect(name, url, callback, combined=True))

    async def _ws_connect(self, name: str, url: str, callback: Callable,
                          combined: bool = False):
        backoff = 1
        while True:
            try:
                async with aiohttp.ClientSession() as ws_session:
                    async with ws_session.ws_connect(url, heartbeat=20) as ws:
                        self.ws_connections[name] = ws
                        backoff = 1
                        logger.info("WebSocket '%s' connected", name)
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                data = json.loads(msg.data)
                                if combined:
                                    data = data.get("data", data)
                                try:
                                    await callback(data)
                                except Exception as e:
                                    logger.error("WS callback error (%s): %s", name, e)
                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                logger.error("WS error (%s): %s", name, ws.exception())
                                break
                            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING):
                                break
            except Exception as e:
                logger.error("WebSocket '%s' connection error: %s", name, e)

            self.ws_connections.pop(name, None)
            logger.warning("WebSocket '%s' disconnected, reconnecting in %ds", name, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)

    def is_ws_connected(self, name: str) -> bool:
        ws = self.ws_connections.get(name)
        return ws is not None and not ws.closed

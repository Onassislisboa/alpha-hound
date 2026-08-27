"""Robinhood Crypto brokerage API.

Reference: https://docs.robinhood.com/crypto/trading

    base    https://trading.robinhood.com
    headers x-api-key, x-timestamp, x-signature
    message f"{api_key}{timestamp}{path}{METHOD}{body}"
    sig     detached Ed25519, base64
    key     base64 of the raw 32-byte Ed25519 seed

Four rules cause essentially every signature failure, so they are enforced
structurally here rather than left to the caller:

  1. `path` includes the query string.
  2. `method` is uppercase.
  3. No body means the body contributes an empty string.
  4. The signed bytes must be the transmitted bytes - so the payload is
     serialized exactly once and that string is both signed and sent.

Scope: this venue trades majors (BTC-USD, ETH-USD, SOL-USD). It is not where a
new token trades, so it is not part of the frontrunning strategy. It is here
because the account exists, the API is real, and having one venue with actual
depth is what lets the bot rotate to majors when the risk layer says stop
touching memecoins.
"""

from __future__ import annotations

import base64
import json
import time
import uuid
from typing import Any

from ..log import get
from ..models import Candidate, Chain, Fill, Quote, Side, VenueId
from ..net import Http, HttpError
from ..settings import Settings
from . import ExecutionError

log = get("robinhood")

BASE_URL = "https://trading.robinhood.com"


class RobinhoodVenue:
    id = VenueId.ROBINHOOD

    def __init__(self, http: Http, settings: Settings) -> None:
        self.http = http
        self.api_key = settings.rh_api_key
        self._seed = settings.rh_private_key
        self._key: Any = None

    def supports(self, chain: Chain) -> bool:
        return chain is Chain.ROBINHOOD_BROKER and bool(self.api_key and self._seed)

    # -- auth --------------------------------------------------------------
    def _private_key(self):
        if self._key is not None:
            return self._key
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        except ImportError as exc:  # pragma: no cover
            raise ExecutionError(
                "pip install 'alphahound[robinhood]' to use the Robinhood API"
            ) from exc
        try:
            seed = base64.b64decode(self._seed)
        except Exception as exc:  # noqa: BLE001
            raise ExecutionError("RH_PRIVATE_KEY must be base64 of the 32-byte seed") from exc
        if len(seed) != 32:
            raise ExecutionError(f"RH_PRIVATE_KEY decodes to {len(seed)} bytes, expected 32")
        self._key = Ed25519PrivateKey.from_private_bytes(seed)
        return self._key

    def _headers(self, method: str, path: str, body: str) -> dict[str, str]:
        timestamp = str(int(time.time()))
        message = f"{self.api_key}{timestamp}{path}{method.upper()}{body}"
        signature = base64.b64encode(self._private_key().sign(message.encode())).decode()
        headers = {
            "x-api-key": self.api_key,
            "x-timestamp": timestamp,
            "x-signature": signature,
        }
        if body:
            headers["content-type"] = "application/json"
        return headers

    async def _request(self, method: str, path: str, body: dict | None = None) -> Any:
        # Serialized once. Serializing twice yields two different strings often
        # enough to cost an afternoon.
        payload = json.dumps(body, separators=(",", ":")) if body is not None else ""
        try:
            return await self.http.request(
                method,
                f"{BASE_URL}{path}",
                content=payload or None,
                headers=self._headers(method, path, payload),
                retries=1,
            )
        except HttpError as exc:
            raise ExecutionError(f"robinhood {method} {path} -> {exc.status}: {exc.body[:200]}") from exc

    # -- market data -------------------------------------------------------
    async def best_bid_ask(self, symbol: str) -> tuple[float, float]:
        path = f"/api/v1/crypto/marketdata/best_bid_ask/?symbol={symbol}"
        data = await self._request("GET", path)
        results = (data or {}).get("results") or []
        if not results:
            raise ExecutionError(f"no quote for {symbol}")
        entry = results[0]
        bid = float(entry.get("bid_inclusive_of_sell_spread") or entry.get("price") or 0.0)
        ask = float(entry.get("ask_inclusive_of_buy_spread") or entry.get("price") or 0.0)
        if bid <= 0 or ask <= 0:
            raise ExecutionError(f"degenerate quote for {symbol}: bid={bid} ask={ask}")
        return bid, ask

    async def buying_power(self) -> float:
        data = await self._request("GET", "/api/v1/crypto/trading/accounts/")
        return float((data or {}).get("buying_power") or 0.0)

    # -- venue interface ---------------------------------------------------
    async def quote(self, candidate: Candidate, side: Side, amount: float) -> Quote:
        # For the brokerage the "address" is the trading pair symbol.
        symbol = candidate.address
        bid, ask = await self.best_bid_ask(symbol)
        if side is Side.BUY:
            qty = amount / ask
            return Quote(
                venue=self.id,
                in_amount=amount,
                out_amount=qty,
                price=ask,
                price_impact=max(0.0, (ask - bid) / bid / 2.0),
                raw={"symbol": symbol, "bid": bid, "ask": ask},
            )
        usd_out = amount * bid
        return Quote(
            venue=self.id,
            in_amount=amount,
            out_amount=usd_out,
            price=bid,
            price_impact=max(0.0, (ask - bid) / bid / 2.0),
            raw={"symbol": symbol, "bid": bid, "ask": ask},
        )

    async def execute(
        self, candidate: Candidate, side: Side, amount: float, quote: Quote
    ) -> Fill:
        symbol = quote.raw.get("symbol") or candidate.address
        quantity = quote.out_amount if side is Side.BUY else amount
        body = {
            "client_order_id": str(uuid.uuid4()),
            "side": side.value,
            "type": "market",
            "symbol": symbol,
            "market_order_config": {"asset_quantity": f"{quantity:.8f}".rstrip("0").rstrip(".")},
        }
        result = await self._request("POST", "/api/v1/crypto/trading/orders/", body)
        state = str((result or {}).get("state", "")).lower()
        if state in {"canceled", "rejected", "failed"}:
            raise ExecutionError(f"robinhood order {state}: {result}")

        filled_qty = float((result or {}).get("filled_asset_quantity") or quantity)
        avg_price = float((result or {}).get("average_price") or quote.price)
        amount_out = filled_qty if side is Side.BUY else filled_qty * avg_price
        slippage = 1.0 - amount_out / quote.out_amount if quote.out_amount > 0 else 0.0
        return Fill(
            venue=self.id,
            side=side,
            amount_in=amount,
            amount_out=amount_out,
            price=avg_price,
            fee_usd=0.0,
            tx_id=str((result or {}).get("id") or body["client_order_id"]),
            slippage_vs_quote=slippage,
        )

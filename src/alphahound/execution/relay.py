"""Intent relay for Fomo and Moby.

Checked 2026-08: neither Fomo (fomo.family) nor Moby / MobyAgent
(mobyscreener.com, built by AssetDash) publishes a trading API. Both are
mobile-only, self-custodial, gasless apps whose execution happens inside the
app. `docs.fomo.com` belongs to an unrelated social-proof product, not the
trading app.

There are three ways to handle that, and only one is honest:

  1. Fake an adapter with invented endpoints. It would look complete, fail on
     first contact, and the failure would arrive during a live trade.
  2. Drive the mobile app with UI automation. Fragile, against their terms, and
     it puts a scraper in the path of your money.
  3. Publish the decision as a signed intent and let a human or your own bridge
     act on it. Slower, real, and it costs nothing when the API does appear.

This is (3). The relay is not a venue - it never claims a fill - it is a sink
the engine notifies alongside whatever it actually executed. Point
RELAY_WEBHOOK_URL at a Telegram bot, a Shortcut, an n8n flow, or your own
service; when either platform ships an API, this file is where the adapter goes
and nothing upstream changes.
"""

from __future__ import annotations

import hashlib
import hmac
import json

from ..log import get
from ..models import Decision, now_ms
from ..net import Http, HttpError
from ..settings import Settings

log = get("relay")


class IntentRelay:
    def __init__(self, http: Http, settings: Settings) -> None:
        self.http = http
        self.url = settings.relay_webhook_url
        # The webhook is authenticated with the Jupiter key if present, purely
        # so a receiver can verify the payload came from this bot. Any shared
        # secret works; the point is that an unauthenticated buy instruction
        # arriving over the internet is an obvious way to get robbed.
        self._secret = (settings.jupiter_api_key or settings.rh_api_key or "").encode()

    @property
    def enabled(self) -> bool:
        return bool(self.url)

    async def publish(self, decision: Decision, size_usd: float, note: str = "") -> bool:
        if not self.enabled:
            return False
        candidate = decision.candidate
        payload = {
            "ts_ms": now_ms(),
            "action": decision.action.value,
            "chain": candidate.chain.value,
            "token": candidate.address,
            "symbol": candidate.symbol,
            "price_usd": candidate.price_usd,
            "liquidity_usd": candidate.liquidity_usd,
            "size_usd": round(size_usd, 2),
            "probability": round(decision.score.probability, 4),
            "expected_value": round(decision.score.expected_value, 4),
            "drivers": [
                {"feature": name, "contribution": round(value, 3)}
                for name, value in decision.score.top_drivers(6)
            ],
            "note": note,
            "venues_without_api": ["fomo", "moby"],
        }
        body = json.dumps(payload, separators=(",", ":"))
        headers = {"content-type": "application/json"}
        if self._secret:
            headers["x-alphahound-signature"] = hmac.new(
                self._secret, body.encode(), hashlib.sha256
            ).hexdigest()
        try:
            await self.http.post(self.url, content=body, headers=headers, retries=1)
        except HttpError as exc:
            log.warning("relay publish failed", extra={"status": exc.status})
            return False
        log.info(
            "intent relayed",
            extra={"token": candidate.symbol or candidate.address, "size_usd": size_usd},
        )
        return True

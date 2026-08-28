"""Candidate discovery.

Being early is a latency problem before it is an analysis problem: a token you
hear about ten minutes late is a token whose retail wave you are joining rather
than preceding. So discovery is ordered by how fast each source is, not how
convenient:

    pump.fun websocket   sub-second, push
    dexscreener profiles ~seconds, poll
    dexscreener boosts   ~seconds, poll, and a signal about the promoter
    watchlist            whenever you say so

Everything is deduplicated and thrown away after `max_candidate_age_minutes`,
because an old candidate is not a candidate, it is a chart.

Honest ceiling: a public REST/websocket feed puts you in the same cohort as
every other bot on the same feeds. The latency ladder above this is a Geyser /
Yellowstone gRPC stream, and above that co-location with a validator. If you
are competing for the first block of a launch, you need those. This layer
targets the 1-10 minute window instead, where analysis quality still decides
the outcome.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass, field

from .log import get
from .models import Candidate, Chain, now_ms
from .origin import launchpad_origin
from .playbook import max_age_minutes as pb_max_age
from .providers import DEX_CHAIN_SLUG, Dexscreener
from .settings import Config, Settings

log = get("discovery")


@dataclass(slots=True)
class DiscoveryStats:
    seen: int = 0
    emitted: int = 0
    dropped_stale: int = 0
    dropped_duplicate: int = 0
    by_source: dict[str, int] = field(default_factory=dict)


class Discovery:
    def __init__(
        self,
        settings: Settings,
        strategy: Config,
        dexscreener: Dexscreener,
    ) -> None:
        self.settings = settings
        self.strategy = strategy
        self.dex = dexscreener
        self.max_age_minutes = float(strategy.get("loop.max_candidate_age_minutes", 180))
        self.stats = DiscoveryStats()

        self._seen: dict[str, int] = {}
        self._queue: asyncio.Queue[Candidate] = asyncio.Queue(maxsize=2000)
        self._tasks: list[asyncio.Task] = []
        self._watchlist: set[str] = set()

    # -- lifecycle ---------------------------------------------------------
    async def start(self) -> None:
        if self.settings.pumpportal_ws_url and Chain.SOLANA in self.settings.enabled_chains:
            self._tasks.append(asyncio.create_task(self._pump_stream(), name="pump-stream"))

    async def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()

    def watch(self, chain: Chain, address: str) -> None:
        self._watchlist.add(f"{chain.value}:{address}")

    # -- polling -----------------------------------------------------------
    async def poll(self) -> list[Candidate]:
        """One round of the polled sources, merged and deduplicated."""
        addresses: list[str] = []
        sources: dict[str, str] = {}

        for source, coro in (
            ("dexscreener_profiles", self.dex.new_profiles()),
            ("dexscreener_boosts", self.dex.boosted()),
        ):
            try:
                found = await coro
            except Exception as exc:  # noqa: BLE001
                log.warning("source failed", extra={"source": source, "error": str(exc)})
                continue
            for address in found:
                if address not in sources:
                    sources[address] = source
                    addresses.append(address)

        for entry in self._watchlist:
            _, _, address = entry.partition(":")
            if address and address not in sources:
                sources[address] = "watchlist"
                addresses.append(address)

        out: list[Candidate] = []
        # token_pairs takes 30 addresses per call, so this is len/30 requests
        # rather than len.
        for chunk_start in range(0, len(addresses), 30):
            chunk = addresses[chunk_start : chunk_start + 30]
            try:
                snaps = await self.dex.token_pairs(chunk)
            except Exception as exc:  # noqa: BLE001
                log.warning("token_pairs failed", extra={"error": str(exc)})
                continue
            for snap in snaps:
                if snap.chain not in self.settings.enabled_chains:
                    continue
                candidate = snap.to_candidate(sources.get(snap.token_address, "dexscreener"))
                if self._accept(candidate):
                    out.append(candidate)

        while not self._queue.empty():
            candidate = self._queue.get_nowait()
            if self._accept(candidate):
                out.append(candidate)
        return out

    def _accept(self, candidate: Candidate) -> bool:
        self.stats.seen += 1
        if candidate.chain not in self.settings.enabled_chains:
            return False
        if not candidate.address:
            return False
        allowed, _reason = launchpad_origin(candidate, self.strategy)
        if not allowed:
            return False

        age = candidate.age_minutes
        if not candidate.created_at_ms or age > pb_max_age(self.strategy, candidate.chain):
            self.stats.dropped_stale += 1
            return False

        last = self._seen.get(candidate.key, 0)
        # Re-emit a known candidate at most once a minute: the engine wants
        # fresh feature vectors on tokens it is watching, not a stampede of
        # duplicates from three sources.
        if now_ms() - last < 60_000:
            self.stats.dropped_duplicate += 1
            return False
        self._seen[candidate.key] = now_ms()

        self.stats.emitted += 1
        self.stats.by_source[candidate.source] = self.stats.by_source.get(candidate.source, 0) + 1
        return True

    def prune(self) -> None:
        cutoff = now_ms() - int(self.max_age_minutes * 60_000) * 2
        for key, ts in list(self._seen.items()):
            if ts < cutoff:
                del self._seen[key]

    # -- streaming ---------------------------------------------------------
    async def _pump_stream(self) -> None:
        """pump.fun new-token firehose via PumpPortal.

        Reconnects forever with backoff. A discovery source that dies quietly is
        worse than one that never existed, because the bot keeps running and you
        conclude the market went quiet.
        """
        try:
            import websockets
        except ImportError:
            log.warning("pip install 'alphahound[stream]' to enable the pump.fun stream")
            return

        url = self.settings.pumpportal_ws_url
        backoff = 1.0
        while True:
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    await ws.send(json.dumps({"method": "subscribeNewToken"}))
                    log.info("pump stream connected")
                    backoff = 1.0
                    async for raw in ws:
                        candidate = self._parse_pump_event(raw)
                        if candidate is None:
                            continue
                        try:
                            self._queue.put_nowait(candidate)
                        except asyncio.QueueFull:
                            # Backpressure: dropping the newest is wrong, but so
                            # is blocking the socket. The engine is the
                            # bottleneck here, and it will see the token again
                            # via polling.
                            pass
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                log.warning("pump stream dropped", extra={"error": str(exc), "retry_in": backoff})
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    @staticmethod
    def _parse_pump_event(raw: str | bytes) -> Candidate | None:
        try:
            event = json.loads(raw)
        except (ValueError, TypeError):
            return None
        mint = event.get("mint") or event.get("ca")
        if not mint:
            return None
        return Candidate(
            chain=Chain.SOLANA,
            address=mint,
            symbol=event.get("symbol", ""),
            name=event.get("name", ""),
            created_at_ms=now_ms(),
            pool_address=event.get("bondingCurveKey", "") or event.get("pool", ""),
            deployer=event.get("traderPublicKey", "") or event.get("creator", ""),
            source="pumpfun_stream",
        )


def chain_supported_by_dexscreener(chain: Chain) -> bool:
    return chain in DEX_CHAIN_SLUG

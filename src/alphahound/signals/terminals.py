"""Who is buying: terminal and bot attribution.

Every Solana trading terminal builds transactions with its own signature - its
router program, and a fee account it pays on every swap. Match a buyer's
transaction against a registry of those and you learn something no price chart
contains: whether the buy pressure is humans in Axiom/Photon/BullX/Moby, copy
bots in Trojan/Maestro, or launch snipers.

Why this is the core of the strategy
------------------------------------
Retail terminal share has opposite sign at the level and at the derivative:

  * HIGH level  -> the crowd has already arrived. You are late, and the people
                   you would be buying from are the people who were early.
  * RISING fast -> the wave is starting. This is the only moment where being
                   fast pays, because the flow that will move the price has
                   committed but has not finished arriving.

Bot share is negative in both. A high sniper share is not confirmation, it is
adverse selection: they entered below you and their exit is your entry.

This is what "frontrunning human traders" means here - beating the retail
adoption curve using public data. It is NOT sandwiching pending transactions,
which is a different activity with different ethics and, for a solo operator
without validator access, negative expected value anyway.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

RETAIL = "retail"
BOT = "bot"
NEUTRAL = "neutral"
UNKNOWN = "unknown"


@dataclass(slots=True)
class BuyerTx:
    """One buy, reduced to what attribution needs."""

    buyer: str
    account_keys: list[str]
    ts_ms: int
    size_usd: float = 0.0


@dataclass(slots=True)
class Attribution:
    total: int = 0
    # label -> share of buys, e.g. {"axiom": 0.31, "photon": 0.12}
    by_label: dict[str, float] = field(default_factory=dict)
    # class -> share of buys
    by_class: dict[str, float] = field(default_factory=dict)
    # Same, weighted by USD size. A terminal with 5% of buys and 60% of volume
    # is telling you something the count hides.
    by_class_usd: dict[str, float] = field(default_factory=dict)

    def label_share(self, label: str) -> float:
        return self.by_label.get(label, 0.0)

    def class_share(self, klass: str) -> float:
        return self.by_class.get(klass, 0.0)


class TerminalRegistry:
    """Address -> (label, class) lookup, from config plus learned labels."""

    def __init__(
        self,
        terminals: list[dict],
        venue_programs: dict[str, str] | None = None,
        learned: dict[str, tuple[str, str]] | None = None,
    ) -> None:
        self._map: dict[str, tuple[str, str]] = {}
        for entry in terminals:
            label = str(entry.get("name", "")).strip()
            klass = str(entry.get("class", UNKNOWN)).strip() or UNKNOWN
            for address in entry.get("fee_accounts", []) or []:
                if address:
                    self._map[address] = (label, klass)
        for label, address in (venue_programs or {}).items():
            # Only fill in as neutral if a terminal has not already claimed it.
            self._map.setdefault(address, (label, NEUTRAL))
        for address, (label, klass) in (learned or {}).items():
            self._map[address] = (label, klass)

        self._known_labels = {label for label, _ in self._map.values()}
        # Only retail/bot labels make the attribution features informative.
        # Counting the AMM programs here would report the registry as populated
        # while every buyer still resolves to "neutral" - the failure mode this
        # distinction exists to prevent.
        self._attributable = {
            label for label, klass in self._map.values() if klass in (RETAIL, BOT)
        }

    @property
    def detectable_labels(self) -> set[str]:
        return set(self._known_labels)

    @property
    def attributable_labels(self) -> set[str]:
        return set(self._attributable)

    def lookup(self, address: str) -> tuple[str, str] | None:
        return self._map.get(address)

    def classify(self, account_keys: list[str]) -> tuple[str, str]:
        """First match wins, with retail/bot taking precedence over neutral.

        A swap routed through a terminal also touches Jupiter or Raydium, so
        matching greedily on the first hit would label every terminal buy as
        "jupiter" and flatten the only signal that matters.
        """
        fallback: tuple[str, str] | None = None
        for key in account_keys:
            hit = self._map.get(key)
            if hit is None:
                continue
            if hit[1] in (RETAIL, BOT):
                return hit
            fallback = fallback or hit
        return fallback or (UNKNOWN, UNKNOWN)


def attribute(txs: list[BuyerTx], registry: TerminalRegistry) -> Attribution:
    result = Attribution(total=len(txs))
    if not txs:
        return result

    label_counts: Counter[str] = Counter()
    class_counts: Counter[str] = Counter()
    class_usd: dict[str, float] = defaultdict(float)
    total_usd = 0.0

    # One wallet spamming twenty small buys is one participant, so attribution
    # is per unique buyer. Without this dedup a single bot can manufacture an
    # entire "retail wave".
    seen: dict[str, tuple[str, str]] = {}
    for tx in txs:
        label, klass = registry.classify(tx.account_keys)
        seen.setdefault(tx.buyer, (label, klass))
        class_usd[klass] += tx.size_usd
        total_usd += tx.size_usd

    for label, klass in seen.values():
        label_counts[label] += 1
        class_counts[klass] += 1

    n = len(seen) or 1
    result.total = n
    result.by_label = {k: v / n for k, v in label_counts.items()}
    result.by_class = {k: v / n for k, v in class_counts.items()}
    if total_usd > 0:
        result.by_class_usd = {k: v / total_usd for k, v in class_usd.items()}
    return result


class ShareTracker:
    """Keeps a short history of attribution shares per token so the engine can
    read the derivative, which is the half of the signal that actually times
    the entry."""

    def __init__(self, max_samples: int = 40) -> None:
        self._history: dict[str, list[tuple[int, dict[str, float]]]] = defaultdict(list)
        self._max = max_samples

    def observe(self, key: str, ts_ms: int, shares: dict[str, float]) -> None:
        series = self._history[key]
        series.append((ts_ms, dict(shares)))
        if len(series) > self._max:
            del series[: len(series) - self._max]

    def delta(self, key: str, name: str, window_ms: int) -> float:
        """Change in a share over the window. Returns 0.0 with a single sample,
        which correctly reads as "no evidence of a wave" rather than as a
        wave."""
        series = self._history.get(key) or []
        if len(series) < 2:
            return 0.0
        latest_ts, latest = series[-1]
        cutoff = latest_ts - window_ms
        base = next((s for s in reversed(series[:-1]) if s[0] <= cutoff), series[0])
        return latest.get(name, 0.0) - base[1].get(name, 0.0)

    def samples(self, key: str) -> int:
        return len(self._history.get(key) or [])

    def forget(self, key: str) -> None:
        self._history.pop(key, None)


# ---------------------------------------------------------------------------
# Discovery: find the fee accounts instead of guessing them.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class FeeAccountCandidate:
    address: str
    distinct_txs: int
    distinct_buyers: int
    score: float


def discover_fee_accounts(
    txs: list[BuyerTx],
    registry: TerminalRegistry,
    *,
    ignore: set[str] | None = None,
    min_distinct_buyers: int = 8,
    limit: int = 40,
) -> list[FeeAccountCandidate]:
    """Rank unlabeled addresses that recur across many independent buyers.

    A terminal's fee account is, by construction, the address that shows up in
    thousands of unrelated wallets' swap transactions. Nothing else has that
    shape except the AMM programs, which are already labeled. Ranking by
    distinct BUYERS rather than distinct transactions is what keeps a single
    busy bot from topping the list.

    The output is a shortlist for a human to label once - deliberately not
    auto-applied, because a mislabeled fee account silently corrupts the single
    most important feature in the model.
    """
    ignore = ignore or set()
    tx_counts: Counter[str] = Counter()
    buyers: dict[str, set[str]] = defaultdict(set)

    for tx in txs:
        for key in set(tx.account_keys):
            if key in ignore or key == tx.buyer or registry.lookup(key) is not None:
                continue
            tx_counts[key] += 1
            buyers[key].add(tx.buyer)

    out: list[FeeAccountCandidate] = []
    for address, count in tx_counts.items():
        n_buyers = len(buyers[address])
        if n_buyers < min_distinct_buyers:
            continue
        # Reward breadth across buyers, discount addresses that only ever show
        # up alongside one wallet.
        out.append(
            FeeAccountCandidate(
                address=address,
                distinct_txs=count,
                distinct_buyers=n_buyers,
                score=n_buyers * (n_buyers / count if count else 0.0),
            )
        )
    out.sort(key=lambda c: -c.score)
    return out[:limit]


def extract(
    attribution: Attribution,
    tracker: ShareTracker,
    key: str,
    ts_ms: int,
    window_ms: int,
) -> dict[str, float]:
    """Assemble the terminal slice of the feature vector."""
    shares = {
        RETAIL: attribution.class_share(RETAIL),
        BOT: attribution.class_share(BOT),
        UNKNOWN: attribution.class_share(UNKNOWN),
        "axiom": attribution.label_share("axiom"),
    }
    tracker.observe(key, ts_ms, shares)
    return {
        "retail_share": shares[RETAIL],
        "retail_share_delta_5m": tracker.delta(key, RETAIL, window_ms),
        "bot_share": shares[BOT],
        "axiom_share": shares["axiom"],
        "axiom_share_delta_5m": tracker.delta(key, "axiom", window_ms),
        "unknown_share": shares[UNKNOWN],
    }

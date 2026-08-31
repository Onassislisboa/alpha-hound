"""Main runner, beta plays, vamps.

A main runner births a pack of name-alikes. Betas ride it up and vanish when
it dumps. A vamp is the same ticker launched minutes or hours later — skip it.

# ponytail: stem + name-token match only. MAGA-as-TRUMP-beta needs a narrative
# feed; upgrade if a name graph exists.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from ..models import Candidate

_STOP = frozenset(
    {
        "COIN",
        "TOKEN",
        "INU",
        "CAT",
        "DOG",
        "SOL",
        "ETH",
        "BNB",
        "MEME",
        "THE",
        "OFFICIAL",
        "CTO",
        "AI",
        "DAO",
    }
)


def ticker_stem(symbol: str, name: str = "") -> str:
    raw = "".join(c for c in (symbol or "").upper() if c.isalnum())
    while raw and raw[-1].isdigit():
        raw = raw[:-1]
    if len(raw) >= 3:
        return raw
    n = "".join(c for c in (name or "").upper() if c.isalnum())
    while n and n[-1].isdigit():
        n = n[:-1]
    return n[:16] if len(n) >= 4 else raw


def name_keys(symbol: str, name: str) -> set[str]:
    keys: set[str] = set()
    stem = ticker_stem(symbol, name)
    if len(stem) >= 3:
        keys.add(stem)
    blob = "".join(ch if ch.isalnum() else " " for ch in (name or "").upper())
    for word in blob.split():
        if len(word) >= 4 and word not in _STOP:
            keys.add(word)
    return keys


@dataclass(slots=True)
class PackTag:
    role: str  # main | beta | vamp | solo
    stem: str
    main_key: str = ""
    main_ret_5m: float = 0.0
    pack_size: int = 1


def tag_pack(candidates: list[Candidate]) -> dict[str, PackTag]:
    """Tag every candidate. Solo if it has no family in the current set."""
    out: dict[str, PackTag] = {}
    if not candidates:
        return out

    by_stem: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for c in candidates:
        stem = ticker_stem(c.symbol, c.name)
        if len(stem) >= 3:
            by_stem[(c.chain.value, stem)].append(c)

    claimed: set[str] = set()
    for (chain, stem), group in by_stem.items():
        if len(group) < 2:
            continue
        group.sort(key=lambda c: (c.created_at_ms or 10**18, -c.mcap_usd, -c.volume_5m_usd))
        original = group[0]
        main = max(group, key=lambda c: (c.mcap_usd, c.volume_5m_usd, -c.created_at_ms))
        for c in group:
            if c.key == main.key:
                role = "main"
            elif c.created_at_ms and original.created_at_ms and c.created_at_ms > original.created_at_ms:
                orig_stem = ticker_stem(original.symbol, original.name)
                mine = ticker_stem(c.symbol, c.name)
                role = "vamp" if mine == orig_stem else "beta"
            else:
                role = "beta"
            out[c.key] = PackTag(
                role=role,
                stem=stem,
                main_key=main.key,
                main_ret_5m=main.ret_5m,
                pack_size=len(group),
            )
            claimed.add(c.key)

    # Name-family betas: share a 4+ letter token with a bigger, older coin.
    by_word: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
    for c in candidates:
        for word in name_keys(c.symbol, c.name):
            by_word[(c.chain.value, word)].append(c)
    for (_chain, word), group in by_word.items():
        uniq: dict[str, Candidate] = {c.key: c for c in group}
        if len(uniq) < 2:
            continue
        members = list(uniq.values())
        main = max(members, key=lambda c: (c.mcap_usd, c.volume_5m_usd, -c.created_at_ms))
        for c in members:
            if c.key in claimed:
                continue
            if c.key == main.key:
                continue
            if main.mcap_usd < 8_000 and main.volume_5m_usd < 3_000:
                continue
            out[c.key] = PackTag(
                role="beta",
                stem=word,
                main_key=main.key,
                main_ret_5m=main.ret_5m,
                pack_size=len(members),
            )
            claimed.add(c.key)
            if main.key not in claimed:
                out[main.key] = PackTag(
                    role="main",
                    stem=word,
                    main_key=main.key,
                    main_ret_5m=main.ret_5m,
                    pack_size=len(members),
                )
                claimed.add(main.key)

    for c in candidates:
        if c.key not in out:
            out[c.key] = PackTag(role="solo", stem=ticker_stem(c.symbol, c.name))
    return out


def apply_tags(candidates: list[Candidate]) -> dict[str, PackTag]:
    tags = tag_pack(candidates)
    for c in candidates:
        tag = tags.get(c.key) or PackTag(role="solo", stem="")
        c.pack_role = tag.role
        c.pack_stem = tag.stem
        c.main_key = tag.main_key
        c.main_ret_5m = tag.main_ret_5m
        c.pack_size = tag.pack_size
    return tags


def dump_beta_keys(tags: dict[str, PackTag], *, threshold: float = -0.20) -> set[str]:
    """Betas whose main is already dumping — they will be gone in minutes."""
    return {
        key
        for key, tag in tags.items()
        if tag.role == "beta" and tag.main_ret_5m <= threshold
    }

# FirstKill

Paper bot for **new** launchpad memecoins. Package name stays `alphahound`.

A multi-chain trading bot that hunts tokens **before** the retail wave arrives, sizes
every trade by expected value net of real costs, and rewrites its own scoring
model from the money it loses.

Solana, BNB Chain, Base, Robinhood Chain, and the Robinhood Crypto brokerage.
Paper mode by default; nothing here spends real money until you tell it to.

```bash
python -m venv .venv && .venv/Scripts/activate     # Linux/macOS: source .venv/bin/activate
pip install -e ".[all]"
cp .env.example .env
python -m alphahound doctor
python -m alphahound run --paper
python -m alphahound preview --port 8765
# open http://127.0.0.1:8765/
```

Paper is on by default. The preview shows what it is allowed to trade (launchpad,
age, copy window) and what is **in view**. Holds stay empty until a mint clears
the gates **and** scores ≥ 58% win / ≥ 4% EV. On the public Solana RPC that
almost never happens: holder counts, terminal attribution and sell quotes come
back unmeasured, and unmeasured features contribute zero.

### Missing for a paper fill you can watch

1. Paid Solana RPC in `SOLANA_RPC_URL` (Helius / Triton / QuickNode). Public RPC rate-limits the enricher.
2. `HELIUS_API_KEY` — exact holder counts. Without it the holder gate cannot pass in live, and paper scores blind.
3. `python -m alphahound discover-terminals` then `label-terminal` at least one retail fee account. Until then `retail_share` is inert and the thesis cannot fire.
4. Optional: `BIRDEYE_API_KEY`, `TWITTER_BEARER_TOKEN`, `BUBBLEMAPS_API_KEY`, `COPE_API_KEY`, wallets in `config/whales.toml`.

### Missing for live (Solana first)

1. `JUPITER_API_KEY` from [portal.jup.ag](https://portal.jup.ag).
2. Fund the Solana hot wallet with a small amount of SOL. It is a hot wallet.
3. `MODE=live`. Keep `ENABLED_CHAINS=solana` until paper has ~40 closed trades.
4. BNB live also needs `ZEROEX_API_KEY` and BNB on that wallet. Robinhood Chain live needs ETH on chain 4663 for gas.

No extra bot framework. Swaps go through **Jupiter** (Solana), **0x** (BNB), **Uniswap V3** (Robinhood Chain). Fomo / Moby / Padre are research-only.

---

## What "frontrunning human traders" means here

This is the most important paragraph in the repository, because the phrase has
two meanings and only one of them is worth building.

**Not this:** watching the mempool for someone else's pending buy and jumping
in front of it. On Solana that means Jito bundle auctions and validator
relationships; on EVM it means competing with searchers who have better
infrastructure than you. For a solo operator it is an arms race with negative
expected value, and it makes money by taking it from the specific person whose
transaction you saw.

**This:** being early in a token's *retail adoption curve*, using only public
data. Concretely, and this is the entire thesis:

> Retail terminal inflow (Axiom, Photon, BullX, GMGN, Moby, Fomo) is
> **accelerating** while its absolute share of buyers is still **low**.

Those two conditions have opposite signs, deliberately:

| Signal | Meaning | Prior weight |
|---|---|---|
| `retail_share` **high** | the crowd already arrived; you would be their exit liquidity | **−0.80** |
| `retail_share_delta_5m` **rising** | the crowd is arriving now; the flow that moves price has committed but not finished | **+1.40** |
| `bot_share` high | snipers and copy bots, entered below you, will sell into you | **−1.00** |

You are competing on *information quality and reaction speed against the crowd*,
not on transaction ordering against an individual. `src/alphahound/signals/terminals.py`
implements it and explains the mechanics.

## What is real and what is not

Verified against live documentation, August 2026. No invented endpoints.

| Integration | Status |
|---|---|
| **Jupiter** Swap v2 (`/order` + `/execute`) | Real. Needs `JUPITER_API_KEY`. |
| **0x** Swap v2 allowance-holder — BNB (56), Base (8453) | Real. Needs `ZEROEX_API_KEY`. |
| **Uniswap V3** SwapRouter02 — Robinhood Chain (4663) | Real. On-chain; 0x does not cover this chain. |
| **Robinhood Crypto** brokerage | Real. Ed25519-signed. Majors only, US only. |
| **Dexscreener** | Real, keyless. Discovery and pricing backbone. |
| **pump.fun** new-token stream via PumpPortal | Real websocket. |
| **Helius / Birdeye** | Real, optional. Exact holder counts and candles. |
| **Paper** | Real simulation: constant-product impact + latency + fees on both legs. |
| **Fomo** (fomo.family) | Research only. Labeled wallets + optional Cope key. Never executes. |
| **MobyScreener** | No public API. Whale % / buy-vs-sell computed on-chain from labeled + large holders. |

Fomo and Moby are **not** execution venues. Paste wallets into `config/whales.toml`
(and optionally set `COPE_API_KEY`) so the bot can see which profiles sit in a
**new** launch, what % they hold, and whether they are buying or selling.
Trades go through Jupiter / 0x / Uniswap V3 / paper.

## The signals

Cheap features first — nothing touches the chain until the free snapshot has had
a chance to disqualify the candidate.

**Chart** (`signals/chart.py`) — deliberately simple, because a 40-minute-old
token has 40 candles and any longer lookback is fitting noise. Momentum, VWAP
deviation, ATR, breakout-as-a-ramp, volume z-score, candle body ratio, and a
`parabolic` flag that is strongly negative: the chart that convinces you is the
chart that already paid someone else.

**Distribution** (`signals/distribution.py`) — top-1 and top-10 share with LP and
burn addresses excluded, Gini (because top-10 hides the shape: ten wallets at 4%
and one at 40% give the same headline and completely different risk), fresh-wallet
percentage, launch-bundle percentage, dev holdings, and a one-hop funding-graph
cluster check — twenty wallets funded by one address are one holder in twenty
hats.

**Terminal attribution** (`signals/terminals.py`) — the thesis above. Per-buyer,
deduplicated, level *and* derivative.

**Flow** (`signals/flow.py`) — USD-weighted buy/sell ratio (count ratios are the
cheapest number in crypto to fake), unique buyer velocity, whale concentration,
smart-money buys, and sniper exit pressure.

**Cost and sellability** (`execution/__init__.py::Router.round_trip`) — quote a
buy, then quote selling exactly what that buy would return. One request pair
that answers three questions: entry cost, exit tax, and *can this be sold at
all*. A honeypot fails here, before you own it.

**Deliberately absent: deployer rug rate.** Scoring a deployer's track record
needs an index of every token that address ever launched, which no free provider
exposes. A feature that is always defaulted to `0.0` is not a missing feature, it
is a *bonus* handed to every unknown deployer — so it is gone rather than
decorative. What survives is the part that needs only the deployer's identity:
the per-deployer exposure cap in `risk.py`.

### Terminal fee accounts are discovered, not hardcoded

`config/terminals.toml` ships the public program IDs (verified) and **empty**
`fee_accounts` lists for each terminal. Publishing a guessed Axiom address would
mislabel every buyer and silently poison the most important feature in the
model. So mine them instead:

```bash
alphahound discover-terminals --tokens 8
```

It ranks unlabeled addresses that recur across many *independent buyers* — the
shape only a fee account has — and writes a shortlist to
`state/terminal_candidates.json`. Check each in an explorer, then:

```bash
alphahound label-terminal <address> axiom retail
alphahound label-terminal <address> trojan bot
```

Until at least one retail/bot label exists, `retail_share` and `axiom_share`
are reported as **unmeasured**, contribute zero, and `doctor` says so.

## Gates, score, expected value — in that order

1. **Gates** are absolute. Rug, honeypot, no exit liquidity, live mint
   authority. These are total losses, not large losses, so no score overrides
   them. A gate whose input could not be measured *abstains*, and in live mode
   an abstention is a veto — a gate you cannot evaluate is a risk you cannot
   see. An unmeasured feature normalizes to neutral, never to a favourable
   value.

2. **Score** is a logistic model over ~32 normalized features. Weights are
   stored by name, so adding a feature is backward compatible. Every decision
   records its per-feature contribution to the logit — without that you cannot
   tell a good model from a lucky one.

3. **Expected value** is the part most bots skip:

   ```
   EV = p·avg_win − (1−p)·avg_loss − round_trip_cost
   ```

   `avg_win` and `avg_loss` come from *your realized history* once there is
   enough of it, not from your stop settings — because stops do not fill where
   you set them. A 60%-win-rate strategy paying 6% round-trip on a 7% average
   win is a machine for converting conviction into fees.

   Before there is history, the prior weights the take-profit rungs by how
   often a winner is assumed to reach them
   (`scoring.prior_winner_ladder_reach`) and runs each case through the real
   exit policy. Pricing the whole ladder instead — the obvious
   `sum((mult−1)·frac)` — assumes every winner does 3.5x, roughly triples prior
   EV, and makes a bot fearless on the day it knows least.

## Risk

Four independent, deliberately redundant brakes:

- **Fractional Kelly** sizing (`f* = (pW − qL)/(WL)`, scaled to 0.25) capped by
  max position percentage, **capped again by pool liquidity** — on a thin pool
  that binds long before the equity cap, and it is the difference between a
  position you can exit and a position you *are*.
- **Daily loss limit** trips the kill switch, which **closes open positions**
  rather than merely blocking new ones.
- **Escalating cooldown** after consecutive losses. Not superstition: N losses
  in a row is the cheapest evidence that the model's current view is wrong.
- **Per-deployer exposure cap** — two tokens from one deployer are one bet.
- Equity is recomputed as starting capital + realized PnL, so a drawdown
  shrinks position sizes instead of compounding into a hole.

Exits (`portfolio.py`), priority-ordered: liquidity drain → hard stop → take-profit
ladder → trailing stop → time stop. Liquidity drain wins because it is the only
condition where waiting one tick can mean not exiting at all.

## Self-refinement

Four loops in `learning.py`, each closing a different gap between what the bot
believed and what happened.

**1. Postmortem.** Every loss gets a class that implies an action. A taxonomy
whose buckets do not map to a parameter change is a dashboard, not a learning
system.

| Class | Detected from | Response |
|---|---|---|
| `late_entry` | fill price vs signal price | abort on drift sooner |
| `slippage_blowout` | realized vs quoted | raise min liquidity, cut size-vs-liquidity |
| `rug` | liquidity-drain exit | raise min liquidity, tighten concentration |
| `exit_too_fast` | peak ≫ realized, on a win | widen the trail |
| `exit_too_slow` | reachable gain given back | tighten the trail |
| `adverse_selection` | high bundle/bot share | tighten the bundle gate |
| `execution_fail` | submit failed | pay more for inclusion |
| `no_edge` | nothing else explains it | demand more EV |

`no_edge` is the honest bucket. If it is not your largest loss class, your
taxonomy is lying to you.

**2. Bounded nudges.** One tunable per dominant class, only when it is both
frequent (≥22% of the last 60 trades) *and* actually losing money. Every change
writes a row to `param_history` with its reason, and **no parameter may drift
more than 3× from the value in your config** — the bot adapts, it does not
redesign the strategy behind your back.

```bash
alphahound params --history 20
```

**3. Weight training.** Online logistic regression on realized outcomes,
weighted by PnL magnitude. Every decision and trade persists **which features
were unmeasurable at the time**, and the trainer neutralizes those rather than
reading the stored `0.0` as an observation — for most of the normalizers here a
zero is a real and non-neutral value, so training on it teaches the model that a
token nobody could measure had no whales, no bundle, and no fresh wallets. On top
of that:

- the hand-set priors as an **L2 anchor**, so twelve lucky trades cannot rewrite the model
- a **minimum observation count** per feature before its weight can move at all
- a **time-ordered holdout** (a random split leaks the future into the past)
- **champion/challenger**: a new weight set must beat the incumbent's holdout log-loss by a margin to be promoted
- **automatic rollback** if the promoted weights lose money live over 20 trades. Log-loss is a proxy; realized PnL is the thing, and when they disagree the money wins.

**4. Filter cost — the loop almost nobody builds.** Every *rejected* candidate
is shadow-tracked for an hour. Without this a bot only ever learns from trades
it already agreed with, and it converges on silence.

```bash
alphahound filter-cost
```

A gate that rejected fifty candidates, forty of which then doubled, is not
protecting you — it *is* the strategy. Such gates get loosened automatically, at
a deliberately slower 10% step, because loosening increases the losses you can
take and should happen slower than tightening.

## Backtesting, honestly

`alphahound backtest` replays the bot's **own decision log**, not historical
market data. Nobody can honestly reconstruct what the terminal attribution or
holder ages looked like at minute four of a token that launched three weeks ago
— the features this strategy depends on are not recoverable after the fact, and
a backtest built on reconstructed features produces a beautiful equity curve and
zero information.

So it replays what was actually observed, with the real exit ladder applied to
shadow-tracked peaks (using the peak raw would credit perfect timing). Results
are horizon-capped at `learning.shadow_track_minutes`, which understates the
right tail. Reported explicitly.

```bash
alphahound backtest              # threshold sweep
alphahound backtest --compare    # prior weights vs learned weights
```

The useful output of the sweep is not the best number, it is the *shape*. A flat
curve means the threshold does not matter and the edge is elsewhere. A sharp
peak on 30 trades means you are fitting noise.

## The latency ladder

Where you actually stand, stated plainly:

| Tier | Latency | This bot |
|---|---|---|
| Public REST polling | seconds | ✅ Dexscreener |
| Public websocket | sub-second | ✅ PumpPortal |
| Geyser / Yellowstone gRPC | ~100ms | ❌ |
| Co-located with a validator | single-digit ms | ❌ |

If you are competing for the *first block* of a launch, you need the bottom two.
This bot targets the 1–10 minute window, where analysis quality still decides the
outcome rather than raw speed. That is a deliberate choice about where a solo
operator can actually win.

### Your RPC is the throughput ceiling

Full enrichment of one Solana candidate costs roughly 90 RPC calls: the mint
account, the largest holders and their owners, a launch-slot lookup, and one
`getTransaction` per signature up to `terminals.max_txs_inspected`. So:

| RPC | Rate | Candidates fully scored |
|---|---|---|
| Public `api.mainnet-beta.solana.com` | ~5/s (self-throttled) | ~1 per 2 minutes |
| Paid (Helius, Triton, QuickNode) | 25–50/s | ~1 per 2–4 seconds |

Measured on the public endpoint, not estimated. Paper mode falls back to it so
the on-chain features exist at all, and warns that they will be patchy; holder
distribution frequently comes back unmeasured because the fan-out gets rate
limited. Live mode refuses the public endpoint outright. **A paid RPC is not an
optimization here, it is the difference between scoring 30 candidates a minute
and scoring one.**

### Two loops, because risk cannot queue behind opportunity

Exits run in their own loop on `loop.tick_seconds`, entry scanning in another on
`loop.scan_seconds`. They were one loop first, and one slow candidate pushed a
pass past two minutes — during which a stop that should have filled at −28%
would instead fill wherever the token had drifted to. Enrichment latency belongs
to somebody else's rate limiter, so it is never allowed on the path that closes a
position.

A `heartbeat` line every `loop.heartbeat_seconds` reports what was seen and what
rejected it. A selective bot is silent for long stretches, and silence is
indistinguishable from a hung loop:

```json
{"msg": "heartbeat", "watching": 23, "open": 0, "equity_usd": 1000.0,
 "since_last": {"enriched": 1, "low_score": 1, "scan_budget_spent": 1},
 "best_probability": 0.061}
```

### One engine per state directory

Startup takes an OS-level lock on `state/`. Two engines sharing a wallet size
positions independently, so a 5% cap silently becomes 10%, and both believe they
own the position. The lock dies with the process, so a crash does not require
manual cleanup before restarting.

## Commands

| Command | Purpose |
|---|---|
| `alphahound run [--paper]` | start the bot |
| `alphahound doctor [--check-network]` | config, state, self-tuned params, connectivity |
| `alphahound preview [--port 8765]` | live PnL, holds, sold — open the URL, leave the tab open |
| `alphahound trades -n 50` | closed trades with error classes |
| `alphahound weights` | learned vs prior weights, with observation counts |
| `alphahound learn [--no-train]` | force a postmortem / training cycle |
| `alphahound filter-cost` | what the gates rejected that would have won |
| `alphahound backtest [--compare]` | replay decisions |
| `alphahound discover-terminals` | mine fee accounts to label |
| `alphahound label-terminal ADDR LABEL CLASS` | label one |
| `alphahound params [--set k=v] [--history N]` | inspect/override tunables |
| `alphahound pause` / `resume` | kill switch |

## Layout

```
config/strategy.toml      every tunable, commented with the reasoning
config/terminals.toml     attribution registry (fee accounts start empty)
config/whales.toml        Fomo/Moby wallets to chase (research, not venues)
src/alphahound/
  models.py               dataclasses; the feature vector contract
  settings.py             .env + TOML, stdlib only
  store.py                sqlite: decisions, shadows, trades, weights, params
  net.py                  per-host token bucket + retry
  providers.py            Dexscreener / Helius / Birdeye / Cope (Fomo research)
  discovery.py            polling + pump.fun websocket
  signals/                chart, distribution, terminals, flow, whales, solana reader
  scoring.py              normalization, priors, gates, EV
  playbook.py             per-chain age / copy / ladder / thesis cut
  risk.py                 Kelly, caps, kill switch, cooldown
  execution/              router + paper, jupiter, evm, robinhood
  portfolio.py            exit policy
  preview.py              localhost operator board
  learning.py             postmortem, nudges, training, filter cost
  backtest.py             decision-log replay
  engine.py               the loop
tests/test_core.py        77 tests, stdlib only, no install needed
```

Core modules are stdlib-only on purpose: `python tests/test_core.py` runs with
nothing installed. `httpx` is the only runtime dependency; `solders`, `web3`,
`eth-account`, `cryptography` and `websockets` are extras, imported lazily by
the adapter that needs them.

```bash
pip install -e ".[all]"    # everything
pip install -e ".[solana]" # just Solana trading
```

## Going live

`Settings.validate()` refuses to start a misconfigured live bot, because the
failure mode of a half-configured one is a wallet, not a stack trace. It
requires, among other things, that `SOLANA_RPC_URL` is **not** the public
endpoint — that endpoint will rate-limit you out of every trade worth making.

Before flipping `MODE=live`:

1. Run paper until `alphahound trades` shows **at least 40 closed trades**. Below
   that the model is running on priors and the learning loop deliberately
   refuses to train.
2. `alphahound backtest --compare` — have the learned weights actually beaten
   the priors?
3. `alphahound filter-cost` — are your gates costing more than they save?
4. Label at least one retail terminal, or the central feature is inert.
5. Set `risk.equity_usd` to money you can lose entirely.
6. Fund the trading wallet with a small balance. It is a hot wallet by
   definition; use a KMS/HSM signer for anything serious.

## Risk notice

This software trades volatile assets automatically and will lose money. The
default configuration is conservative and still loses money. Memecoin trading has
a negative expected value for most participants, and a bot does not change that —
it only makes the outcome arrive faster and more consistently.

The Robinhood adapter uses their official API; automated trading may be subject
to their terms and to securities regulation in your jurisdiction. Fomo and Moby
have no trading API and this project does not automate their apps.

You are responsible for your own compliance, your own keys, and your own losses.

## Licence

MIT. See [LICENSE](LICENSE).

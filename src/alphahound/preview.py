"""Live operator preview.

The engine writes `state/preview.json` every risk tick. `alphahound preview`
serves that file plus closed trades from sqlite on localhost so a browser tab
stays current without another data API.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from .models import Chain, now_ms
from .playbook import max_age_minutes
from .settings import Config, Settings
from .store import Store

PREVIEW_NAME = "preview.json"


def write_preview(state_dir: Path, payload: dict[str, Any]) -> None:
    path = state_dir / PREVIEW_NAME
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_preview(state_dir: Path) -> dict[str, Any]:
    path = state_dir / PREVIEW_NAME
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def universe(settings: Settings | None, strategy: Config | None) -> dict[str, Any]:
    if settings is None or strategy is None:
        return {"mode": "", "chains": []}
    pubs = settings.wallet_pubkeys()
    rows = []
    for chain in (Chain.SOLANA, Chain.BNB, Chain.ROBINHOOD_CHAIN):
        lp = strategy.section(f"launchpads.{chain.value}")
        pb = strategy.section(f"playbooks.{chain.value}")
        rows.append(
            {
                "chain": chain.value,
                "enabled": chain in settings.enabled_chains,
                "launchpads": list(lp.get("dex_ids") or []),
                "max_age_min": max_age_minutes(strategy, chain),
                "copy_max_age_min": float(pb.get("copy_max_age_minutes", 25)),
                "copy_max_mcap": float(pb.get("copy_max_mcap_usd", 400_000)),
                "min_volume_5m": float(pb.get("min_volume_5m", 0) or 0),
                "min_twitter": float(pb.get("min_twitter_mentions", 0) or 0),
                "wallet": pubs.get(chain.value, ""),
            }
        )
    return {"mode": settings.mode, "chains": rows}


def assemble(
    state_dir: Path,
    store: Store,
    equity: float,
    settings: Settings | None = None,
    strategy: Config | None = None,
) -> dict[str, Any]:
    live = read_preview(state_dir)
    trades = store.trades(limit=40)
    wins = sum(1 for t in trades if t.won)
    fees = sum(t.fees_usd for t in trades)
    sold = [
        {
            "key": t.key,
            "chain": t.chain.value,
            "venue": t.venue.value,
            "size_usd": round(t.size_usd, 2),
            "pnl_usd": round(t.pnl_usd, 2),
            "pnl_pct": round(t.pnl_pct, 4),
            "exit": t.exit_reason.value,
            "klass": t.error_class.value,
            "mfe": round(t.max_favorable_excursion, 4),
            "closed_at_ms": t.closed_at_ms,
        }
        for t in trades
    ]
    stale_s = 0.0
    if live.get("ts_ms"):
        stale_s = max(0.0, (now_ms() - int(live["ts_ms"])) / 1000.0)
    watch = live.get("watch") or []
    return {
        "ts_ms": now_ms(),
        "bot_ts_ms": live.get("ts_ms"),
        "stale_s": round(stale_s, 1),
        "running": stale_s < 15.0 if live.get("ts_ms") else False,
        "mode": live.get("mode") or (settings.mode if settings else ""),
        "halted": live.get("halted", False),
        "halt_reason": live.get("halt_reason", ""),
        "equity_usd": round(float(live.get("equity_usd", equity)), 2),
        "realized_pnl": round(store.realized_pnl(), 2),
        "fees_usd": round(fees, 2),
        "closed": len(trades) if trades else store.trade_count(),
        "wins": wins,
        "watching": live.get("watching", len(watch)),
        "best_probability": live.get("best_probability", 0.0),
        "tick": live.get("tick") or {},
        "holds": live.get("holds") or [],
        "watch": watch,
        "sold": sold,
        "universe": universe(settings, strategy),
    }


HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>alpha-hound</title>
<style>
  :root { color-scheme: dark; --line:#1c222b; --muted:#6b7785; --text:#d7dde5; --hi:#f2f5f8; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 14px/1.45 ui-monospace, SFMono-Regular, Consolas, monospace;
         background: #0b0d10; color: var(--text); }
  header { display: flex; gap: 28px; flex-wrap: wrap; align-items: flex-end;
           padding: 18px 22px; border-bottom: 1px solid var(--line); }
  h1 { font-size: 11px; letter-spacing: .14em; text-transform: uppercase;
       color: var(--muted); font-weight: 600; margin: 0 0 4px; }
  .n { font-size: 22px; color: var(--hi); }
  .up { color: #3ddc97; } .dn { color: #ff6b6b; } .muted { color: var(--muted); }
  .pill { font-size: 11px; padding: 2px 8px; border: 1px solid #2a3340; border-radius: 99px; }
  .on { border-color: #3ddc97; color: #3ddc97; }
  .off { opacity: .45; }
  main { display: grid; grid-template-columns: 1.1fr 1fr; }
  section { padding: 16px 22px; }
  section + section { border-left: 1px solid var(--line); }
  .full { grid-column: 1 / -1; border-top: 1px solid var(--line); border-left: 0 !important; }
  table { width: 100%; border-collapse: collapse; }
  th { text-align: left; color: var(--muted); font-weight: 500; font-size: 11px;
       letter-spacing: .08em; text-transform: uppercase; padding: 6px 10px 8px 0; }
  td { padding: 7px 10px 7px 0; border-top: 1px solid #161b22; vertical-align: top; }
  .sym { color: var(--hi); }
  .ca { color: var(--muted); font-size: 12px; }
  .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
  .card { border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; }
  .card h2 { margin: 0 0 8px; font-size: 13px; color: var(--hi); font-weight: 600; }
  .card p { margin: 0 0 6px; font-size: 12px; color: var(--muted); }
  .addr { font-size: 11px; color: var(--text); word-break: break-all; }
  @media (max-width: 900px) {
    main, .grid { grid-template-columns: 1fr; }
    section + section { border-left: 0; border-top: 1px solid var(--line); }
  }
</style>
</head>
<body>
<header>
  <div><h1>alpha-hound</h1><div id="status" class="muted">loading</div></div>
  <div><h1>equity</h1><div class="n" id="equity">—</div></div>
  <div><h1>realized</h1><div class="n" id="pnl">—</div></div>
  <div><h1>fees</h1><div class="n" id="fees">—</div></div>
  <div><h1>closed</h1><div class="n" id="closed">—</div></div>
  <div><h1>watching</h1><div class="n" id="watching">—</div></div>
</header>
<section class="full" style="padding:16px 22px 8px">
  <h1>what it trades</h1>
  <div class="grid" id="universe"></div>
</section>
<main>
  <section>
    <h1>in view</h1>
    <table id="watch"><thead><tr>
      <th>token</th><th>chain</th><th>age</th><th>mcap</th><th>vol 5m</th>
    </tr></thead><tbody></tbody></table>
  </section>
  <section>
    <h1>holds</h1>
    <table id="holds"><thead><tr>
      <th>token</th><th>size</th><th>pnl</th><th>left</th><th>age</th>
    </tr></thead><tbody></tbody></table>
    <h1 style="margin-top:22px">sold</h1>
    <table id="sold"><thead><tr>
      <th>token</th><th>size</th><th>pnl</th><th>exit</th>
    </tr></thead><tbody></tbody></table>
  </section>
</main>
<script>
const $ = id => document.getElementById(id);
const usd = n => (n<0?'−':'') + '$' + Math.abs(n).toFixed(2);
const compact = n => n>=1e6 ? '$'+(n/1e6).toFixed(1)+'m' : n>=1e3 ? '$'+(n/1e3).toFixed(1)+'k' : usd(n||0);
const pct = n => (n>=0?'+':'') + (n*100).toFixed(1) + '%';
const cls = n => n>0?'up':n<0?'dn':'';
const short = k => (k.split(':')[1] || k).slice(0, 8);

function rows(el, items, html, empty) {
  const body = el.querySelector('tbody');
  body.innerHTML = items.length ? items.map(html).join('') : '<tr><td class="muted" colspan="5">'+(empty||'none')+'</td></tr>';
}

function chainCard(c) {
  const pads = (c.launchpads||[]).join(', ') || '—';
  const extra = [];
  if (c.min_volume_5m) extra.push('vol ≥ '+compact(c.min_volume_5m));
  if (c.min_twitter) extra.push('twitter ≥ '+c.min_twitter);
  extra.push('copy <'+c.copy_max_age_min+'m · $'+(c.copy_max_mcap/1000)+'k');
  return `<div class="card ${c.enabled?'':'off'}">
    <h2>${c.chain} ${c.enabled?'<span class="pill on">on</span>':'<span class="pill">off</span>'}</h2>
    <p>${pads} · max ${c.max_age_min}m</p>
    <p>${extra.join(' · ')}</p>
    <div class="addr">${c.wallet || 'no wallet'}</div>
  </div>`;
}

async function tick() {
  const d = await (await fetch('/api')).json();
  const live = d.running ? (d.mode || 'run') : 'stopped';
  const halt = d.halted ? (' · paused ' + (d.halt_reason || '')) : '';
  $('status').textContent = live + halt + ' · stale ' + d.stale_s + 's';
  $('equity').textContent = usd(d.equity_usd);
  $('pnl').textContent = usd(d.realized_pnl);
  $('pnl').className = 'n ' + cls(d.realized_pnl);
  $('fees').textContent = usd(d.fees_usd);
  $('closed').textContent = d.closed + (d.closed ? '  ' + d.wins + 'w' : '');
  $('watching').textContent = d.watching;
  $('universe').innerHTML = (d.universe.chains||[]).map(chainCard).join('');
  rows($('watch'), d.watch||[], w => `<tr>
    <td><div class="sym">${w.symbol || short(w.address||'')}</div><div class="ca">${(w.address||'').slice(0,10)}</div></td>
    <td class="muted">${w.chain}</td>
    <td>${w.age_min}m</td>
    <td>${compact(w.mcap)}</td>
    <td>${compact(w.vol5m)}</td></tr>`, d.running ? 'scanning…' : 'start the bot to fill this');
  rows($('holds'), d.holds, h => `<tr>
    <td class="sym">${h.symbol || short(h.key)}</td>
    <td>${usd(h.size_usd)}</td>
    <td class="${cls(h.unrealized_pct)}">${pct(h.unrealized_pct)}</td>
    <td>${Math.round(h.remaining_pct*100)}%</td>
    <td class="muted">${h.age_min}m r${h.ladder}</td></tr>`);
  rows($('sold'), d.sold, t => `<tr>
    <td class="sym">${short(t.key)}</td>
    <td>${usd(t.size_usd)}</td>
    <td class="${cls(t.pnl_usd)}">${usd(t.pnl_usd)} ${pct(t.pnl_pct)}</td>
    <td class="muted">${t.exit}</td></tr>`);
}
tick();
setInterval(tick, 3000);
</script>
</body>
</html>
"""


def serve(
    state_dir: Path,
    store: Store,
    equity: float,
    port: int,
    settings: Settings | None = None,
    strategy: Config | None = None,
) -> None:
    def snapshot() -> bytes:
        fresh = Store(state_dir)
        try:
            return json.dumps(assemble(state_dir, fresh, equity, settings, strategy)).encode()
        finally:
            fresh.close()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] == "/api":
                try:
                    body = snapshot()
                except Exception as exc:  # noqa: BLE001
                    body = json.dumps({"error": str(exc)}).encode()
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML.encode())

    httpd = HTTPServer(("127.0.0.1", port), Handler)
    print(f"preview  http://127.0.0.1:{port}/   (ctrl+c to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        httpd.server_close()
        store.close()

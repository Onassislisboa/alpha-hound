"""Jupiter Swap v2 (Solana).

Endpoints per developers.jup.ag/docs/swap:

    GET  https://api.jup.ag/swap/v2/order    quote + assembled transaction
    POST https://api.jup.ag/swap/v2/execute  submit the signed transaction

`x-api-key` is required on both. Omitting `taker` on /order returns pricing
without a transaction, which is exactly what the round-trip probe wants: it can
price a hypothetical exit without a wallet, a signature, or a fee.

`solders` is imported lazily so that paper mode, the backtester and the tests
all run on a machine with nothing installed.
"""

from __future__ import annotations

import base64
from typing import Any

from ..log import get
from ..models import Candidate, Chain, Fill, Quote, Side, VenueId, now_ms
from ..net import Http
from ..settings import Config, Settings
from ..signals.solana import USDC, WSOL, SolanaReader
from . import ExecutionError
from ..fees import FeePlan, slip_bps

log = get("jupiter")

BASE_URL = "https://api.jup.ag/swap/v2"
USDC_DECIMALS = 6
SOL_DECIMALS = 9


class JupiterVenue:
    id = VenueId.JUPITER

    def __init__(self, http: Http, settings: Settings, strategy: Config) -> None:
        self.http = http
        self.settings = settings
        self.slippage_bps = int(strategy.get("execution.slippage_bps", 250))
        self.headers = {"x-api-key": settings.jupiter_api_key}
        self._decimals: dict[str, int] = {WSOL: SOL_DECIMALS, USDC: USDC_DECIMALS}
        self._sol_price: tuple[int, float] = (0, 0.0)
        self._reader = (
            SolanaReader(http, settings.solana_rpc_url) if settings.solana_rpc_url else None
        )
        self._keypair: Any = None

    def supports(self, chain: Chain) -> bool:
        return chain is Chain.SOLANA

    # -- helpers -----------------------------------------------------------
    async def _order(self, params: dict[str, Any]) -> dict[str, Any]:
        data = await self.http.get(f"{BASE_URL}/order", params=params, headers=self.headers)
        if not isinstance(data, dict):
            raise ExecutionError(f"jupiter /order returned {type(data)}")
        # /order signals a priceable-but-unbuildable route with transaction ""
        # and an errorCode, rather than an HTTP error.
        if data.get("errorCode"):
            raise ExecutionError(
                f"jupiter /order error {data['errorCode']}: {data.get('errorMessage', '')}"
            )
        return data

    async def _sol_price_usd(self) -> float:
        """Priced through /order itself rather than a separate price endpoint,
        so this adapter depends on exactly one API surface."""
        ts, cached = self._sol_price
        if cached > 0 and now_ms() - ts < 60_000:
            return cached
        data = await self._order(
            {
                "inputMint": WSOL,
                "outputMint": USDC,
                "amount": 10**SOL_DECIMALS,
                "slippageBps": 50,
            }
        )
        out = float(data.get("outAmount") or 0) / 10**USDC_DECIMALS
        if out > 0:
            self._sol_price = (now_ms(), out)
            return out
        if cached > 0:
            return cached
        raise ExecutionError("could not price SOL")

    async def _token_decimals(self, mint: str) -> int:
        if mint in self._decimals:
            return self._decimals[mint]
        if self._reader is None:
            raise ExecutionError("SOLANA_RPC_URL is required to read token decimals")
        info = await self._reader.mint_info(mint)
        if info is None:
            raise ExecutionError(f"mint {mint} not found")
        self._decimals[mint] = info.decimals
        return info.decimals

    def _signer(self):
        if self._keypair is not None:
            return self._keypair
        if not self.settings.solana_private_key:
            raise ExecutionError("SOLANA_PRIVATE_KEY is not set")
        try:
            from solders.keypair import Keypair
        except ImportError as exc:  # pragma: no cover
            raise ExecutionError("pip install 'alphahound[solana]' to trade on Solana") from exc
        self._keypair = Keypair.from_base58_string(self.settings.solana_private_key)
        return self._keypair

    # -- venue interface ---------------------------------------------------
    async def quote(
        self, candidate: Candidate, side: Side, amount: float, fees: FeePlan | None = None
    ) -> Quote:
        sol_price = await self._sol_price_usd()
        decimals = await self._token_decimals(candidate.address)
        slip = slip_bps(fees, self.slippage_bps)

        if side is Side.BUY:
            lamports = int((amount / sol_price) * 10**SOL_DECIMALS)
            if lamports <= 0:
                raise ExecutionError(f"buy size {amount} USD rounds to zero lamports")
            params = {
                "inputMint": WSOL,
                "outputMint": candidate.address,
                "amount": lamports,
                "slippageBps": slip,
                "restrictIntermediateTokens": "true",
            }
        else:
            base_units = int(amount * 10**decimals)
            if base_units <= 0:
                raise ExecutionError(f"sell size {amount} rounds to zero base units")
            params = {
                "inputMint": candidate.address,
                "outputMint": WSOL,
                "amount": base_units,
                "slippageBps": slip,
                "restrictIntermediateTokens": "true",
            }
        if fees is not None:
            params["priorityFeeLamports"] = fees.priority_lamports
            params["dynamicComputeUnitLimit"] = "true"

        data = await self._order(params)
        data["_params"] = params
        impact = abs(float(data.get("priceImpactPct") or 0.0))

        if side is Side.BUY:
            tokens = float(data.get("outAmount") or 0) / 10**decimals
            price = amount / tokens if tokens > 0 else 0.0
            return Quote(
                venue=self.id,
                in_amount=amount,
                out_amount=tokens,
                price=price,
                price_impact=impact,
                raw=data,
            )

        sol_out = float(data.get("outAmount") or 0) / 10**SOL_DECIMALS
        usd_out = sol_out * sol_price
        return Quote(
            venue=self.id,
            in_amount=amount,
            out_amount=usd_out,
            price=usd_out / amount if amount > 0 else 0.0,
            price_impact=impact,
            raw=data,
        )

    async def execute(
        self, candidate: Candidate, side: Side, amount: float, quote: Quote
    ) -> Fill:
        keypair = self._signer()
        try:
            from solders.transaction import VersionedTransaction
        except ImportError as exc:  # pragma: no cover
            raise ExecutionError("pip install 'alphahound[solana]' to trade on Solana") from exc

        # The probe quote was built without a taker, so it carries no
        # transaction. Re-quote with the taker set to get one.
        params = dict(quote.raw.get("_params") or {})
        if not params:
            params = self._params_for(candidate, side, amount, quote)
        params["taker"] = str(keypair.pubkey())
        order = await self._order(params)

        tx_b64 = order.get("transaction")
        if not tx_b64:
            raise ExecutionError(
                f"jupiter returned no transaction (errorCode={order.get('errorCode')})"
            )

        unsigned = VersionedTransaction.from_bytes(base64.b64decode(tx_b64))
        signed = VersionedTransaction(unsigned.message, [keypair])
        payload = {
            "signedTransaction": base64.b64encode(bytes(signed)).decode(),
            "requestId": order["requestId"],
        }
        result = await self.http.post(
            f"{BASE_URL}/execute", json_body=payload, headers=self.headers, retries=1
        )
        if not isinstance(result, dict):
            raise ExecutionError(f"jupiter /execute returned {type(result)}")
        status = str(result.get("status", "")).lower()
        if status and status != "success":
            sig = str(result.get("signature") or "")
            raise ExecutionError(
                f"jupiter execute failed: {result.get('error') or result.get('code') or result}",
                retryable=not bool(sig),
            )

        decimals = await self._token_decimals(candidate.address)
        sol_price = await self._sol_price_usd()
        if side is Side.BUY:
            filled = float(order.get("outAmount") or 0) / 10**decimals
        else:
            filled = float(order.get("outAmount") or 0) / 10**SOL_DECIMALS * sol_price

        slippage = 0.0
        if quote.out_amount > 0 and filled > 0:
            slippage = 1.0 - filled / quote.out_amount

        return Fill(
            venue=self.id,
            side=side,
            amount_in=amount,
            amount_out=filled,
            price=(amount / filled if side is Side.BUY and filled else filled / amount if amount else 0.0),
            fee_usd=self._fee_usd(order, sol_price),
            tx_id=str(result.get("signature") or ""),
            slippage_vs_quote=slippage,
        )

    def _params_for(
        self, candidate: Candidate, side: Side, amount: float, quote: Quote
    ) -> dict[str, Any]:
        raw = quote.raw or {}
        params = {
            "inputMint": raw.get("inputMint") or (WSOL if side is Side.BUY else candidate.address),
            "outputMint": raw.get("outputMint")
            or (candidate.address if side is Side.BUY else WSOL),
            "amount": raw.get("inAmount"),
            "slippageBps": raw.get("slippageBps") or self.slippage_bps,
            "restrictIntermediateTokens": "true",
        }
        if raw.get("priorityFeeLamports") is not None:
            params["priorityFeeLamports"] = raw["priorityFeeLamports"]
            params["dynamicComputeUnitLimit"] = raw.get("dynamicComputeUnitLimit", "true")
        return params

    @staticmethod
    def _fee_usd(order: dict[str, Any], sol_price: float) -> float:
        platform = order.get("platformFee") or {}
        fee_bps = float(order.get("feeBps") or platform.get("feeBps") or 0)
        in_amount = float(order.get("inAmount") or 0)
        if not fee_bps or not in_amount:
            return 0.0
        if order.get("inputMint") == WSOL:
            return (in_amount / 10**SOL_DECIMALS) * sol_price * fee_bps / 10_000.0
        return 0.0

"""Module B: Strategic Portfolio Stress Testing for Wholesale Banking Assets.

Simulates the impact of high-impact real-world financial events (e.g. Geopolitical
conflict, Macroeconomic interest rate shifts, Credit/Default contagion) on a synthetic
portfolio of wholesale banking assets (corporate loans, fixed-income bonds, derivatives,
and strategic equity holdings).

Subscribes to:
- Event Classification (DeBERTa NLI zero-shot)
- Sentiment Score (FinBERT in [-1, 1])
- Market Impact Score (Composite [1, 10])

Triggers stress test when impact_score >= trigger_threshold (default 7.0).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .models import Article

log = logging.getLogger(__name__)

# ── Wholesale Banking Asset Portfolio (Base: $1,000,000,000 / $1 Billion) ───────

DEFAULT_WHOLESALE_PORTFOLIO: list[dict[str, Any]] = [
    # ── 1. Commercial & Wholesale Loans ($410M total) ──
    {
        "id": "LOAN-001",
        "name": "Syndicated Term Loan B - Apex Industrial Corp",
        "asset_class": "loans",
        "sector": "Industrials",
        "notional": 120_000_000,
        "base_value": 118_500_000,
        "coupon_rate": "SOFR + 2.75%",
        "rating": "BB+",
        "maturity_years": 4.5,
        "duration": 0.25,  # floating rate has near-zero duration
        "spread_bps": 275,
        "collateral": "First Lien Senior Secured",
    },
    {
        "id": "LOAN-002",
        "name": "Revolving Credit Facility - Global Energy Infrastructure",
        "asset_class": "loans",
        "sector": "Energy",
        "notional": 90_000_000,
        "base_value": 89_100_000,
        "coupon_rate": "SOFR + 2.25%",
        "rating": "BBB-",
        "maturity_years": 3.0,
        "duration": 0.25,
        "spread_bps": 225,
        "collateral": "Corporate Guarantee & Refinery Assets",
    },
    {
        "id": "LOAN-003",
        "name": "Commercial Real Estate Facility - Metro Plaza Development",
        "asset_class": "loans",
        "sector": "Real Estate",
        "notional": 80_000_000,
        "base_value": 78_000_000,
        "coupon_rate": "5.45% Fixed",
        "rating": "BB",
        "maturity_years": 5.0,
        "duration": 4.1,
        "spread_bps": 340,
        "collateral": "Prime Commercial Real Estate Mortgage",
    },
    {
        "id": "LOAN-004",
        "name": "Trade Finance Facility - Pacific Rim Logistics",
        "asset_class": "loans",
        "sector": "Logistics & Shipping",
        "notional": 70_000_000,
        "base_value": 69_800_000,
        "coupon_rate": "SOFR + 1.85%",
        "rating": "A-",
        "maturity_years": 1.5,
        "duration": 0.25,
        "spread_bps": 185,
        "collateral": "Export Receivables & Bills of Lading",
    },
    {
        "id": "LOAN-005",
        "name": "Sponsor-Backed Leveraged Loan - TechEdge Cloud Holdings",
        "asset_class": "loans",
        "sector": "Technology",
        "notional": 50_000_000,
        "base_value": 48_500_000,
        "coupon_rate": "SOFR + 3.85%",
        "rating": "B+",
        "maturity_years": 5.5,
        "duration": 0.25,
        "spread_bps": 385,
        "collateral": "Software IP & SaaS Recurring ARR",
    },

    # ── 2. Fixed Income / Bonds ($310M total) ──
    {
        "id": "BOND-001",
        "name": "US Treasury Benchmark 10Y Sovereign Note",
        "asset_class": "bonds",
        "sector": "Sovereign Government",
        "notional": 140_000_000,
        "base_value": 138_600_000,
        "coupon_rate": "4.125% Fixed",
        "rating": "AAA",
        "maturity_years": 9.2,
        "duration": 7.8,
        "spread_bps": 0,
        "collateral": "Full Faith & Credit US Government",
    },
    {
        "id": "BOND-002",
        "name": "Investment-Grade Senior Unsecured - Microsoft / Apple Corp",
        "asset_class": "bonds",
        "sector": "Technology",
        "notional": 80_000_000,
        "base_value": 79_200_000,
        "coupon_rate": "4.75% Fixed",
        "rating": "AA+",
        "maturity_years": 6.0,
        "duration": 5.2,
        "spread_bps": 85,
        "collateral": "Senior Corporate Obligation",
    },
    {
        "id": "BOND-003",
        "name": "High-Yield Corporate Bond - Pioneer Telecoms 2029",
        "asset_class": "bonds",
        "sector": "Telecommunications",
        "notional": 50_000_000,
        "base_value": 47_500_000,
        "coupon_rate": "7.25% Fixed",
        "rating": "B",
        "maturity_years": 4.8,
        "duration": 3.9,
        "spread_bps": 410,
        "collateral": "Unsecured High Yield Debenture",
    },
    {
        "id": "BOND-004",
        "name": "Emerging Market Sovereign Debt - Latin America 7Y",
        "asset_class": "bonds",
        "sector": "Emerging Markets",
        "notional": 40_000_000,
        "base_value": 38_800_000,
        "coupon_rate": "6.00% Fixed",
        "rating": "BBB-",
        "maturity_years": 6.8,
        "duration": 5.6,
        "spread_bps": 260,
        "collateral": "Sovereign External Debt",
    },

    # ── 3. Derivatives & Off-Balance Sheet ($180M total) ──
    {
        "id": "DERIV-001",
        "name": "Interest Rate Swap - 10Y Receive Fixed 3.85% / Pay SOFR",
        "asset_class": "derivatives",
        "sector": "Rates",
        "notional": 100_000_000,
        "base_value": 4_200_000,  # Net MTM positive value
        "coupon_rate": "Receive Fixed 3.85% / Pay Float",
        "rating": "AA-",
        "maturity_years": 7.5,
        "duration": 6.8,  # Duration equivalent
        "spread_bps": 0,
        "collateral": "Bilateral CSA Cash Collateralized",
    },
    {
        "id": "DERIV-002",
        "name": "Cross-Currency FX Swap - EUR/USD Notional Hedge",
        "asset_class": "derivatives",
        "sector": "Foreign Exchange",
        "notional": 45_000_000,
        "base_value": 1_800_000,
        "coupon_rate": "FX Forward Rate 1.0850",
        "rating": "AA",
        "maturity_years": 1.2,
        "duration": 1.1,
        "spread_bps": 0,
        "collateral": "ISDA Master Agreement Daily Variation Margin",
    },
    {
        "id": "DERIV-003",
        "name": "Credit Default Swap (CDS) - Sold Protection on CDX NA IG 5Y",
        "asset_class": "derivatives",
        "sector": "Credit Derivatives",
        "notional": 35_000_000,
        "base_value": 2_100_000,
        "coupon_rate": "100 bps Annual Premium",
        "rating": "A+",
        "maturity_years": 4.0,
        "duration": 3.8,
        "spread_bps": 75,
        "collateral": "Centrally Cleared (ICE Clear Credit)",
    },

    # ── 4. Wholesale Equities & Strategic Participations ($100M total) ──
    {
        "id": "EQ-001",
        "name": "Strategic Wholesale Equity - Global Clean Energy Consortium",
        "asset_class": "equities",
        "sector": "Utilities & Renewables",
        "notional": 55_000_000,
        "base_value": 55_000_000,
        "coupon_rate": "Dividend Yield 3.2%",
        "rating": "BBB+",
        "maturity_years": 0.0,
        "duration": 0.0,
        "spread_bps": 0,
        "collateral": "Direct Equity Shares & Board Representation",
    },
    {
        "id": "EQ-002",
        "name": "Wholesale Market Equity Portfolio - Blue-Chip Tech & Industrial",
        "asset_class": "equities",
        "sector": "Diversified Large Cap",
        "notional": 45_000_000,
        "base_value": 45_000_000,
        "coupon_rate": "Dividend Yield 1.8%",
        "rating": "A",
        "maturity_years": 0.0,
        "duration": 0.0,
        "spread_bps": 0,
        "collateral": "Listed Public Equities",
    },
]


# ── Stress Testing Engine ───────────────────────────────────────────────────────

@dataclass
class ShockProfile:
    """Set of macroeconomic and market risk factor shocks for an event."""
    rate_shock_bps: float = 0.0          # parallel yield curve shift in basis points (e.g. +200 bps = +2.0%)
    gov_spread_shock_bps: float = 0.0    # sovereign flight-to-safety yield shift in bps (e.g. -40 bps)
    ig_spread_shock_bps: float = 0.0     # investment grade corporate credit spread widening in bps
    hy_spread_shock_bps: float = 0.0     # high yield credit spread widening in bps
    equity_shock_pct: float = 0.0        # equity price shock percentage (e.g. -12.0%)
    loan_impairment_pct: float = 0.0     # expected credit loss / default rate shock on loans (e.g. -6.0%)
    fx_shock_pct: float = 0.0            # foreign exchange / currency volatility shock
    commodity_shock_pct: float = 0.0     # energy/commodity price surge
    narrative: str = ""


def get_shock_profile_for_event(
    event_type: str,
    sentiment_score: float,
    impact_score: float,
) -> ShockProfile:
    """Determine quantitative risk factor shocks based on NLP event category and severity."""
    # Scale factor lambda in [0.4, 1.4] driven by impact score (7.0 to 10.0)
    severity_mult = max(0.4, min(1.4, (impact_score - 4.0) / 4.0))

    # Sentiment direction multiplier (negative sentiment exacerbates risk)
    sent_mult = 1.0 + max(0.0, -sentiment_score) * 0.5

    scale = severity_mult * sent_mult

    ev = (event_type or "").lower().strip()

    if ev == "geopolitical":
        # Geopolitical crisis (conflict, war, sanctions, embargoes):
        # - Flight to safety: Sovereign yields drop (bond prices rise)
        # - Credit spreads widen sharply
        # - Equities sell off heavily
        # - Commodity/Energy prices surge, shipping/logistics loans stressed
        return ShockProfile(
            rate_shock_bps=-35.0 * scale,          # Flight to quality drops 10Y treasury yields
            gov_spread_shock_bps=-35.0 * scale,
            ig_spread_shock_bps=85.0 * scale,      # IG spreads widen +85 bps
            hy_spread_shock_bps=240.0 * scale,     # High-yield spreads blow out +240 bps
            equity_shock_pct=-14.0 * scale,        # Global equity drop -14%
            loan_impairment_pct=-7.5 * scale,      # Trade/logistics and supply chain loan ECL
            fx_shock_pct=-8.0 * scale,             # USD surges, non-USD FX drops
            commodity_shock_pct=+22.0 * scale,     # Energy/oil supply shock
            narrative=(
                f"Severe Geopolitical Crisis: Flight-to-safety drives sovereign yields down by "
                f"{abs(35.0 * scale):.0f} bps, while high-yield credit spreads widen by "
                f"+{240.0 * scale:.0f} bps and equities fall by {14.0 * scale:.1f}%."
            ),
        )

    elif ev == "macroeconomic":
        # Macroeconomic shock (inflation spike, aggressive Fed interest rate hike):
        # - Upward interest rate shift (+180 bps)
        # - Large duration loss on fixed-rate bonds and fixed loans
        # - Receive fixed swaps suffer mark-to-market losses
        # - Equities decline
        return ShockProfile(
            rate_shock_bps=+185.0 * scale,         # +185 bps parallel curve shift
            gov_spread_shock_bps=0.0,
            ig_spread_shock_bps=65.0 * scale,
            hy_spread_shock_bps=160.0 * scale,
            equity_shock_pct=-11.0 * scale,
            loan_impairment_pct=-4.0 * scale,
            fx_shock_pct=-4.0 * scale,
            commodity_shock_pct=+5.0 * scale,
            narrative=(
                f"Aggressive Macroeconomic Tightening: Benchmark interest rates shock upward by "
                f"+{185.0 * scale:.0f} bps, triggering substantial duration losses across fixed-rate "
                f"bonds and receive-fixed interest rate swaps."
            ),
        )

    elif ev == "credit_event":
        # Credit event (sovereign/corporate debt default, liquidity freeze, banking contagion):
        # - Extreme credit spread blowout
        # - High loan defaults / write-downs
        # - Sold CDS protection incurs severe liability
        # - High-yield bond crash
        return ShockProfile(
            rate_shock_bps=-20.0 * scale,
            gov_spread_shock_bps=-20.0 * scale,
            ig_spread_shock_bps=140.0 * scale,     # IG spreads widen +140 bps
            hy_spread_shock_bps=380.0 * scale,     # HY spreads widen +380 bps
            equity_shock_pct=-16.5 * scale,        # Severe systemic contagion
            loan_impairment_pct=-14.0 * scale,     # Substantial credit facility haircuts
            fx_shock_pct=-6.0 * scale,
            commodity_shock_pct=-8.0 * scale,
            narrative=(
                f"Systemic Credit Contagion: High-yield spreads blow out by +{380.0 * scale:.0f} bps. "
                f"Severe loan impairments of {14.0 * scale:.1f}% applied across vulnerable wholesale facilities "
                f"and credit derivative liabilities."
            ),
        )

    elif ev in ("earnings", "mergers_acquisitions"):
        # Corporate restructuring / corporate earnings collapse:
        return ShockProfile(
            rate_shock_bps=+15.0 * scale,
            gov_spread_shock_bps=0.0,
            ig_spread_shock_bps=45.0 * scale,
            hy_spread_shock_bps=120.0 * scale,
            equity_shock_pct=-12.5 * scale,
            loan_impairment_pct=-5.0 * scale,
            fx_shock_pct=-2.0 * scale,
            commodity_shock_pct=0.0,
            narrative=(
                f"Corporate Sector Distress: Equity valuation compression of {12.5 * scale:.1f}% "
                f"with corporate debt spread widening across leveraged facilities."
            ),
        )

    else:
        # Generic high-impact market disturbance
        return ShockProfile(
            rate_shock_bps=+40.0 * scale,
            gov_spread_shock_bps=-10.0 * scale,
            ig_spread_shock_bps=50.0 * scale,
            hy_spread_shock_bps=130.0 * scale,
            equity_shock_pct=-8.0 * scale,
            loan_impairment_pct=-3.5 * scale,
            fx_shock_pct=-3.0 * scale,
            commodity_shock_pct=0.0,
            narrative=(
                f"Broad Market Turmoil: Risk-off re-pricing with +{130.0 * scale:.0f} bps spread widening "
                f"and {8.0 * scale:.1f}% equity drawdown."
            ),
        )


def apply_asset_stress(asset: dict[str, Any], shock: ShockProfile) -> dict[str, Any]:
    """Calculate the stressed valuation of a single wholesale banking asset."""
    base_val = float(asset["base_value"])
    asset_class = asset["asset_class"]
    duration = float(asset.get("duration", 0.0))
    rating = asset.get("rating", "BBB")
    sector = asset.get("sector", "General")

    delta_pct = 0.0
    primary_driver = ""

    if asset_class == "bonds":
        # Bond pricing: price change = -Duration * (delta_rate + delta_spread)
        if "AAA" in rating or "Sovereign Government" in sector:
            # Sovereign government debt
            delta_yield = (shock.rate_shock_bps + shock.gov_spread_shock_bps) / 10000.0
            delta_pct = -duration * delta_yield
            primary_driver = f"Sovereign Yield Shift ({(shock.rate_shock_bps + shock.gov_spread_shock_bps):+.0f} bps)"
        elif "A" in rating or "AA" in rating or "BBB" in rating:
            # Investment grade corporate bond
            delta_yield = (shock.rate_shock_bps + shock.ig_spread_shock_bps) / 10000.0
            delta_pct = -duration * delta_yield
            primary_driver = f"IG Spread (+{shock.ig_spread_shock_bps:.0f} bps) & Rates"
        else:
            # High yield bond
            delta_yield = (shock.rate_shock_bps + shock.hy_spread_shock_bps) / 10000.0
            delta_pct = -duration * delta_yield
            primary_driver = f"HY Spread Blowout (+{shock.hy_spread_shock_bps:.0f} bps)"

    elif asset_class == "loans":
        # Loan valuation: floating loans have minimal duration, but suffer credit impairment / ECL
        if "Fixed" in asset.get("coupon_rate", ""):
            # Fixed rate loan: duration loss + credit loss
            rate_loss = -duration * (shock.rate_shock_bps / 10000.0)
            credit_loss = shock.loan_impairment_pct / 100.0
            delta_pct = rate_loss + credit_loss
            primary_driver = f"Duration ({rate_loss * 100:.1f}%) + ECL Credit Loss ({credit_loss * 100:.1f}%)"
        else:
            # Floating rate loan: primarily expected credit loss (ECL) & spread widening
            spread_factor = shock.hy_spread_shock_bps if ("B" in rating and "BBB" not in rating) else shock.ig_spread_shock_bps
            spread_loss = -0.5 * (spread_factor / 10000.0)  # secondary market loan discount
            credit_loss = shock.loan_impairment_pct / 100.0
            delta_pct = spread_loss + credit_loss
            primary_driver = f"Loan Impairment & ECL ({credit_loss * 100:.1f}%)"

    elif asset_class == "derivatives":
        if "Swap" in asset["name"] and "Rates" in sector:
            # Receive Fixed / Pay Float: gains when rates drop, loses when rates rise
            # Delta V = -Duration * Notional * delta_rate
            delta_yield = shock.rate_shock_bps / 10000.0
            mtm_change = -duration * float(asset["notional"]) * delta_yield
            delta_pct = mtm_change / base_val
            primary_driver = f"IRS DV01 Curve Shift ({shock.rate_shock_bps:+.0f} bps)"
        elif "Cross-Currency" in asset["name"]:
            # FX Swap: currency move
            delta_pct = shock.fx_shock_pct / 100.0
            primary_driver = f"FX Volatility & Spread ({shock.fx_shock_pct:+.1f}%)"
        elif "Credit Default Swap" in asset["name"]:
            # Sold protection: loses value when credit spreads widen
            cds_spread_shift = shock.ig_spread_shock_bps / 10000.0
            mtm_loss = -duration * float(asset["notional"]) * cds_spread_shift
            delta_pct = mtm_loss / base_val
            primary_driver = f"CDS Spread Widening (+{shock.ig_spread_shock_bps:.0f} bps)"
        else:
            delta_pct = shock.equity_shock_pct / 100.0
            primary_driver = "Derivatives MTM Mark"

    elif asset_class == "equities":
        delta_pct = shock.equity_shock_pct / 100.0
        primary_driver = f"Broad Equity Shock ({shock.equity_shock_pct:+.1f}%)"

    # Bound extreme losses
    delta_pct = max(-0.85, delta_pct)
    stressed_val = max(0.0, base_val * (1.0 + delta_pct))
    pnl = stressed_val - base_val

    return {
        "id": asset["id"],
        "name": asset["name"],
        "asset_class": asset_class,
        "sector": sector,
        "notional": float(asset["notional"]),
        "base_value": round(base_val, 2),
        "stressed_value": round(stressed_val, 2),
        "pnl": round(pnl, 2),
        "pnl_pct": round(delta_pct * 100.0, 2),
        "rating": rating,
        "duration": duration,
        "coupon_rate": asset.get("coupon_rate", ""),
        "primary_driver": primary_driver,
    }


def run_portfolio_stress_test(
    portfolio: list[dict[str, Any]] | None = None,
    event_type: str = "geopolitical",
    sentiment_score: float = -0.75,
    impact_score: float = 8.5,
    trigger_threshold: float = 7.0,
    article_title: str = "",
) -> dict[str, Any]:
    """Execute the strategic portfolio stress test simulation.

    Parameters
    -------
    portfolio          : wholesale banking asset list (defaults to DEFAULT_WHOLESALE_PORTFOLIO)
    event_type         : classified NLP event type
    sentiment_score    : sentiment score in [-1.0, 1.0]
    impact_score       : market impact score in [1.0, 10.0]
    trigger_threshold  : threshold to trigger stress test (default 7.0)
    article_title      : triggering news headline text
    """
    portfolio = portfolio or DEFAULT_WHOLESALE_PORTFOLIO
    is_triggered = impact_score >= trigger_threshold

    shock = get_shock_profile_for_event(event_type, sentiment_score, impact_score)

    asset_results = []
    base_total = 0.0
    stressed_total = 0.0

    class_totals: dict[str, dict[str, float]] = {
        "loans":       {"base": 0.0, "stressed": 0.0, "pnl": 0.0},
        "bonds":       {"base": 0.0, "stressed": 0.0, "pnl": 0.0},
        "derivatives": {"base": 0.0, "stressed": 0.0, "pnl": 0.0},
        "equities":    {"base": 0.0, "stressed": 0.0, "pnl": 0.0},
    }

    for asset in portfolio:
        st = apply_asset_stress(asset, shock)
        asset_results.append(st)

        b = st["base_value"]
        s = st["stressed_value"]
        p = st["pnl"]
        ac = st["asset_class"]

        base_total += b
        stressed_total += s

        if ac in class_totals:
            class_totals[ac]["base"] += b
            class_totals[ac]["stressed"] += s
            class_totals[ac]["pnl"] += p

    total_pnl = stressed_total - base_total
    total_pnl_pct = (total_pnl / base_total * 100.0) if base_total > 0 else 0.0

    # Asset class summaries
    class_summary = {}
    for ac, vals in class_totals.items():
        b = vals["base"]
        p = vals["pnl"]
        pct = (p / b * 100.0) if b > 0 else 0.0
        class_summary[ac] = {
            "base_value": round(b, 2),
            "stressed_value": round(vals["stressed"], 2),
            "pnl": round(p, 2),
            "pnl_pct": round(pct, 2),
        }

    # Regulatory Capital & Solvency Impact (Wholesale Banking Metrics)
    # Assume baseline CET1 Capital of $135M on $1,000M risk-weighted assets (13.50% CET1 ratio)
    baseline_cet1_capital = 135_000_000.0
    stressed_cet1_capital = max(0.0, baseline_cet1_capital + total_pnl)
    baseline_cet1_ratio = 13.50
    stressed_cet1_ratio = (stressed_cet1_capital / base_total) * 100.0
    cet1_drawdown_bps = (stressed_cet1_ratio - baseline_cet1_ratio) * 100.0

    return {
        "is_stress_triggered": is_triggered,
        "trigger_threshold": trigger_threshold,
        "triggering_event": {
            "title": article_title or "Synthetic Event Scenario",
            "event_type": event_type,
            "sentiment_score": round(sentiment_score, 4),
            "impact_score": round(impact_score, 2),
            "narrative": shock.narrative,
        },
        "shocks_applied": {
            "rate_shock_bps": round(shock.rate_shock_bps, 1),
            "gov_spread_shock_bps": round(shock.gov_spread_shock_bps, 1),
            "ig_spread_shock_bps": round(shock.ig_spread_shock_bps, 1),
            "hy_spread_shock_bps": round(shock.hy_spread_shock_bps, 1),
            "equity_shock_pct": round(shock.equity_shock_pct, 2),
            "loan_impairment_pct": round(shock.loan_impairment_pct, 2),
            "fx_shock_pct": round(shock.fx_shock_pct, 2),
            "commodity_shock_pct": round(shock.commodity_shock_pct, 2),
        },
        "portfolio_summary": {
            "total_assets_count": len(portfolio),
            "base_portfolio_value": round(base_total, 2),
            "stressed_portfolio_value": round(stressed_total, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl_pct, 2),
            "base_portfolio_value_formatted": f"${base_total / 1e6:.1f}M",
            "stressed_portfolio_value_formatted": f"${stressed_total / 1e6:.1f}M",
            "total_pnl_formatted": f"{'+' if total_pnl >= 0 else ''}${total_pnl / 1e6:.2f}M",
        },
        "asset_class_breakdown": class_summary,
        "regulatory_capital": {
            "baseline_cet1_ratio": baseline_cet1_ratio,
            "stressed_cet1_ratio": round(stressed_cet1_ratio, 2),
            "cet1_drawdown_bps": round(cet1_drawdown_bps, 1),
            "baseline_capital_m": round(baseline_cet1_capital / 1e6, 2),
            "stressed_capital_m": round(stressed_cet1_capital / 1e6, 2),
            "capital_surplus_after_stress": round((stressed_cet1_capital - (base_total * 0.08)) / 1e6, 2),  # 8.0% min pillar 1
            "var_99_loss_m": round(abs(min(0.0, total_pnl * 1.15)) / 1e6, 2),
        },
        "asset_details": asset_results,
    }

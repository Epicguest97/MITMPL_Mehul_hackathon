"""Step 8: portfolio module - deterministic mapping signals -> target weights under hard limits."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskLimits:
    max_position: float = 0.10     # |weight| per name
    max_gross: float = 1.0         # sum |weights|
    max_net: float = 0.5           # |sum weights|
    min_signal: float = 0.10       # ignore weak signals
    allow_short: bool = True
    max_turnover: float = 0.5      # per rebalance, sum |w_new - w_old|
    scale: float = 0.25            # weight per unit signal before limits


def target_weights(signals: dict[str, float], limits: RiskLimits = RiskLimits(),
                   current: dict[str, float] | None = None) -> dict[str, float]:
    current = current or {}
    w: dict[str, float] = {}
    for t in sorted(signals):  # sorted => deterministic
        s = signals[t]
        if abs(s) < limits.min_signal or (s < 0 and not limits.allow_short):
            continue
        w[t] = max(-limits.max_position, min(limits.max_position, s * limits.scale))
    # gross limit
    gross = sum(abs(x) for x in w.values())
    if gross > limits.max_gross:
        w = {t: x * limits.max_gross / gross for t, x in w.items()}
    # net limit: scale the heavier side down
    net = sum(w.values())
    if abs(net) > limits.max_net:
        heavy = [t for t, x in w.items() if (x > 0) == (net > 0)]
        hsum = sum(abs(w[t]) for t in heavy)
        excess = abs(net) - limits.max_net
        f = (hsum - excess) / hsum
        for t in heavy:
            w[t] *= f
    # turnover limit: move proportionally toward target
    names = set(w) | set(current)
    turnover = sum(abs(w.get(t, 0.0) - current.get(t, 0.0)) for t in names)
    if turnover > limits.max_turnover:
        k = limits.max_turnover / turnover
        w = {t: current.get(t, 0.0) + k * (w.get(t, 0.0) - current.get(t, 0.0)) for t in names}
        w = {t: x for t, x in w.items() if abs(x) > 1e-12}
    return w

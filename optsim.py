"""Shared engine: trade a stock signal with a single long call or put.

Signals are evaluated at a bar's close and filled at the NEXT bar's open. The contract is
bought `dte` trading days from expiry at the strike whose delta is closest to `delta`, priced
with Black-Scholes on trailing 60-day realized vol x `iv_mult`, paying `half_spread` (a
fraction of the mid) on the way in and out.

A trade ends on the first of:
  * the signal's own exit (sold at the next open),
  * `max_hold` trading days (sold at the next open), so the contract still has time left,
  * a stock stop `stop_atr` x ATR beyond the entry, trailing with the price (filled at the
    stop or a worse open),
  * an option stop: the contract's value at the day's worst price falls to (1 - opt_stop) of
    what was paid (filled at that level or a worse open).

Each trade also carries `path`: the contract's value as a multiple of what was paid at every
close it was held, and `low_path`: the same at each day's worst stock price, so an account
simulation can mark positions and check an intraday drawdown floor.
"""
from math import erf, exp, log, sqrt

import numpy as np

from backtest import COST, atr

RATE = 0.04
MIN_VOL = 0.10


def _ncdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def price(kind, s, k, t, vol):
    if t <= 0:
        return max(s - k, 0.0) if kind == "C" else max(k - s, 0.0)
    d1 = (log(s / k) + (RATE + vol * vol / 2) * t) / (vol * sqrt(t))
    d2 = d1 - vol * sqrt(t)
    if kind == "C":
        return s * _ncdf(d1) - k * exp(-RATE * t) * _ncdf(d2)
    return k * exp(-RATE * t) * _ncdf(-d2) - s * _ncdf(-d1)


def delta(kind, s, k, t, vol):
    d1 = (log(s / k) + (RATE + vol * vol / 2) * t) / (vol * sqrt(t))
    return _ncdf(d1) if kind == "C" else _ncdf(d1) - 1


def pick_strike(kind, s, t, vol, target):
    """Listed-looking strike whose |delta| is closest to target (in the money for target > 0.5)."""
    step = 0.5 if s < 25 else 1 if s < 100 else 5 if s < 500 else 10
    sign = -1 if kind == "C" else 1  # ITM calls sit below spot, ITM puts above
    best, best_err = s, 9
    k = round(s / step) * step
    while 0.4 * s <= k <= 1.6 * s:
        err = abs(abs(delta(kind, s, k, t, vol)) - target)
        if err < best_err:
            best, best_err = k, err
        k += sign * step
    return best


def run(df, entry, exit_, kind="C", dte=90, max_hold=50, target_delta=0.70, iv_mult=1.0,
        half_spread=0.015, stop_atr=None, opt_stop=None, ticker=""):
    entry, exit_ = entry.fillna(False).values, exit_.fillna(False).values
    o, h, l, c = df.open.values, df.high.values, df.low.values, df.close.values
    a = atr(df).values
    rv = (np.log(df.close).diff().rolling(60).std() * sqrt(252)).values
    dates = df.index
    up = kind == "C"
    worst = l if up else h  # the day's most adverse stock price for this option

    trades = []
    pos = None
    pending_entry = pending_exit = False

    def vol_at(i):
        return max(rv[i], MIN_VOL) * iv_mult

    def value(i, s):
        return price(kind, s, pos["k"], max(dte - (i - pos["i"]), 0) / 252, vol_at(i - 1))

    def close(i, s, reason):
        nonlocal pos
        mid = value(i, s)
        stock = (s * (1 - COST) / (pos["px"] * (1 + COST)) - 1) if up else (pos["px"] * (1 - COST) / (s * (1 + COST)) - 1)
        trades.append({"ticker": ticker, "entry": dates[pos["i"]], "exit": dates[i], "days": i - pos["i"],
                       "spot": pos["px"], "strike": pos["k"], "premium": pos["cost"],
                       "stock_%": 100 * stock, "opt_%": 100 * (mid * (1 - half_spread) / pos["cost"] - 1),
                       "reason": reason, "path": pos["path"] + [mid * (1 - half_spread) / pos["cost"]],
                       "low_path": pos["low_path"]})
        pos = None

    for i in range(1, len(df)):
        if pending_exit and pos:
            close(i, o[i], pending_exit)
        elif pending_entry and not pos and not np.isnan(rv[i - 1]):
            t = dte / 252
            k = pick_strike(kind, o[i], t, vol_at(i - 1), target_delta)
            mid = price(kind, o[i], k, t, vol_at(i - 1))
            if mid > 0:
                stop = (o[i] - stop_atr * a[i - 1] if up else o[i] + stop_atr * a[i - 1]) if stop_atr else None
                pos = {"i": i, "px": o[i], "k": k, "cost": mid * (1 + half_spread), "stop": stop,
                       "path": [], "low_path": []}
        pending_entry = pending_exit = False
        if not pos:
            if entry[i]:
                pending_entry = True
            continue

        # intraday stops, checked against the day's worst stock price
        if pos["stop"] is not None and (l[i] <= pos["stop"] if up else h[i] >= pos["stop"]):
            fill = min(o[i], pos["stop"]) if up else max(o[i], pos["stop"])
            close(i, fill, "stock stop")
            continue
        low_val = value(i, worst[i]) * (1 - half_spread) / pos["cost"]
        if opt_stop and low_val <= 1 - opt_stop:
            open_val = value(i, o[i]) * (1 - half_spread) / pos["cost"]
            frac = min(open_val, 1 - opt_stop)
            pos["path"].append(frac)
            pos["low_path"].append(frac)
            trades.append(_stopped_at(pos, i, frac, dates, ticker, up, c[i]))
            pos = None
            continue

        pos["path"].append(value(i, c[i]) * (1 - half_spread) / pos["cost"])
        pos["low_path"].append(low_val)
        if pos["stop"] is not None:
            pos["stop"] = max(pos["stop"], c[i] - stop_atr * a[i]) if up else min(pos["stop"], c[i] + stop_atr * a[i])
        if exit_[i]:
            pending_exit = "signal"
        elif i - pos["i"] + 1 >= max_hold:
            pending_exit = "time"
    if pos:
        close(len(df) - 1, c[-1], "open")
    return trades


def _stopped_at(pos, i, frac, dates, ticker, up, close_px):
    """Option stop hit: the contract was sold at `frac` of its cost."""
    stock = close_px / pos["px"] - 1 if up else pos["px"] / close_px - 1
    return {"ticker": ticker, "entry": dates[pos["i"]], "exit": dates[i], "days": i - pos["i"],
            "spot": pos["px"], "strike": pos["k"], "premium": pos["cost"], "stock_%": 100 * stock,
            "opt_%": 100 * (frac - 1), "reason": "option stop", "path": pos["path"], "low_path": pos["low_path"]}

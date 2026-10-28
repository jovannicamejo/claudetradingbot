"""Which bearish signals, alone or behind a trend/market filter, make money as long puts?

Every signal is evaluated at a bar's close and filled at the NEXT bar's open. Each trade is
recorded two ways:
  * short_%: the stock move from entry to exit, as a short seller would see it (after COST).
  * put_%:   the return on a ~0.70-delta put bought PUT_DTE trading days from expiry, priced
             with Black-Scholes on trailing 60-day realized vol x IV_MULT, paying HALF_SPREAD
             on the way in and out. This is what a Strix account would actually have earned.

A trade ends on the signal's own exit, a protective stop STOP_ATR x ATR above the entry
(trailing down with the price), or MAX_HOLD trading days, whichever comes first. MAX_HOLD
is set so the put still has ~40 trading days left when it is sold.

Findings (120 tickers, Oct 2021 - Oct 2026, run 2026-10-28):
  * All 14 signals lose money as puts with no filter (avg -3% to -9% per trade), and the
    stock moves after them are ~0 or against the short, even for trades entered in 2022.
  * A 2 ATR stop is hit on ~2/3 of trades; wider stops or none help but do not create an edge.
  * The best of ~340 combinations (bearish engulfing near a 20d high while SPY < SMA50, no
    stop: 76 trades, +6.8% avg) has a t-stat of ~1.3 and rides a handful of weeks, so it is
    indistinguishable from luck. Treat bearish signals as a reason to skip calls, not to buy puts.

Usage:
    python bearish.py            # every ticker in data/ that is on the Strix whitelist
"""
import glob
from math import erf, exp, log, sqrt

import numpy as np
import pandas as pd

from backtest import COST, SPLIT_DATE, atr, ema, rsi, sma

PUT_DTE = 90          # trading days to expiry at entry (~4 calendar months)
MAX_HOLD = 50         # sell with ~40 trading days left, before time decay speeds up
TARGET_DELTA = -0.70
IV_MULT = 1.10        # puts usually trade above realized vol (skew)
HALF_SPREAD = 0.015   # per side, as a fraction of the option's mid
STOP_ATR = 2.0
RATE = 0.04
MIN_TRADES = 40       # pooled trades a combination needs before it is ranked

# --------------------------------------------------------------------------- #
# Bearish signals: each returns (entry, exit) boolean Series evaluated at the close.
# --------------------------------------------------------------------------- #
def cross_below(a, b):
    return (a < b) & (a.shift() >= b.shift())


def ma_cross_down(fast, slow, kind=sma):
    def f(df):
        a, b = kind(df.close, fast), kind(df.close, slow)
        return cross_below(a, b), cross_below(b, a)
    return f


def macd_cross_down(df):
    line = ema(df.close, 12) - ema(df.close, 26)
    sig = ema(line, 9)
    return cross_below(line, sig), cross_below(sig, line)


def macd_cross_down_below_zero(df):
    """Bearish MACD cross that happens while MACD is already negative (trend confirmation)."""
    line = ema(df.close, 12) - ema(df.close, 26)
    sig = ema(line, 9)
    return cross_below(line, sig) & (line < 0), cross_below(sig, line)


def donchian_breakdown(entry_n, exit_n):
    def f(df):
        lo = df.low.rolling(entry_n).min().shift()
        hi = df.high.rolling(exit_n).max().shift()
        return df.close < lo, df.close > hi
    return f


def sma50_break_on_volume(df):
    """First close below the 50-day SMA on 1.5x average volume; exit on a close back above it."""
    m = sma(df.close, 50)
    vol_ok = df.volume > 1.5 * df.volume.rolling(20).mean().shift()
    return cross_below(df.close, m) & vol_ok, df.close > m


def rsi_overbought_fade(df):
    """RSI14 drops back below 70 after being overbought; exit when RSI14 reaches 40."""
    r = rsi(df.close, 14)
    return cross_below(r, pd.Series(70, index=df.index)), r < 40


def bollinger_upper_reject(df):
    """Close back inside the upper band after closing above it; exit at the middle band."""
    mid = sma(df.close, 20)
    upper = mid + 2 * df.close.rolling(20).std()
    outside = df.close > upper
    return outside.shift().fillna(False) & ~outside, df.close < mid


def downtrend_bounce(df):
    """Mirror of the long trend pullback: short a short-term bounce (RSI3 > 80) below the 50 EMA."""
    trend = df.close < ema(df.close, 50)
    r3 = rsi(df.close, 3)
    return trend & (r3 > 80), (r3 < 30) | ~trend


def bear_flag_reject(df):
    """In a downtrend (EMA20 < SMA50), price rallies into the 20 EMA and closes back under it lower."""
    e20, m50 = ema(df.close, 20), sma(df.close, 50)
    tagged = df.high >= e20
    rejected = tagged & (df.close < e20) & (df.close < df.open) & (e20 < m50)
    return rejected, df.close > m50


def distribution_breakdown(df):
    """Close at a 20-day low on 2x average volume; exit on a close above the 10-day EMA."""
    lo = df.low.rolling(20).min().shift()
    vol_ok = df.volume > 2 * df.volume.rolling(20).mean().shift()
    return (df.close < lo) & vol_ok, df.close > ema(df.close, 10)


def bearish_engulfing_high(df):
    """Bearish engulfing candle within 2% of a 20-day high, RSI14 > 60; exit on RSI14 < 40."""
    prev_up = df.close.shift() > df.open.shift()
    engulf = (df.open >= df.close.shift()) & (df.close < df.open.shift()) & (df.close < df.open)
    near_high = df.high >= 0.98 * df.high.rolling(20).max()
    r = rsi(df.close, 14)
    return prev_up & engulf & near_high & (r > 60), r < 40


SIGNALS = {
    "Death cross SMA 50/200": ma_cross_down(50, 200),
    "SMA 20/50 cross down": ma_cross_down(20, 50),
    "EMA 9/21 cross down": ma_cross_down(9, 21, ema),
    "MACD cross down": macd_cross_down,
    "MACD cross down, below zero": macd_cross_down_below_zero,
    "Donchian 20/10 breakdown": donchian_breakdown(20, 10),
    "Donchian 55/20 breakdown": donchian_breakdown(55, 20),
    "SMA50 break on volume": sma50_break_on_volume,
    "RSI14 overbought fade": rsi_overbought_fade,
    "Bollinger upper reject": bollinger_upper_reject,
    "Downtrend bounce (EMA50+RSI3)": downtrend_bounce,
    "Bear flag reject at EMA20": bear_flag_reject,
    "Distribution breakdown (20d lo, 2x vol)": distribution_breakdown,
    "Bearish engulfing near high": bearish_engulfing_high,
}

# --------------------------------------------------------------------------- #
# Filters: a boolean Series that must be True at the signal's close. They only gate
# entries; exits are unchanged.
# --------------------------------------------------------------------------- #
spy = pd.read_csv("data/SPY_daily.csv", parse_dates=["date"]).set_index("date").close
vix = pd.read_csv("data/VIX_daily.csv", parse_dates=["date"]).set_index("date").close


def _align(s, idx):
    return s.reindex(idx).ffill()


FILTERS = {
    "none": lambda df: pd.Series(True, index=df.index),
    "stock < SMA200": lambda df: df.close < sma(df.close, 200),
    "SPY < SMA50": lambda df: _align(spy < sma(spy, 50), df.index),
    "SPY < SMA200": lambda df: _align(spy < sma(spy, 200), df.index),
    "weak vs SPY (63d)": lambda df: df.close.pct_change(63) < _align(spy.pct_change(63), df.index),
    "VIX >= 20": lambda df: _align(vix >= 20, df.index),
    "stock < SMA200 + weak vs SPY": lambda df: (df.close < sma(df.close, 200))
    & (df.close.pct_change(63) < _align(spy.pct_change(63), df.index)),
    "stock < SMA200 + SPY < SMA50": lambda df: (df.close < sma(df.close, 200))
    & _align(spy < sma(spy, 50), df.index),
}


# --------------------------------------------------------------------------- #
# Put pricing
# --------------------------------------------------------------------------- #
def _ncdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def put_price(s, k, t, vol):
    if t <= 0:
        return max(k - s, 0.0)
    d1 = (log(s / k) + (RATE + vol * vol / 2) * t) / (vol * sqrt(t))
    d2 = d1 - vol * sqrt(t)
    return k * exp(-RATE * t) * _ncdf(-d2) - s * _ncdf(-d1)


def put_delta(s, k, t, vol):
    d1 = (log(s / k) + (RATE + vol * vol / 2) * t) / (vol * sqrt(t))
    return _ncdf(d1) - 1


def pick_strike(s, t, vol):
    """Strike (as a listed-looking round number) whose put delta is closest to TARGET_DELTA."""
    step = 0.5 if s < 25 else 1 if s < 100 else 5 if s < 500 else 10
    best, best_err = s, 9
    k = round(s / step) * step
    while k <= 1.6 * s:
        err = abs(put_delta(s, k, t, vol) - TARGET_DELTA)
        if err < best_err:
            best, best_err = k, err
        k += step
    return best


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
def run_short(df, entry, exit_, ticker):
    entry, exit_ = entry.fillna(False).values, exit_.fillna(False).values
    o, h, c = df.open.values, df.high.values, df.close.values
    a = atr(df).values
    rv = (np.log(df.close).diff().rolling(60).std() * sqrt(252)).values
    dates = df.index
    trades = []
    in_pos = pending_entry = pending_exit = False
    for i in range(1, len(df)):
        if pending_exit and in_pos:
            trades.append(_close(pos, i, o[i], rv[i - 1], dates, ticker))
            in_pos = False
        elif pending_entry and not in_pos and not np.isnan(rv[i - 1]):
            vol = max(rv[i - 1], 0.10) * IV_MULT
            k = pick_strike(o[i], PUT_DTE / 252, vol)
            mid = put_price(o[i], k, PUT_DTE / 252, vol)
            pos = {"i": i, "px": o[i], "k": k, "cost": mid * (1 + HALF_SPREAD),
                   "stop": o[i] + STOP_ATR * a[i - 1]}
            in_pos = True
        pending_entry = pending_exit = False

        if in_pos and h[i] >= pos["stop"]:  # gap-up fills at the open
            trades.append(_close(pos, i, max(o[i], pos["stop"]), rv[i - 1], dates, ticker, stopped=True))
            in_pos = False
            continue
        if in_pos:
            pos["stop"] = min(pos["stop"], c[i] + STOP_ATR * a[i])
            if exit_[i] or i - pos["i"] + 1 >= MAX_HOLD:
                pending_exit = True
        elif entry[i]:
            pending_entry = True
    if in_pos:
        trades.append(_close(pos, len(df) - 1, c[-1], rv[-1], dates, ticker))
    return trades


def _close(pos, i, px, vol, dates, ticker, stopped=False):
    t_left = max(PUT_DTE - (i - pos["i"]), 0) / 252
    mid = put_price(px, pos["k"], t_left, max(vol, 0.10) * IV_MULT)
    return {"ticker": ticker, "entry": dates[pos["i"]], "exit": dates[i], "days": i - pos["i"],
            "short_%": 100 * (pos["px"] * (1 - COST) / (px * (1 + COST)) - 1),
            "put_%": 100 * (mid * (1 - HALF_SPREAD) / pos["cost"] - 1), "stopped": stopped}


def summarize(t):
    p = t["put_%"]
    wins, losses = p[p > 0].sum(), -p[p < 0].sum()
    by_year = t.groupby(t.entry.dt.year)["put_%"].mean()
    early, late = t[t.entry < SPLIT_DATE]["put_%"], t[t.entry >= SPLIT_DATE]["put_%"]
    return {"trades": len(t), "tickers": t.ticker.nunique(), "win_%": 100 * (p > 0).mean(),
            "avg_put_%": p.mean(), "median_put_%": p.median(), "put_PF": wins / losses if losses else np.inf,
            "avg_short_%": t["short_%"].mean(), "avg_days": t.days.mean(),
            "pre2025_put_%": early.mean(), "2025+_put_%": late.mean(), "2025+_trades": len(late),
            "yrs_positive": f"{(by_year > 0).sum()}/{len(by_year)}"}


def load_universe():
    wl = {w for line in open("strix_whitelist.txt") if not line.startswith("#") for w in line.split()}
    out = {}
    for path in sorted(glob.glob("data/*_daily.csv")):
        t = path.split("/")[-1].split("_")[0]
        if t in wl and t != "VIX":
            out[t] = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    return out


def main():
    universe = load_universe()
    print(f"{len(universe)} whitelisted tickers; put ~{abs(TARGET_DELTA):.2f} delta, {PUT_DTE}d to expiry, "
          f"max hold {MAX_HOLD}d, stop {STOP_ATR} ATR, IV = RV60 x {IV_MULT}, half-spread {HALF_SPREAD:.1%}\n")
    all_trades = []
    for sname, sig in SIGNALS.items():
        for fname, filt in FILTERS.items():
            for t, df in universe.items():
                entry, exit_ = sig(df)
                for tr in run_short(df, entry & filt(df), exit_, t):
                    all_trades.append({"signal": sname, "filter": fname, **tr})
    trades = pd.DataFrame(all_trades)
    trades.to_csv("bearish_trades.csv", index=False)

    rows = {k: summarize(g) for k, g in trades.groupby(["signal", "filter"])}
    res = pd.DataFrame(rows).T
    res.index.names = ["signal", "filter"]
    res = res.infer_objects()
    res.to_csv("bearish_results.csv")
    pd.set_option("display.width", 250, "display.float_format", "{:,.1f}".format)

    print("=== Each signal with no filter (pooled across tickers) ===")
    print(res.xs("none", level="filter").sort_values("avg_put_%", ascending=False).to_string(), "\n")

    ranked = res[res.trades >= MIN_TRADES].sort_values("avg_put_%", ascending=False)
    print(f"=== Top 20 signal + filter combinations (>= {MIN_TRADES} trades) ===")
    print(ranked.head(20).to_string(), "\n")

    print("=== Average put return % by filter (rows) and signal (columns) ===")
    grid = res["avg_put_%"].unstack("signal")
    print(grid.round(1).T.to_string())


if __name__ == "__main__":
    main()

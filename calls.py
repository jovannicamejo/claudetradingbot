"""Which bullish signals, filters and stops make money as 2-3 month calls?

Every long strategy from backtest.py is traded as a ~0.70-delta call bought 90 trading days
from expiry and sold after at most 50 (see optsim.py for pricing and fills), on every ticker
in data/ that is on the Strix whitelist. Each signal is run behind several entry filters and
with several stop rules.

Combinations are ranked on trades entered before SPLIT_DATE only; the 2025+ columns are
out of sample. Trades entered the same week are mostly one bet on the market, so the
t-stat is computed on weekly average returns, not on single trades.

Usage:
    python calls.py              # writes calls_results.csv and calls_trades.pkl
"""
import glob
from multiprocessing import Pool

import numpy as np
import pandas as pd

import optsim
from backtest import SPLIT_DATE, STRATEGIES, ema, sma
from bearish import bearish_engulfing_high
from vix_filter import TREND

DTE, MAX_HOLD, DELTA, HALF_SPREAD = 90, 50, 0.70, 0.015
STOPS = {"none": {}, "2.5 ATR": {"stop_atr": 2.5}, "option -35%": {"opt_stop": 0.35},
         "option -50%": {"opt_stop": 0.50}}
SKIP = {"Buy & hold", "EMA 9/21 + 2.5 ATR trail", "Bollinger 20/2 + 3 ATR stop", "Donchian 20/10 + 2 ATR trail",
        "Vol breakout + 2 ATR trail"}  # same entries as other rows; stops are tested separately here
MIN_TRADES = 60

spy = pd.read_csv("data/SPY_daily.csv", parse_dates=["date"]).set_index("date").close
vix = pd.read_csv("data/VIX_daily.csv", parse_dates=["date"]).set_index("date").close
vix_spike = (vix > 1.25 * vix.rolling(20).mean()).rolling(5).max().astype(bool)


def _align(s, idx):
    return s.reindex(idx).ffill().fillna(False).astype(bool)


def vix_regime(name, idx):
    """The regime vix_filter.py found better for this strategy family."""
    good = vix < 20 if name in TREND else (vix >= 20) | vix_spike
    return _align(good, idx)


def no_bear_warning(df):
    """SPY above its 50-day SMA and no bearish engulfing candle near a high in the last 5 days."""
    warn = bearish_engulfing_high(df)[0].fillna(False).rolling(5).max().astype(bool)
    return _align(spy > sma(spy, 50), df.index) & ~warn


def strong_vs_spy(df):
    return df.close.pct_change(63) > spy.pct_change(63).reindex(df.index).ffill()


FILTERS = {
    "none": lambda df, n: pd.Series(True, index=df.index),
    "VIX regime": lambda df, n: vix_regime(n, df.index),
    "SPY > SMA50": lambda df, n: _align(spy > sma(spy, 50), df.index),
    "SPY > SMA200": lambda df, n: _align(spy > sma(spy, 200), df.index),
    "no bear warning": lambda df, n: no_bear_warning(df),
    "stock > SMA200": lambda df, n: df.close > sma(df.close, 200),
    "strong vs SPY": lambda df, n: strong_vs_spy(df),
    "stock > SMA200 + strong vs SPY": lambda df, n: (df.close > sma(df.close, 200)) & strong_vs_spy(df),
    "stock > SMA200 + SPY > SMA50": lambda df, n: (df.close > sma(df.close, 200))
    & _align(spy > sma(spy, 50), df.index),
    "stock > SMA200 + strong + VIX regime": lambda df, n: (df.close > sma(df.close, 200)) & strong_vs_spy(df)
    & vix_regime(n, df.index),
}


def load_universe():
    wl = {w for line in open("strix_whitelist.txt") if not line.startswith("#") for w in line.split()}
    return {p.split("/")[-1].split("_")[0]: p for p in sorted(glob.glob("data/*_daily.csv"))
            if p.split("/")[-1].split("_")[0] in wl - {"VIX"}}


def ticker_trades(args):
    ticker, path = args
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    out = []
    for sname, (strat, kw) in STRATEGIES.items():
        if sname in SKIP:
            continue
        entry, exit_ = strat(df)
        hold = min(MAX_HOLD, kw.get("max_hold") or MAX_HOLD)
        for fname, filt in FILTERS.items():
            gated = entry & filt(df, sname)
            for stop, skw in STOPS.items():
                for t in optsim.run(df, gated, exit_, kind="C", dte=DTE, max_hold=hold, target_delta=DELTA,
                                    half_spread=HALF_SPREAD, ticker=ticker, **skw):
                    out.append({"signal": sname, "filter": fname, "stop": stop, **t})
    return out


def summarize(t):
    p = t["opt_%"]
    wk = t.groupby(t.entry.dt.to_period("W"))["opt_%"].mean()
    early, late = t[t.entry < SPLIT_DATE], t[t.entry >= SPLIT_DATE]
    wins, losses = p[p > 0].sum(), -p[p < 0].sum()
    by_year = t.groupby(t.entry.dt.year)["opt_%"].mean()
    return {"trades": len(t), "tickers": t.ticker.nunique(), "win_%": 100 * (p > 0).mean(), "avg_%": p.mean(),
            "PF": wins / losses if losses else np.inf, "week_t": wk.mean() / wk.std() * np.sqrt(len(wk)) if len(wk) > 2 else 0,
            "avg_days": t.days.mean(), "pre_trades": len(early), "pre_avg_%": early["opt_%"].mean(),
            "pre_PF": early["opt_%"].clip(lower=0).sum() / -early["opt_%"].clip(upper=0).sum() if len(early) else np.nan,
            "post_trades": len(late), "post_avg_%": late["opt_%"].mean(),
            "post_PF": late["opt_%"].clip(lower=0).sum() / -late["opt_%"].clip(upper=0).sum() if len(late) else np.nan,
            "yrs_pos": f"{(by_year > 0).sum()}/{len(by_year)}"}


def main():
    universe = load_universe()
    with Pool() as pool:
        rows = [t for chunk in pool.map(ticker_trades, universe.items()) for t in chunk]
    trades = pd.DataFrame(rows)
    trades.to_pickle("calls_trades.pkl")

    res = pd.DataFrame({k: summarize(g) for k, g in trades.groupby(["signal", "filter", "stop"])}).T.infer_objects()
    res.index.names = ["signal", "filter", "stop"]
    res.to_csv("calls_results.csv")
    pd.set_option("display.width", 260, "display.float_format", "{:,.2f}".format, "display.max_colwidth", 40)
    print(f"{len(universe)} tickers, {len(trades):,} trades; call ~{DELTA} delta, {DTE}d to expiry, max hold {MAX_HOLD}d\n")

    print("=== Each signal, no filter, no stop ===")
    base = res.xs(("none", "none"), level=("filter", "stop"))
    print(base.sort_values("avg_%", ascending=False).to_string(), "\n")

    print("=== Average return % by stop rule (all signals and filters pooled) ===")
    print(trades.groupby("stop")["opt_%"].agg(["size", "mean", "median"]).to_string(), "\n")

    print("=== Average return % by filter (all signals, no stop) ===")
    ns = trades[trades.stop == "none"]
    print(ns.groupby("filter")["opt_%"].agg(["size", "mean", "median"]).sort_values("mean", ascending=False).to_string(), "\n")

    ranked = res[(res.pre_trades >= MIN_TRADES) & (res.post_trades >= 20)].sort_values("pre_avg_%", ascending=False)
    print(f"=== Top 25 combinations ranked on pre-{SPLIT_DATE} trades (2025+ is out of sample) ===")
    print(ranked.head(25).to_string())


if __name__ == "__main__":
    main()

"""Does the VIX improve the swing strategies?

Each strategy from backtest.py is re-run with an extra entry condition based on
the VIX (exits are unchanged), and the result is compared with the unfiltered
strategy on every ticker in data/. VIX values are taken from the prior close
(no look-ahead): a signal at today's close only uses today's VIX close.

Usage:
    python vix_filter.py
"""
import glob

import pandas as pd

from backtest import STRATEGIES, run, stats

TREND = {"SMA 10/30 cross", "SMA 20/50 cross", "EMA 9/21 cross", "EMA 9/21 + 2.5 ATR trail", "MACD 12/26/9",
         "Donchian 20/10 breakout", "Donchian 20/10 + 2 ATR trail", "Turtle 55/20 breakout",
         "Volume breakout (20d hi, 2x vol)", "Vol breakout + 2 ATR trail"}
SKIP = {"Buy & hold"}
EXCLUDE = {"VIX", "QQQ", "XLK", "SMH", "SPY"}

vix = pd.read_csv("data/VIX_daily.csv", parse_dates=["date"]).set_index("date").close
FILTERS = {
    "VIX < 20 (calm)": lambda v: v < 20,
    "VIX >= 20 (fearful)": lambda v: v >= 20,
    "VIX falling (below 10-day avg)": lambda v: v < v.rolling(10).mean(),
    "VIX rising (above 10-day avg)": lambda v: v > v.rolling(10).mean(),
    "VIX spike (>= 25% above 20-day avg)": lambda v: v > 1.25 * v.rolling(20).mean(),
}


def filtered(strat, cond):
    def f(df):
        entry, exit_ = strat(df)
        c = cond.reindex(df.index).ffill().fillna(False).astype(bool)
        return entry & c, exit_
    return f


def main():
    rows = []
    tickers = sorted(p.split("/")[-1].split("_")[0] for p in glob.glob("data/*_daily.csv"))
    for t in [t for t in tickers if t not in EXCLUDE]:
        df = pd.read_csv(f"data/{t}_daily.csv", parse_dates=["date"]).set_index("date")
        v = vix.reindex(df.index).ffill()
        for name, (strat, kw) in STRATEGIES.items():
            if name in SKIP:
                continue
            base = stats(*run(df, strat, **kw))
            fam = "Trend" if name in TREND else "Dip-buy"
            rows.append({"ticker": t, "strategy": name, "family": fam, "filter": "None", "ret": base["total_%"],
                         "dd": base["max_dd_%"], "trades": base["trades"]})
            for fname, f in FILTERS.items():
                r = stats(*run(df, filtered(strat, f(v)), **kw))
                rows.append({"ticker": t, "strategy": name, "family": fam, "filter": fname, "ret": r["total_%"],
                             "dd": r["max_dd_%"], "trades": r["trades"]})
    res = pd.DataFrame(rows)
    base = res[res["filter"] == "None"].set_index(["ticker", "strategy"])
    res = res.join(base[["ret", "dd"]], on=["ticker", "strategy"], rsuffix="_base")
    res["d_ret"] = res.ret - res.ret_base
    res["d_dd"] = res.dd - res.dd_base  # positive = shallower drawdown
    res.to_csv("vix_results.csv", index=False)
    f = res[res["filter"] != "None"]
    pd.set_option("display.width", 200, "display.float_format", "{:,.1f}".format)
    print("=== Median change vs. no filter, by strategy family (pct points) ===")
    g = f.groupby(["family", "filter"]).agg(median_d_ret=("d_ret", "median"), share_improved=("d_ret", lambda s: 100 * (s > 0).mean()),
                                           median_d_dd=("d_dd", "median"), median_trades=("trades", "median"))
    print(g.to_string())


if __name__ == "__main__":
    main()


def per_trade(n_placebo=20, seed=0):
    """Fairer test: average return per trade, vs a random filter that keeps the same share of entry days."""
    import numpy as np
    rng = np.random.default_rng(seed)
    out = []
    tickers = sorted(p.split("/")[-1].split("_")[0] for p in glob.glob("data/*_daily.csv"))
    for t in [t for t in tickers if t not in EXCLUDE]:
        df = pd.read_csv(f"data/{t}_daily.csv", parse_dates=["date"]).set_index("date")
        v = vix.reindex(df.index).ffill()
        for name, (strat, kw) in STRATEGIES.items():
            if name in SKIP:
                continue
            fam = "Trend" if name in TREND else "Dip-buy"
            for fname, f in FILTERS.items():
                cond = f(v).fillna(False).astype(bool)
                _, tr = run(df, filtered(strat, cond), **kw)
                if len(tr) < 3:
                    continue
                keep = cond.mean()
                plac = []
                for _ in range(n_placebo):
                    rc = pd.Series(rng.random(len(df)) < keep, index=df.index)
                    _, tp = run(df, filtered(strat, rc), **kw)
                    if tp:
                        plac.append(np.mean(tp))
                out.append({"ticker": t, "strategy": name, "family": fam, "filter": fname,
                            "avg_trade": 100 * np.mean(tr), "placebo_avg_trade": 100 * np.mean(plac) if plac else np.nan,
                            "trades": len(tr)})
    res = pd.DataFrame(out)
    res["edge"] = res.avg_trade - res.placebo_avg_trade
    print("\n=== Avg return per trade with the VIX filter vs. a random filter of the same strictness ===")
    print(res.groupby(["family", "filter"]).agg(avg_trade=("avg_trade", "median"), random_filter=("placebo_avg_trade", "median"),
                                                 edge=("edge", "median"), beats_random_pct=("edge", lambda s: 100 * (s > 0).mean()),
                                                 cases=("edge", "size")).to_string())
    return res

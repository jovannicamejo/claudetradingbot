"""Does buying a setup a few sessions after it fired still work?

The scan lists setups that fired up to 5 sessions ago. This re-runs the three strix_sim.py
setups with the entry delayed by k sessions, taken only if the setup has not exited in the
meantime and the close has not run more than 1 ATR past the signal close (the scan's rule),
with the same contract as strix_sim.py (DTE, DELTA, MAX_HOLD from calls.py, OPT_STOP).

These are modeled option returns (see strix_rules.py's pricing caveat) under the EARLIER rule
set that holds through earnings; they do not include the strix-v1 earnings exits, so they do
not validate the live rules, and a group average is not a forecast for any one trade.

Findings (run 2026-10-28): in the modeled sample, every setup stays positive with entries up to 5 sessions late. The
RSI14 dip-buy's 2025+ average falls from ~30% per call at 0-2 sessions to ~15% at 4-5; its
pre-2025 average barely moves (16-21%). The SMA 20/50 cross's 2025+ average also drops, from
~23% at 0-1 sessions to 8-15% at 3-5; the Turtle breakout changes little in either period.

Usage:
    python entry_delay.py
"""
from multiprocessing import Pool

import pandas as pd

import optsim
from backtest import SPLIT_DATE, STRATEGIES, atr
from calls import DELTA, DTE, FILTERS, MAX_HOLD, load_universe
from strix_sim import OPT_STOP, SETUPS

DELAYS = range(0, 6)


def ticker_trades(args):
    ticker, path = args
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    a = atr(df)
    out = []
    for sname, fname in SETUPS:
        strat, kw = STRATEGIES[sname]
        entry, exit_ = strat(df)
        gated = (entry & FILTERS[fname](df, sname)).fillna(False)
        hold = min(MAX_HOLD, kw.get("max_hold") or MAX_HOLD)
        for k in DELAYS:
            if k == 0:
                late = gated
            else:
                exited = exit_.fillna(False).astype(int).rolling(k).max().astype(bool)
                late = gated.shift(k, fill_value=False) & ~exited & (df.close - df.close.shift(k) <= a)
            for t in optsim.run(df, late, exit_, kind="C", dte=DTE, max_hold=hold, target_delta=DELTA,
                                opt_stop=OPT_STOP, ticker=ticker):
                out.append({"setup": sname, "delay": k, "entry": t["entry"], "opt_%": t["opt_%"]})
    return out


def main():
    with Pool() as pool:
        rows = [t for c in pool.map(ticker_trades, load_universe().items()) for t in c]
    t = pd.DataFrame(rows)
    t["period"] = (t.entry >= SPLIT_DATE).map({False: "pre-2025", True: "2025+"})
    pd.set_option("display.width", 200, "display.float_format", "{:,.1f}".format)
    print("Average return % per call by entry delay (sessions after the signal)")
    print(t.pivot_table(index=["setup", "period"], columns="delay", values="opt_%", aggfunc="mean").to_string())
    print("\ntrades")
    print(t.pivot_table(index="setup", columns="delay", values="opt_%", aggfunc="size").to_string())


if __name__ == "__main__":
    main()

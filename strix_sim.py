"""Would the best call setups have passed a Strix 50K evaluation?

Account rules (Strix 50K, https://strixoptionsfunding.com/rules, 2026-10-28):
  * start $50,000, pass at $56,000 (12%), no deadline
  * trailing drawdown $3,000 below the highest END-OF-DAY equity, but a breach counts the
    moment equity touches the floor intraday. Positions are marked at each day's worst
    stock price to check this.
  * single-leg long calls/puts only, whitelisted tickers only

Sizing: RISK_PER_TRADE is a hard cap on a trade's PLANNED loss at its option stop (contracts =
floor(RISK_PER_TRADE / (OPT_STOP x premium x 100)); a trade where even one contract exceeds it
is skipped, so at a 50% stop the most a $250 trade can pay is $500). New trades are only taken
while the planned loss of all open positions stays at or below MAX_OPEN_RISK, and only one
position per ticker. Planned loss is not maximum loss: a stop can fill worse than its level
(the simulation fills gaps at the open), and the full premium is what is actually exposed.
Positions in the same sector are NOT limited, so correlated signals (e.g. several energy names)
can all be taken; that is a known gap.

An evaluation is started on the first trading day of every month and run until it passes,
fails, or the data ends.

Findings (58 monthly starts, Jan 2022 - Oct 2026, re-run 2026-10-28 with the $250 hard cap;
starts overlap, so they are not independent samples, and single runs are fragile - see
strix_research.py):
  * $250 hard cap / $750 open: 41-57% passed, 2-16% failed, the rest still running when the data
    ends, across the four pricing scenarios. The median pass took ~300-380 trading days.
  * The earlier $600 / $1,500 sizing failed ~57% of evaluations; the drawdown floor, not the
    edge, is the limit.
  * Taking profit at +100% did not help consistently.

What this backtest is and is not:
  * Option prices are Black-Scholes on trailing realized vol, not historical option quotes.
  * Signals were chosen in calls.py from ~480 signal/filter/stop combinations on pre-2025 trades,
    with 2025+ held out; the sizing/expiry research in strix_research.py used the full period.
  * Entries fill at the next open after the signal; entry_delay.py covers later entries.
  * No sector limit, no earnings exits (no earnings history), no early-session data gaps.

Usage:
    python strix_sim.py
"""
from multiprocessing import Pool

import numpy as np
import pandas as pd

import optsim
from backtest import STRATEGIES
from calls import DELTA, DTE, FILTERS, MAX_HOLD, load_universe

START, TARGET, DRAWDOWN = 50_000, 56_000, 3_000
RISK_PER_TRADE, MAX_OPEN_RISK = 250, 750
OPT_STOP = 0.50

# (signal, filter), in priority order when several fire on the same day. Chosen on
# pre-2025 trades by calls.py.
SETUPS = [
    ("RSI14 <30 -> >70", "VIX regime"),
    ("SMA 20/50 cross", "stock > SMA200 + strong vs SPY"),
    ("Turtle 55/20 breakout", "SPY > SMA200"),
]
STRESS = {"base": (1.0, 0.015), "IV x1.25": (1.25, 0.015), "spread 3%": (1.0, 0.03), "IV x1.25 + spread 3%": (1.25, 0.03)}


def setup_trades(args):
    ticker, path, iv_mult, spread = args
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    out = []
    for prio, (sname, fname) in enumerate(SETUPS):
        strat, kw = STRATEGIES[sname]
        entry, exit_ = strat(df)
        hold = min(MAX_HOLD, kw.get("max_hold") or MAX_HOLD)
        for t in optsim.run(df, entry & FILTERS[fname](df, sname), exit_, kind="C", dte=DTE, max_hold=hold,
                            target_delta=DELTA, iv_mult=iv_mult, half_spread=spread, opt_stop=OPT_STOP, ticker=ticker):
            out.append({"setup": sname, "prio": prio, **t})
    return out


def simulate(trades, days, start):
    """One evaluation from `start`. Returns (outcome, trading days used, trades taken)."""
    pos = {d: i for i, d in enumerate(days)}
    todo = trades[trades.entry >= start].sort_values(["entry", "prio", "premium"])
    by_day = {d: g for d, g in todo.groupby("entry")}
    cash, peak, open_, taken = START, START, [], 0
    for di in range(pos[start], len(days)):
        d = days[di]
        # exits first: positions whose last mark is today are sold
        still = []
        for p in open_:
            if di - p["i0"] >= len(p["path"]) - 1:
                cash += p["n"] * 100 * p["cost"] * p["path"][-1]
            else:
                still.append(p)
        open_ = still
        for _, t in by_day.get(d, pd.DataFrame()).iterrows():
            risk_one = OPT_STOP * t.premium * 100
            if risk_one > RISK_PER_TRADE or any(p["ticker"] == t.ticker for p in open_):
                continue
            n = int(RISK_PER_TRADE // risk_one)
            if sum(p["risk"] for p in open_) + n * risk_one > MAX_OPEN_RISK:
                continue
            cash -= n * 100 * t.premium
            open_.append({"ticker": t.ticker, "i0": di, "n": n, "cost": t.premium, "risk": n * risk_one,
                          "path": t.path, "low": t.low_path})
            taken += 1
        # marks: worst intraday, then close
        low_eq = cash + sum(p["n"] * 100 * p["cost"] * (p["low"][di - p["i0"]] if di - p["i0"] < len(p["low"])
                                                       else p["path"][di - p["i0"]]) for p in open_)
        eq = cash + sum(p["n"] * 100 * p["cost"] * p["path"][min(di - p["i0"], len(p["path"]) - 1)] for p in open_)
        if low_eq <= peak - DRAWDOWN:
            return "fail", di - pos[start], taken
        if eq >= TARGET:
            return "pass", di - pos[start], taken
        peak = max(peak, eq)
    return "open", len(days) - 1 - pos[start], taken


def main():
    universe = load_universe()
    days = pd.read_csv("data/SPY_daily.csv", parse_dates=["date"]).date
    starts = days[(days >= "2022-01-01")].groupby(days.dt.to_period("M")).first().tolist()
    days = list(days)
    pd.set_option("display.width", 220, "display.float_format", "{:,.1f}".format)
    for label, (iv, spread) in STRESS.items():
        with Pool() as pool:
            rows = [t for c in pool.map(setup_trades, [(t, p, iv, spread) for t, p in universe.items()]) for t in c]
        trades = pd.DataFrame(rows)
        # align every trade's marks to the SPY calendar (tickers share NYSE days)
        trades = trades[trades.entry.isin(days)]
        per = trades.groupby("setup")["opt_%"].agg(["size", "mean"]).rename(columns={"size": "trades", "mean": "avg_%"})
        res = pd.DataFrame([dict(zip(("outcome", "days", "taken"), simulate(trades, days, s)), start=s.date())
                            for s in starts])
        n = len(res)
        passed = res[res.outcome == "pass"]
        print(f"=== {label}: IV = RV60 x {iv}, half-spread {spread:.1%} ===")
        print(per.to_string())
        print(f"evaluations started: {n} | passed {len(passed)} ({100 * len(passed) / n:.0f}%) | "
              f"failed {(res.outcome == 'fail').sum()} | still open {(res.outcome == 'open').sum()}")
        if len(passed):
            print(f"trading days to pass: median {passed.days.median():.0f}, "
                  f"range {passed.days.min()}-{passed.days.max()}; trades taken median {passed.taken.median():.0f}")
        print(res.set_index("start").T.to_string(), "\n")


if __name__ == "__main__":
    main()

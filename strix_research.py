"""Which expiry, strike, contract price and sizing pass a Strix 50K evaluation fastest and safest?

Uses the three setups and the account rules of strix_sim.py, and varies:
  * months to expiry at entry (21 trading days per month); a contract is sold after at most
    60% of its life so it never sits in the steepest part of time decay,
  * strike: ~0.50 delta (at the money, cheaper) or ~0.70 delta (in the money),
  * risk per trade at the option stop (max open risk = 3 trades' worth),
  * the option stop itself (a tighter stop lets the same risk buy a pricier contract).

Part 1 is per trade, split by what one contract cost. Part 2 replays an evaluation from the
first trading day of every month since Jan 2022 and reports how many passed within 6 months
(126 trading days), failed, or passed at any point.

Single runs are fragile: moving the expiry by 3 trading days or the max hold by 5 changed the
fail rate of one configuration from 5% to 53%, because a $3,000 floor turns on which handful of
trades happen to fit. So every configuration in Part 2 is the average of 9 nearby variants
(expiry -3/0/+3 trading days x max hold 55/60/65% of the contract's life).

Findings (run 2026-10-28, 58 monthly starts Jan 2022 - Oct 2026):
  * Per trade, 3-6 month contracts beat 1-2 month ones, and contracts under $1,500 beat pricier
    ones by a wide margin (e.g. 4 months, 0.70 delta: +22-27% vs +9-18% per trade).
  * No configuration passed within 6 months more than ~29% of the time, and every one that got
    near that failed 57-73% of evaluations. Faster passing always came with more failures.
  * The lowest-failure setting (6 months, 0.50 delta, -50% option stop, $250 risk) failed ~21%
    and passed ~56% eventually, with a median of ~350 trading days.
  * Bigger risk per trade and a tighter -35% stop both raise the 6-month pass rate a little and
    the failure rate a lot.

Usage:
    python strix_research.py
"""
from multiprocessing import Pool

import numpy as np
import pandas as pd

import optsim
import strix_sim
from backtest import STRATEGIES
from calls import FILTERS, load_universe

MONTHS = [1, 2, 3, 4, 6]
DELTAS = [0.50, 0.70]
RISKS = [250, 400, 600]
STOPS = [0.35, 0.50]
JITTER = [(d, h) for d in (-3, 0, 3) for h in (0.55, 0.60, 0.65)]
SIX_MONTHS = 126
PRICE_BANDS = [0, 600, 1500, 3000, np.inf]


def gen(args):
    ticker, path, months, dlt, stop, iv, spread, jitter = args
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    dte = 21 * months + jitter[0]
    out = []
    for prio, (sname, fname) in enumerate(strix_sim.SETUPS):
        strat, kw = STRATEGIES[sname]
        entry, exit_ = strat(df)
        hold = min(round(jitter[1] * dte), kw.get("max_hold") or dte)
        for t in optsim.run(df, entry & FILTERS[fname](df, sname), exit_, kind="C", dte=dte, max_hold=hold,
                            target_delta=dlt, iv_mult=iv, half_spread=spread, opt_stop=stop, ticker=ticker):
            out.append({"setup": sname, "prio": prio, **t})
    return out


def trades_for(universe, days, pool, months, dlt, stop, iv=1.0, spread=0.015, jitter=(0, 0.60)):
    rows = [t for c in pool.map(gen, [(t, p, months, dlt, stop, iv, spread, jitter) for t, p in universe.items()])
            for t in c]
    d = pd.DataFrame(rows)
    return d[d.entry.isin(days)]


def evaluate(trades, days, starts, risk, stop):
    strix_sim.OPT_STOP = stop
    strix_sim.RISK_PER_TRADE, strix_sim.MAX_OPEN_RISK = risk, 3 * risk
    res = pd.DataFrame([strix_sim.simulate(trades, days, s) for s in starts], columns=["outcome", "days", "taken"])
    passed = res[res.outcome == "pass"]
    fast = (passed.days <= SIX_MONTHS).sum()
    early_fail = ((res.outcome == "fail") & (res.days <= SIX_MONTHS)).sum()
    return {"pass_6mo_%": 100 * fast / len(res), "fail_6mo_%": 100 * early_fail / len(res),
            "pass_ever_%": 100 * len(passed) / len(res), "fail_ever_%": 100 * (res.outcome == "fail").mean(),
            "med_days_to_pass": passed.days.median(), "max_contract_$": risk / stop,
            "trades_per_6mo": res.taken.mean() * SIX_MONTHS / max(res.days.mean(), 1)}


def main():
    universe = load_universe()
    days = pd.read_csv("data/SPY_daily.csv", parse_dates=["date"]).date
    starts = days[days >= "2022-01-01"].groupby(days.dt.to_period("M")).first().tolist()
    days = list(days)
    pd.set_option("display.width", 220, "display.float_format", "{:,.1f}".format)
    per_trade, grid, cache = [], [], {}
    with Pool() as pool:
        for months in MONTHS:
            for dlt in DELTAS:
                for stop in STOPS:
                    tr = trades_for(universe, days, pool, months, dlt, stop)
                    cache[(months, dlt, stop)] = tr
                    band = pd.cut(tr.premium * 100, PRICE_BANDS, labels=["<$600", "$600-1.5k", "$1.5k-3k", ">$3k"])
                    for b, g in tr.groupby(band, observed=True):
                        p = g["opt_%"]
                        per_trade.append({"months": months, "delta": dlt, "stop": f"-{stop:.0%}", "contract": b,
                                          "trades": len(g), "win_%": 100 * (p > 0).mean(), "avg_%": p.mean(),
                                          "PF": p.clip(lower=0).sum() / -p.clip(upper=0).sum()})
                    variants = [tr] + [trades_for(universe, days, pool, months, dlt, stop, jitter=j)
                                       for j in JITTER if j != (0, 0.60)]
                    for risk in RISKS:
                        runs = pd.DataFrame([evaluate(v, days, starts, risk, stop) for v in variants])
                        grid.append({"months": months, "delta": dlt, "stop": f"-{stop:.0%}", "risk": risk,
                                     **runs.mean().to_dict(), "fail_ever_range": f"{runs['fail_ever_%'].min():.0f}-"
                                                                               f"{runs['fail_ever_%'].max():.0f}"})
                    print(f"done {months}m delta {dlt} stop {stop}", flush=True)

        pt = pd.DataFrame(per_trade)
        print("\n=== Part 1: per-trade result by contract price (all three setups pooled, option stop -50%) ===")
        print(pt[pt.stop == "-50%"].pivot_table(index=["months", "delta"], columns="contract", values="avg_%",
                                                observed=True).to_string())
        print("\ntrade counts:")
        print(pt[pt.stop == "-50%"].pivot_table(index=["months", "delta"], columns="contract", values="trades",
                                                observed=True).to_string())

        g = pd.DataFrame(grid).sort_values(["pass_6mo_%", "fail_ever_%"], ascending=[False, True])
        g.to_csv("strix_research.csv", index=False)
        print("\n=== Part 2: evaluations started monthly since Jan 2022 (average of 9 variants each) ===")
        print(g.to_string(index=False))
        for col in ("months", "delta", "stop", "risk"):
            print(f"\naverage effect of {col}:")
            print(g.groupby(col)[["pass_6mo_%", "fail_6mo_%", "pass_ever_%", "fail_ever_%"]].mean().to_string())

        print("\n=== Stress test of the top 8 by pass-within-6-months: options 25% richer, 3% half-spread ===")
        rows = []
        for _, r in g.head(8).iterrows():
            stop = float(r.stop.strip("-%")) / 100
            runs = pd.DataFrame([evaluate(trades_for(universe, days, pool, int(r.months), r.delta, stop, iv=1.25,
                                                         spread=0.03, jitter=j), days, starts, r.risk, stop)
                                 for j in JITTER])
            rows.append({"months": r.months, "delta": r.delta, "stop": r.stop, "risk": r.risk, **runs.mean().to_dict()})
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()

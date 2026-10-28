"""Daily Strix scan: which of the three tested call setups fired at the latest close?

The setups and filters are the ones strix_sim.py replays (chosen on pre-2025 trades by
calls.py). For each whitelisted ticker this lists:
  * new:     the setup's entry fired at the latest close, so it would be bought at the next open,
  * recent:  it fired within the last RECENT_DAYS sessions, the setup still holds the trade,
             and the price has not run more than 1 ATR past the signal close,
plus that setup's own record on that ticker (as 4-month ~0.70-delta calls with a -50% option
stop) and how stretched the entry is (distance above the 20-day SMA in ATRs).

Grade:
  A   the setup has at least 3 past trades on this ticker averaging >= +15%, and the entry is
      no more than 2 ATR above the 20-day SMA
  A-  at least 3 past trades averaging > 0%, entry no more than 2.5 ATR above the 20-day SMA
  B   anything else (fewer trades, a losing record on this ticker, or a stretched entry)

Option contracts, earnings dates and sizing are checked separately against live quotes.

Usage:
    python strix_scan.py
"""
from multiprocessing import Pool

import pandas as pd

import optsim
from backtest import STRATEGIES, atr, sma
from calls import FILTERS, load_universe
from strix_sim import SETUPS

RECENT_DAYS = 5
DTE, HOLD, DELTA, OPT_STOP = 84, 50, 0.70, 0.50


def scan_ticker(args):
    ticker, path = args
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    a, m20 = atr(df), sma(df.close, 20)
    rows = []
    for sname, fname in SETUPS:
        strat, kw = STRATEGIES[sname]
        entry, exit_ = strat(df)
        gated = (entry & FILTERS[fname](df, sname)).fillna(False)
        hold = min(HOLD, kw.get("max_hold") or HOLD)
        pos = optsim.run(df, gated, exit_, kind="C", dte=DTE, max_hold=hold, target_delta=DELTA, opt_stop=OPT_STOP)
        past = [t["opt_%"] for t in pos if t["reason"] != "open"]
        fired = [i for i in range(len(df) - RECENT_DAYS, len(df)) if gated.iloc[i]]
        if not fired:
            continue
        i = fired[-1]
        age = len(df) - 1 - i
        still_in = age == 0 or (pos and pos[-1]["reason"] == "open")
        ran = (df.close.iloc[-1] - df.close.iloc[i]) / a.iloc[-1]
        if not still_in or ran > 1:
            continue
        stretch = (df.close.iloc[-1] - m20.iloc[-1]) / a.iloc[-1]
        n, avg = len(past), (sum(past) / len(past) if past else float("nan"))
        grade = ("A" if n >= 3 and avg >= 15 and stretch <= 2 else
                 "A-" if n >= 3 and avg > 0 and stretch <= 2.5 else "B")
        rows.append({"grade": grade, "ticker": ticker, "setup": sname, "status": "new" if age == 0 else f"{age}d ago",
                     "close": round(df.close.iloc[-1], 2), "signal_close": round(df.close.iloc[i], 2),
                     "atr": round(a.iloc[-1], 2), "above_sma20_atr": round(stretch, 1),
                     "past_trades": n, "past_avg_%": round(avg, 1),
                     "past_win_%": round(100 * sum(p > 0 for p in past) / n) if n else None})
    return rows


def main():
    universe = load_universe()
    with Pool() as pool:
        rows = [r for chunk in pool.map(scan_ticker, universe.items()) for r in chunk]
    vix = pd.read_csv("data/VIX_daily.csv", parse_dates=["date"]).set_index("date").close
    spy = pd.read_csv("data/SPY_daily.csv", parse_dates=["date"]).set_index("date").close
    asof = spy.index[-1].date()
    print(f"As of {asof} close | SPY {spy.iloc[-1]:.2f} ({'above' if spy.iloc[-1] > sma(spy, 200).iloc[-1] else 'BELOW'} "
          f"its 200-day) | VIX {vix.iloc[-1]:.2f} ({'calm' if vix.iloc[-1] < 20 else 'elevated'})\n")
    if not rows:
        print("No setups fired. No trade is a valid outcome.")
        return
    out = pd.DataFrame(rows)
    out["g"] = out.grade.map({"A": 0, "A-": 1, "B": 2})
    out = out.sort_values(["g", "past_avg_%"], ascending=[True, False]).drop(columns="g")
    pd.set_option("display.width", 220)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()

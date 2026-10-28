"""Daily Strix scan: which of the three tested call setups fired at the latest close?

The setups and filters are the ones strix_sim.py replays (chosen on pre-2025 trades by
calls.py). For each whitelisted ticker this lists:
  * new:     the setup's entry fired at the latest close, so it would be bought at the next open,
  * recent:  it fired within the last RECENT_DAYS sessions, the setup still holds the trade,
             and the price has not run more than 1 ATR past the signal close,
plus that setup's own record on that ticker (traded with exactly the contract strix_sim.py
tested, from strix_rules.py) and how stretched the entry is (distance above the 20-day SMA in
ATRs).

Grade:
  A   the setup has at least 3 past trades on this ticker averaging >= +15%, and the entry is
      no more than 2 ATR above the 20-day SMA
  A-  at least 3 past trades averaging > 0%, entry no more than 2.5 ATR above the 20-day SMA
  B   anything else (fewer trades, a losing record on this ticker, or a stretched entry)

A row is only `eligible` (a rule check, not evidence of profitability) when it grades A or A-, its latest bar is official (not provisional,
see rh_import.py), and the earnings policy in strix_rules.py allows the entry. Every scan is
saved to scans/; when a scan for the same date is re-run on official bars, the rows that changed
are printed. The specific contract is checked with strix_ticket.py. Nothing here places orders.

Usage:
    python strix_scan.py
"""
import os
from multiprocessing import Pool

import pandas as pd

import optsim
from backtest import STRATEGIES, atr, sma
from calls import FILTERS, load_universe
from strix_rules import DELTA, DTE, MAX_HOLD, OPT_STOP, RULES_VERSION, SESSION, planned_exit, provisional_bars
from strix_sim import SETUPS

RECENT_DAYS = 5


def scan_ticker(args):
    ticker, path = args
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    a, m20 = atr(df), sma(df.close, 20)
    rows = []
    for sname, fname in SETUPS:
        strat, kw = STRATEGIES[sname]
        entry, exit_ = strat(df)
        gated = (entry & FILTERS[fname](df, sname)).fillna(False)
        hold = min(MAX_HOLD, kw.get("max_hold") or MAX_HOLD)
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
        exit_day, when, _, blocked = planned_exit(ticker, df.index[-1] + SESSION)
        rows.append({"grade": grade, "ticker": ticker, "setup": sname, "status": "new" if age == 0 else f"{age}d ago",
                     "asof": str(df.index[-1].date()), "earnings_block": blocked or "", "exit_by": f"{exit_day.date()} {when}",
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
    asof = str(spy.index[-1].date())
    prov = provisional_bars()
    market_prov = {t for t in ("SPY", "VIX") if (t, asof) in prov}
    print(f"Rules {RULES_VERSION} | as of {asof} close | SPY {spy.iloc[-1]:.2f} ({'above' if spy.iloc[-1] > sma(spy, 200).iloc[-1] else 'BELOW'} "
          f"its 200-day) | VIX {vix.iloc[-1]:.2f} ({'calm' if vix.iloc[-1] < 20 else 'elevated'})")
    n_prov = sum(1 for t in universe if (t, asof) in prov)
    if n_prov or market_prov:
        print(f"PROVISIONAL DATA: {n_prov} tickers{' and ' + '/'.join(sorted(market_prov)) if market_prov else ''} "
              f"have a provisional {asof} bar. Entries are blocked until official bars are imported and this is re-run.")
    print()
    if not rows:
        print("No setups fired. No trade is a valid outcome.")
        return
    out = pd.DataFrame(rows)
    out["provisional"] = [(t, d) in prov or bool(market_prov) for t, d in zip(out.ticker, out["asof"])]
    out["eligible"] = out.grade.isin(["A", "A-"]) & ~out.provisional & (out.earnings_block == "")
    out["rules"] = RULES_VERSION
    out["g"] = out.grade.map({"A": 0, "A-": 1, "B": 2})
    out = out.sort_values(["g", "past_avg_%"], ascending=[True, False]).drop(columns="g")
    pd.set_option("display.width", 260, "display.max_colwidth", 70)
    print(out.drop(columns=["asof", "rules"]).to_string(index=False))

    os.makedirs("scans", exist_ok=True)
    kind = "provisional" if out.provisional.any() else "official"
    out.to_csv(f"scans/scan_{asof}_{kind}.csv", index=False)
    other = f"scans/scan_{asof}_provisional.csv"
    if kind == "official" and os.path.exists(other):
        before = pd.read_csv(other).set_index(["ticker", "setup"])
        now = out.set_index(["ticker", "setup"])
        gone, new = before.index.difference(now.index), now.index.difference(before.index)
        both = before.index.intersection(now.index)
        regraded = [i for i in both if before.loc[i, "grade"] != now.loc[i, "grade"]]
        print(f"\nvs the provisional scan for {asof}: {len(new)} new, {len(gone)} gone, {len(regraded)} regraded")
        for i in new:
            print("  new:", i)
        for i in gone:
            print("  gone:", i)
        for i in regraded:
            print(f"  regraded: {i} {before.loc[i, 'grade']} -> {now.loc[i, 'grade']}")


if __name__ == "__main__":
    main()

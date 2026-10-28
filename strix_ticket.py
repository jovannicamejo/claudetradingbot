"""Check one specific option contract against the live rules before paper-trading it.

Give it the contract and its current quote; it prints PASS or REJECT with every reason, the
contracts allowed, the planned stop loss, the full premium exposed, mark-to-mid and immediate
liquidation (sell at the bid) P&L, and the planned exit date under the earnings policy.
It never places an order.

Usage:
    python strix_ticket.py BAC 2027-03-19 50 --bid 3.80 --ask 4.10 --delta 0.62 [--entry 2026-10-29]
"""
import argparse

import pandas as pd

from strix_rules import (DELTA_BAND, DTE, DTE_BAND, MAX_SPREAD, OPT_STOP, RISK_PER_TRADE, SESSION,
                         planned_exit, provisional_bars, sessions_between)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker")
    ap.add_argument("expiry")
    ap.add_argument("strike", type=float)
    ap.add_argument("--bid", type=float, required=True)
    ap.add_argument("--ask", type=float, required=True)
    ap.add_argument("--delta", type=float, required=True)
    ap.add_argument("--entry", help="session the order would fill (default: next session after the data)")
    a = ap.parse_args()

    df = pd.read_csv(f"data/{a.ticker}_daily.csv", parse_dates=["date"]).set_index("date")
    last = df.index[-1]
    entry = pd.Timestamp(a.entry) if a.entry else last + SESSION
    mid = (a.bid + a.ask) / 2
    spread = (a.ask - a.bid) / mid
    dte = sessions_between(entry - SESSION, a.expiry)
    stop_loss_one = OPT_STOP * mid * 100
    n = int(RISK_PER_TRADE // stop_loss_one)
    exit_day, exit_why, blocked = planned_exit(a.ticker, entry)

    reasons = []
    if (a.ticker, str(last.date())) in provisional_bars():
        reasons.append(f"latest bar ({last.date()}) is provisional; import the official daily bar and re-run the scan")
    if n == 0:
        reasons.append(f"one contract plans to lose ${stop_loss_one:,.0f} at the -{OPT_STOP:.0%} stop, "
                       f"over the ${RISK_PER_TRADE} cap (max premium ${RISK_PER_TRADE / OPT_STOP / 100:.2f})")
    if spread > MAX_SPREAD:
        reasons.append(f"spread {spread:.1%} of mid is wider than the {MAX_SPREAD:.0%} the tests cover")
    if not DELTA_BAND[0] <= a.delta <= DELTA_BAND[1]:
        reasons.append(f"delta {a.delta:.2f} is outside the tested band {DELTA_BAND}")
    if not DTE_BAND[0] <= dte <= DTE_BAND[1]:
        reasons.append(f"{dte} sessions to expiry is outside the tested band {DTE_BAND} (target {DTE})")
    if blocked:
        reasons.append(blocked)

    print("PAPER TRADE ONLY - this tool never places orders.\n")
    print(f"{a.ticker} {a.expiry} {a.strike:g} call | bid {a.bid:.2f} ask {a.ask:.2f} mid {mid:.2f} | "
          f"spread {spread:.1%} | delta {a.delta:.2f} | {dte} sessions to expiry at entry {entry.date()}")
    k = max(n, 1)
    print(f"contracts allowed at ${RISK_PER_TRADE} planned risk: {n}" + ("" if n else "  (figures below are for 1)"))
    print(f"  limit buy at mid ${mid:.2f}: premium exposed ${k * mid * 100:,.0f} (the real maximum loss)")
    print(f"  planned stop: sell if the option trades at ${mid * (1 - OPT_STOP):.2f}; planned loss "
          f"${k * stop_loss_one:,.0f} (a stop can fill worse, or not at all as a stop-limit)")
    print(f"  right after a fill at mid:  mark-to-mid P&L $0 | liquidation at bid ${k * (a.bid - mid) * 100:,.0f}")
    print(f"  right after a fill at ask:  mark-to-mid P&L ${k * (mid - a.ask) * 100:,.0f} | liquidation at bid "
          f"${k * (a.bid - a.ask) * 100:,.0f} ({(a.bid - a.ask) / a.ask:.1%})")
    print(f"  planned exit by the close of {exit_day.date()} ({exit_why}), unless the setup's exit or the stop comes first")
    print("\nPASS" if not reasons else "\nREJECT\n  - " + "\n  - ".join(reasons))


if __name__ == "__main__":
    main()

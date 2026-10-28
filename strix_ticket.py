"""Check one specific option contract against the live rules before paper-trading it.

Give it the contract and its current quote. It sizes the trade from the highest permitted fill
(--limit, which may not exceed the mid; default the mid), then prints ELIGIBLE or REJECT with
every reason, the contracts allowed, the full premium exposed, the planned stop loss,
mark-to-mid and immediate liquidation (sell at the bid) P&L, and the planned exit under the
earnings policy. With --fill it recomputes all of that from the actual fill, and with --record
it appends the paper trade, tagged with the rules version, to paper_trades.csv.

ELIGIBLE means the trade passes the rules in strix_rules.py. It is not evidence that the trade
will make money. This tool never places orders.

Usage:
    python strix_ticket.py BAC 2027-03-19 50 --bid 3.80 --ask 4.10 --delta 0.62 [--limit 3.95]
    python strix_ticket.py BAC 2027-03-19 50 --bid 3.80 --ask 4.10 --delta 0.62 --fill 3.93 --record
"""
import argparse
import os
from datetime import datetime, timezone

import pandas as pd

from strix_rules import (DELTA_BAND, DTE, DTE_BAND, MAX_SPREAD, OPT_STOP, RISK_PER_TRADE, RULES_VERSION, SESSION,
                         planned_exit, provisional_bars, sessions_between)

LOG = "paper_trades.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker")
    ap.add_argument("expiry")
    ap.add_argument("strike", type=float)
    ap.add_argument("--bid", type=float, required=True)
    ap.add_argument("--ask", type=float, required=True)
    ap.add_argument("--delta", type=float, required=True)
    ap.add_argument("--limit", type=float, help="highest price you will pay (default and maximum: the mid)")
    ap.add_argument("--fill", type=float, help="actual fill price, once filled")
    ap.add_argument("--entry", help="session the order fills (default: next session after the data)")
    ap.add_argument("--record", action="store_true", help="append an eligible filled trade to paper_trades.csv")
    a = ap.parse_args()

    df = pd.read_csv(f"data/{a.ticker}_daily.csv", parse_dates=["date"]).set_index("date")
    last = df.index[-1]
    entry = pd.Timestamp(a.entry) if a.entry else last + SESSION
    mid = (a.bid + a.ask) / 2
    limit = mid if a.limit is None else a.limit
    price = a.fill if a.fill is not None else limit
    spread = (a.ask - a.bid) / mid
    dte = sessions_between(entry - SESSION, a.expiry)
    stop_loss_one = OPT_STOP * price * 100
    n = int(RISK_PER_TRADE // stop_loss_one)
    exit_day, when, exit_why, blocked = planned_exit(a.ticker, entry)

    reasons = []
    if (a.ticker, str(last.date())) in provisional_bars():
        reasons.append(f"latest bar ({last.date()}) is provisional; import the official daily bar and re-run the scan")
    if limit > mid + 1e-9:
        reasons.append(f"limit {limit:.2f} is above the mid {mid:.2f}; the rules allow buying at or below the mid")
    if a.fill is not None and a.fill > limit + 1e-9:
        reasons.append(f"fill {a.fill:.2f} is above the limit {limit:.2f}")
    if n == 0:
        reasons.append(f"one contract at {price:.2f} plans to lose ${stop_loss_one:,.0f} at the -{OPT_STOP:.0%} stop, "
                       f"over the ${RISK_PER_TRADE} cap (max price ${RISK_PER_TRADE / OPT_STOP / 100:.2f})")
    if spread > MAX_SPREAD:
        reasons.append(f"spread {spread:.1%} of the current mid is wider than {MAX_SPREAD:.0%} "
                       f"(${MAX_SPREAD * mid:.2f} at this mid)")
    if not DELTA_BAND[0] <= a.delta <= DELTA_BAND[1]:
        reasons.append(f"delta {a.delta:.2f} is outside the tested band {DELTA_BAND}")
    if not DTE_BAND[0] <= dte <= DTE_BAND[1]:
        reasons.append(f"{dte} sessions to expiry is outside the tested band {DTE_BAND} (target {DTE})")
    if blocked:
        reasons.append(blocked)

    k = max(n, 1)
    basis = "actual fill" if a.fill is not None else "limit (highest permitted fill)"
    print(f"PAPER TRADE ONLY - rules {RULES_VERSION}. This tool never places orders.\n")
    print(f"{a.ticker} {a.expiry} {a.strike:g} call | bid {a.bid:.2f} ask {a.ask:.2f} mid {mid:.2f} | "
          f"spread {spread:.1%} | delta {a.delta:.2f} | {dte} sessions to expiry at entry {entry.date()}")
    print(f"sized from the {basis} ${price:.2f}: {n} contract(s)" + ("" if n else "  (figures below are for 1)"))
    print(f"  premium exposed ${k * price * 100:,.0f} (the real maximum loss)")
    print(f"  planned stop: sell if the option trades at ${price * (1 - OPT_STOP):.2f}; planned loss "
          f"${k * stop_loss_one:,.0f} (a stop can fill worse, or not at all as a stop-limit)")
    print(f"  right after buying at ${price:.2f}: mark-to-mid ${k * (mid - price) * 100:,.0f} | "
          f"liquidation at the bid ${k * (a.bid - price) * 100:,.0f} ({(a.bid - price) / price:.1%})")
    print(f"  planned exit: {when} of {exit_day.date()} ({exit_why}), unless the setup's exit or the stop comes first")
    if reasons:
        print("\nREJECT\n  - " + "\n  - ".join(reasons))
    else:
        print(f"\nELIGIBLE under rules {RULES_VERSION} (a rule check, not evidence the trade will make money)")

    if a.record:
        if reasons or a.fill is None:
            print("\nnot recorded: only an eligible trade with an actual --fill is logged")
            return
        row = {"recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "rules": RULES_VERSION,
               "ticker": a.ticker, "expiry": a.expiry, "strike": a.strike, "entry": str(entry.date()), "bid": a.bid,
               "ask": a.ask, "mid": round(mid, 4), "limit": limit, "fill": a.fill, "delta": a.delta, "contracts": n,
               "premium_$": round(n * a.fill * 100, 2), "stop_price": round(a.fill * (1 - OPT_STOP), 2),
               "planned_loss_$": round(n * stop_loss_one, 2), "planned_exit": f"{exit_day.date()} {when}"}
        pd.DataFrame([row]).to_csv(LOG, mode="a", index=False, header=not os.path.exists(LOG))
        print(f"\nrecorded in {LOG}")


if __name__ == "__main__":
    main()

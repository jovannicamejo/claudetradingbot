"""The live trading rules, in one place, so the scan, the trade check and the backtests agree.

RULES_VERSION is frozen: change any number below only together with a new version string, and
record the version with every scan and paper trade, so forward results can be tied to one
unchanged rule set. "Eligible" under these rules is a rule check, not evidence of profitability.

Contract (the one strix_sim.py tested):
  * single-leg long call, ~DELTA delta (accepted band DELTA_BAND), DTE trading days to expiry at
    entry (accepted band DTE_BAND), or when it loses OPT_STOP of what was paid.
  * Time exit, the same convention as optsim.py: the fill session is session 1, including for an
    entry made sessions after the signal; the position is sold at the OPEN of session
    MAX_HOLD + 1 (the order is decided at the close of session MAX_HOLD).

Sizing (Strix 50K evaluation):
  * RISK_PER_TRADE is a hard cap on a trade's PLANNED loss at the option stop:
    contracts = floor(RISK_PER_TRADE / (OPT_STOP x fill x 100)); zero contracts means no trade.
    At a 50% stop that caps one contract at $500.
  * MAX_OPEN_RISK caps the planned loss of all open positions together. The two caps are separate.
  * Planned loss is not maximum loss. A stop is a trigger, not a guaranteed price: it can fill
    worse, or (as a stop-limit) not at all. The full premium paid is what is actually exposed.

Execution:
  * Buy with a limit at or below the mid; size from that limit (the highest permitted fill), then
    recompute exposure and the stop level from the actual fill. Skip when the bid/ask spread is
    wider than MAX_SPREAD of the current mid: the stress test assumed a 3% half-spread, so a 6%
    full spread is the most the tested results cover.
  * Report two P&Ls: mark-to-mid, and liquidation (selling at the bid now).

Pricing caveat: the backtests price options with Black-Scholes on trailing 60-day realized vol
(point-in-time: the value used at any session comes from closes before it). On 2026-10-28 that
model priced 17 quoted calls at a median 0.92x their live mid (0.79-1.02), because implied vol
ran ~1.15x realized; strix_sim.py's IV x1.15 scenario reflects that.

Earnings policy (one rule, applied to every ticker):
  * Never hold through a scheduled earnings report. The planned exit is the earliest of the
    setup's exit, the option stop, MAX_HOLD sessions, and the last close before the report (the
    prior session's close for a before-the-open report, that day's close for an after-close one).
  * No entry when the report would force an exit within MIN_SESSIONS_BEFORE_EARNINGS sessions,
    or when a stock's next report date is not in data/earnings.csv.
  * ETFs have no report of their own; their holdings' reports are ignored (a known gap).
  * The backtests hold through earnings because there is no earnings history in data/, so this
    rule shortens some trades relative to what was tested. It is untested.
"""
import os

import pandas as pd

from calls import DELTA, DTE, MAX_HOLD
from strix_sim import MAX_OPEN_RISK, OPT_STOP, RISK_PER_TRADE

RULES_VERSION = "strix-v1 (frozen 2026-10-28)"
DELTA_BAND = (0.60, 0.80)
DTE_BAND = (75, 130)
MAX_SPREAD = 0.06
MIN_SESSIONS_BEFORE_EARNINGS = 15
ETFS = {"DIA", "EEM", "EWZ", "FXI", "GDX", "GLD", "IBIT", "IWM", "KWEB", "QQQ", "SLV", "SMH", "SPY", "TLT",
        "USO", "XLE"}
EARNINGS = "data/earnings.csv"

# NYSE full-day closures through 2027 (sessions are counted as weekdays minus these).
HOLIDAYS = pd.to_datetime(["2026-11-26", "2026-12-25", "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26",
                           "2027-05-31", "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24"])
SESSION = pd.offsets.CustomBusinessDay(holidays=HOLIDAYS)


def sessions_between(a, b):
    """Trading sessions after `a` up to and including `b`."""
    return len(pd.date_range(pd.Timestamp(a) + SESSION, pd.Timestamp(b), freq=SESSION))


def next_earnings(ticker, after):
    """(date, timing) of the next report after `after`, ("ETF", None) for an ETF, or (None, None)."""
    if ticker in ETFS:
        return "ETF", None
    if not os.path.exists(EARNINGS):
        return None, None
    e = pd.read_csv(EARNINGS, parse_dates=["date"])
    e = e[(e.ticker == ticker) & (e.date > pd.Timestamp(after))].sort_values("date")
    return (e.date.iloc[0], e.timing.iloc[0]) if len(e) else (None, None)


def planned_exit(ticker, entry_day):
    """Latest planned exit for a position filled at `entry_day`'s open, and why.

    Returns (exit_date, when, reason, blocked_reason or None); `when` is "open" or "close".
    """
    entry_day = pd.Timestamp(entry_day)
    time_exit = entry_day + MAX_HOLD * SESSION  # open of session MAX_HOLD + 1
    held = f"open after {MAX_HOLD} sessions held"
    report, timing = next_earnings(ticker, entry_day - SESSION)
    if report is None:
        return time_exit, "open", held, "next earnings date unknown; add it to data/earnings.csv"
    if isinstance(report, str):
        return time_exit, "open", held + " (ETF: holdings' reports ignored)", None
    last_close = report - SESSION if timing == "am" else report
    if last_close >= time_exit:
        return time_exit, "open", held + f"; earnings {report.date()} {timing} is later", None
    room = sessions_between(entry_day - SESSION, last_close)
    blocked = (f"earnings {report.date()} {timing} forces an exit after {room} sessions "
               f"(< {MIN_SESSIONS_BEFORE_EARNINGS})") if room < MIN_SESSIONS_BEFORE_EARNINGS else None
    return last_close, "close", f"last close before earnings {report.date()} {timing}", blocked


def provisional_bars():
    path = "data/provisional.csv"
    if not os.path.exists(path):
        return set()
    p = pd.read_csv(path, dtype={"date": str})
    return set(zip(p.ticker, p.date))

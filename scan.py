"""Daily signal scan: which backtested setups are active right now, and should the VIX let you take them?

For every ticker in data/ (index ETFs and VIX excluded unless listed in ETFS), it lists the strategies
that are holding or would buy at the latest close, with that strategy's 5-year result on that ticker,
then grades each one using the VIX regime findings from vix_filter.py:

  * Dip-buys had better average trades when VIX >= 20 or spiking (>= 25% above its 20-day average)
    in the last 5 sessions, and worse ones when VIX was calm (< 20, no recent spike).
  * Trend strategies had better average trades when VIX < 20, and much worse ones when VIX >= 20.

Earnings dates are not in the price data, so add them to EARNINGS by hand; a setup whose holding
window crosses a report is flagged.

Usage:
    python scan.py
"""
import glob
from datetime import date

import pandas as pd

from backtest import STRATEGIES, table
from options_backtest import SIGNALS, stock_positions
from vix_filter import TREND

EXCLUDE = {"VIX"}
ETFS = {"QQQ", "XLK", "SMH", "SPY"}
MIN_BACKTEST = 20  # % total return a strategy needs on that ticker to count as a real edge
# Upcoming earnings (update by hand). Source: Robinhood earnings calendar, 2026-10-20.
EARNINGS = {"SOFI": "2026-10-27", "HOOD": "2026-10-27", "SMH": "2026-10-28", "XLK": "2026-10-28", "QQQ": "2026-10-28"}
EARNINGS_WINDOW = 10  # trading days a swing trade is typically held


def vix_regime():
    v = pd.read_csv("data/VIX_daily.csv", parse_dates=["date"]).set_index("date").close
    spike = (v > 1.25 * v.rolling(20).mean()).tail(5).any()
    level = v.iloc[-1]
    return level, bool(spike), v.index[-1].date()


def grade(family, level, spike):
    if family == "Trend":
        return "favorable" if level < 20 else "unfavorable"
    if level >= 20 or spike:
        return "favorable"
    return "unfavorable"


def main():
    level, spike, asof = vix_regime()
    print(f"VIX {level:.2f} on {asof}; spike in last 5 sessions: {'yes' if spike else 'no'}")
    print("Dip-buys:", grade("Dip-buy", level, spike), "| Trend:", grade("Trend", level, spike), "\n")
    rows = []
    tickers = sorted(p.split("/")[-1].split("_")[0] for p in glob.glob("data/*_daily.csv"))
    for t in [t for t in tickers if t not in EXCLUDE]:
        df = pd.read_csv(f"data/{t}_daily.csv", parse_dates=["date"]).set_index("date")
        bt = table(df)["total_%"]
        last = df.index[-1].date()
        days_to_er = None
        if t in EARNINGS:
            days_to_er = int(pd.bdate_range(last, EARNINGS[t]).size) - 1
        for s in SIGNALS:
            strat, kw = STRATEGIES[s]
            pos = stock_positions(df, strat, kw.get("max_hold"))
            entry, _ = strat(df)
            new = bool(entry.fillna(False).iloc[-1]) and not pos[-1]
            if not (pos[-1] or new):
                continue
            fam = "Trend" if s in TREND else "Dip-buy"
            g = grade(fam, level, spike)
            edge = bt[s] >= MIN_BACKTEST
            er = days_to_er is not None and 0 <= days_to_er <= EARNINGS_WINDOW
            score = "A" if edge and g == "favorable" and not er else "B" if edge and not er else "C"
            rows.append({"ticker": t, "strategy": s, "status": "new" if new else "holding", "family": fam,
                         "5y_%": round(bt[s]), "vix": g, "earnings_in": days_to_er if er else "",
                         "grade": score, "etf": t in ETFS})
    out = pd.DataFrame(rows).sort_values(["grade", "5y_%"], ascending=[True, False])
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))
    print("\nA = strategy has worked on this ticker, VIX regime favors it, no earnings in the window."
          "\nB = strategy has worked, but the VIX regime is against it."
          "\nC = weak or negative history, or an earnings report falls inside the holding window.")


if __name__ == "__main__":
    main()

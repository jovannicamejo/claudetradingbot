"""Trade the stock swing signals from backtest.py with options instead of shares.

There is no free history of option prices, so contracts are priced with
Black-Scholes using the stock's trailing 60-day realized volatility as the
implied-vol proxy (IV_MULT scales it). On 2026-10-07 the quoted ~45-day ATM IVs
were 0.8-1.3x RV60 for these names, so 1.0 is a fair central case; the report
also reruns everything with options 25% more expensive.

Mechanics
  * Entry/exit come from the stock strategy: an option position is open exactly
    while the stock version would hold shares (fills at the next open).
  * Long premium structures spend RISK_PCT of equity per position; a loss is
    capped at that premium. Short puts are cash-secured (strike x 100 per
    contract = all equity), so they are directly comparable to owning stock.
  * Positions are rolled to a fresh contract when ROLL_DTE trading days remain.
  * Every leg pays a half-spread on each open/close, taken from the bid/ask
    widths quoted on 2026-10-07 (HALF_SPREAD below).

Usage:
    python options_backtest.py                   # all tickers in data/
    python options_backtest.py SOFI HOOD
"""
import glob
import sys
from math import erf, exp, log, sqrt

import numpy as np
import pandas as pd

from backtest import STRATEGIES

RATE = 0.04
RISK_PCT = 0.10  # premium spent per long-option position, as a share of equity
ROLL_DTE = 5  # trading days left at which a position is rolled
IV_WINDOW = 60
# Half the quoted bid/ask width as a fraction of the option's mid, per ticker
# (ATM, ~45 DTE, 2026-10-07). OTM and short-dated contracts are usually wider.
HALF_SPREAD = {"FCEL": 0.035, "HOOD": 0.01, "POET": 0.04, "SOFI": 0.015, "BULL": 0.03,
               "QQQ": 0.005, "SPY": 0.005, "XLK": 0.01, "SMH": 0.01}
MIN_TICK = 0.01

SIGNALS = [
    "SMA 10/30 cross",
    "SMA 20/50 cross",
    "EMA 9/21 cross",
    "MACD 12/26/9",
    "Donchian 20/10 breakout",
    "Turtle 55/20 breakout",
    "Volume breakout (20d hi, 2x vol)",
    "RSI14 <30 -> >50",
    "RSI2 <10 -> >70 (max 10d)",
    "Bollinger 20/2 -> mid",
    "Trend pullback (EMA50+RSI3)",
]

# name: (legs, days to expiry). Leg = (+1 long / -1 short, "C"/"P", strike as a multiple of spot)
STRUCTURES = {
    "Long call ATM 45d": ([(1, "C", 1.00)], 30),
    "Long call 10% OTM 30d": ([(1, "C", 1.10)], 21),
    "Long call 10% ITM 60d": ([(1, "C", 0.90)], 42),
    "Call spread ATM/+15% 45d": ([(1, "C", 1.00), (-1, "C", 1.15)], 30),
    "Short put 5% OTM 30d (cash-secured)": ([(-1, "P", 0.95)], 21),
}


def _ncdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def bs(kind, s, k, t, vol):
    if t <= 0:
        return max(s - k, 0) if kind == "C" else max(k - s, 0)
    d1 = (log(s / k) + (RATE + vol * vol / 2) * t) / (vol * sqrt(t))
    d2 = d1 - vol * sqrt(t)
    if kind == "C":
        return s * _ncdf(d1) - k * exp(-RATE * t) * _ncdf(d2)
    return k * exp(-RATE * t) * _ncdf(-d2) - s * _ncdf(-d1)


def stock_positions(df, strat, max_hold=None):
    """1 on days the stock strategy is holding through the open, mirroring backtest.run (no stops)."""
    entry, exit_ = strat(df)
    entry, exit_ = entry.fillna(False).values, exit_.fillna(False).values
    pos = np.zeros(len(df), dtype=int)
    holding, held = False, 0
    for i in range(1, len(df)):
        if holding and (exit_[i - 1] or (max_hold and held >= max_hold)):
            holding = False
        elif not holding and entry[i - 1]:
            holding, held = True, 0
        if holding:
            held += 1
        pos[i] = holding
    return pos


class Position:
    def __init__(self, legs, dte, spot, i, vol, half_spread, equity, cash_secured):
        self.legs = [(q, kind, round(m * spot, 2)) for q, kind, m in legs]
        self.expiry = i + dte
        self.half_spread = half_spread
        mids = self._mids(spot, i, vol)
        # pay the spread on every leg: buy at mid+h, sell at mid-h
        net = sum(q * (m + q * self._h(m)) for (q, _, _), m in zip(self.legs, mids))
        if cash_secured:
            self.n = equity / (self.legs[0][2] * 100)
        else:
            self.n = RISK_PCT * equity / (net * 100) if net > 0 else 0
        self.cash_flow = -net * 100 * self.n  # premium paid (<0) or received (>0)

    def _h(self, mid):
        return max(MIN_TICK, self.half_spread * mid)

    def _mids(self, spot, i, vol):
        t = max(self.expiry - i, 0) / 252
        return [bs(kind, spot, k, t, vol) for _, kind, k in self.legs]

    def value(self, spot, i, vol):
        """Mark-to-market (mid) value of the position."""
        return 100 * self.n * sum(q * m for (q, _, _), m in zip(self.legs, self._mids(spot, i, vol)))

    def close(self, spot, i, vol):
        """Cash received (or paid, if negative) to flatten, crossing the spread."""
        mids = self._mids(spot, i, vol)
        return 100 * self.n * sum(q * (m - q * self._h(m)) for (q, _, _), m in zip(self.legs, mids))


def run_options(df, pos, structure, half_spread, iv_mult=1.0):
    legs, dte = STRUCTURES[structure]
    cash_secured = legs[0][0] == -1 and legs[0][1] == "P"
    o, c = df.open.values, df.close.values
    lr = np.log(df.close).diff()
    vol = (lr.rolling(IV_WINDOW, min_periods=20).std() * sqrt(252) * iv_mult).clip(0.25, 3.0).bfill().values

    cash, p, trades, entry_cash = 1.0, None, [], 0.0
    equity = np.empty(len(df))

    def open_(spot, i, v):
        nonlocal cash, p, entry_cash
        p = Position(legs, dte, spot, i, v, half_spread, cash, cash_secured)
        entry_cash = cash
        cash += p.cash_flow

    def close_(spot, i, v):
        nonlocal cash, p
        cash += p.close(spot, i, v)
        trades.append(cash / entry_cash - 1)
        p = None

    for i in range(len(df)):
        v_prev = vol[i - 1] if i else vol[0]
        if p is not None and not pos[i]:
            close_(o[i], i, v_prev)
        elif p is None and pos[i]:
            open_(o[i], i, v_prev)
        if p is not None and p.expiry - i <= ROLL_DTE:  # roll at the close
            close_(c[i], i, vol[i])
            if i + 1 < len(df) and pos[i + 1]:
                open_(c[i], i, vol[i])
        equity[i] = cash + (p.value(c[i], i, vol[i]) if p is not None else 0)
    if p is not None:
        close_(c[-1], len(df) - 1, vol[-1])
        equity[-1] = cash
    return pd.Series(equity, index=df.index), trades


def summarize(equity, trades):
    t = np.array(trades)
    eq = np.maximum(equity, 1e-9)
    return {
        "total_%": 100 * (equity.iloc[-1] - 1),
        "max_dd_%": 100 * (eq / eq.cummax() - 1).min(),
        "trades": len(t),
        "win_%": 100 * (t > 0).mean() if len(t) else 0,
        "avg_trade_%": 100 * t.mean() if len(t) else 0,
    }


def load(ticker):
    return pd.read_csv(f"data/{ticker}_daily.csv", parse_dates=["date"]).set_index("date")


def stock_total(df, pos):
    """Same signals traded in shares with 0.2%/side, for reference."""
    r = df.open.pct_change().shift(-1).fillna(0).values  # open-to-open return while held
    eq = np.cumprod(1 + pos * r)
    trades = np.diff(np.concatenate([[0], pos])) != 0
    return 100 * (eq[-1] * (1 - 0.002) ** trades.sum() - 1)


def main(tickers):
    pd.set_option("display.width", 250, "display.float_format", "{:,.0f}".format)
    results = []
    for ticker in tickers:
        df = load(ticker)
        hs = HALF_SPREAD.get(ticker, 0.03)
        for sig in SIGNALS:
            strat, kw = STRATEGIES[sig]
            pos = stock_positions(df, strat, kw.get("max_hold"))
            row = {"ticker": ticker, "signal": sig, "structure": "Shares (100% of equity)",
                   "total_%": stock_total(df, pos)}
            results.append(row)
            for st in STRUCTURES:
                for mult in (1.0, 1.25):
                    eq, tr = run_options(df, pos, st, hs, mult)
                    r = summarize(eq, tr)
                    results.append({"ticker": ticker, "signal": sig, "structure": st, "iv_mult": mult, **r})
    res = pd.DataFrame(results)
    res["iv_mult"] = res["iv_mult"].fillna(1.0)
    base = res[res.iv_mult == 1.0]

    print(f"Option sizing: long premium = {RISK_PCT:.0%} of equity per position; short puts cash-secured.\n")
    print("=== Total return % by structure (median over the 11 signals) ===")
    med = base.pivot_table(index="structure", columns="ticker", values="total_%", aggfunc="median")[tickers]
    med["median"] = med.median(axis=1)
    print(med.sort_values("median", ascending=False).to_string(), "\n")

    print("=== Same, with options priced 25% richer (IV = 1.25 x RV60) ===")
    rich = res[(res.iv_mult == 1.25)].pivot_table(index="structure", columns="ticker", values="total_%",
                                                   aggfunc="median")[tickers]
    rich["median"] = rich.median(axis=1)
    print(rich.sort_values("median", ascending=False).to_string(), "\n")

    for ticker in tickers:
        t = base[base.ticker == ticker]
        print(f"=== {ticker}: total return % (signal x structure) ===")
        print(t.pivot_table(index="signal", columns="structure", values="total_%")
              .loc[SIGNALS].to_string())
        best = t[t.structure != "Shares (100% of equity)"].nlargest(5, "total_%")
        print(f"\nTop 5 option combos on {ticker}:")
        print(best[["signal", "structure", "total_%", "max_dd_%", "trades", "win_%", "avg_trade_%"]]
              .to_string(index=False), "\n")
    res.to_csv("options_results.csv", index=False)


if __name__ == "__main__":
    tickers = sys.argv[1:] or sorted(p.split("/")[-1].split("_")[0] for p in glob.glob("data/*_daily.csv"))
    main(tickers)

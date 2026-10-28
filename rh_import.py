"""Convert Robinhood get_equity_historicals JSON dumps into data/<TICKER>_daily.csv.

Each file holds the response of one get_equity_historicals call (interval=day). Bars marked
interpolated are gap-fill and are dropped. An existing CSV is merged with the new bars, the
new bars winning on any date both contain.

With --hourly, the dumps hold interval=hour bars instead. Regular-session hours are rolled up
into one bar per date, and only dates the CSV does not have yet are added. This fills in
today's bar before Robinhood publishes the daily one; it starts at the first full hour, so its
open, high, low and volume are approximate, but its close is exact. The next daily import
replaces it.

Usage:
    python rh_import.py dump1.json [dump2.json ...]
    python rh_import.py --hourly dump1.json [...]
"""
import json
import os
import sys

import pandas as pd

COLS = ["open", "high", "low", "close", "volume"]


def bars_to_frame(bars):
    rows = [{"date": b["begins_at"][:10], "open": float(b["open_price"]), "high": float(b["high_price"]),
             "low": float(b["low_price"]), "close": float(b["close_price"]), "volume": int(b["volume"])}
            for b in bars if not b.get("interpolated")]
    return pd.DataFrame(rows).set_index("date")


def hourly_to_daily(bars):
    hours = bars_to_frame([b for b in bars if b.get("session", "reg") == "reg"])
    hours.index = hours.index.str[:10]
    return hours.groupby(level=0).agg({"open": "first", "high": "max", "low": "min", "close": "last",
                                       "volume": "sum"})


def main(paths, hourly=False):
    for path in paths:
        with open(path) as f:
            results = json.load(f)["data"]["results"]
        for r in results:
            sym = r["symbol"]
            new = hourly_to_daily(r["bars"]) if hourly else bars_to_frame(r["bars"])
            if new.empty:
                print(f"{sym}: no bars")
                continue
            out = f"data/{sym}_daily.csv"
            if hourly:
                old = pd.read_csv(out, index_col="date")
                new = pd.concat([old, new[~new.index.isin(old.index)]]).sort_index()
            elif os.path.exists(out):
                old = pd.read_csv(out, index_col="date")
                new = pd.concat([old[~old.index.isin(new.index)], new]).sort_index()
            new[COLS].to_csv(out)
            print(f"{sym}: {len(new)} bars, {new.index[0]} to {new.index[-1]}")


if __name__ == "__main__":
    args = sys.argv[1:]
    main([a for a in args if a != "--hourly"], hourly="--hourly" in args)

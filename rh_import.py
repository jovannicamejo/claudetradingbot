"""Convert Robinhood get_equity_historicals JSON dumps into data/<TICKER>_daily.csv.

Each file holds the response of one get_equity_historicals call (interval=day). Bars marked
interpolated are gap-fill and are dropped. An existing CSV is merged with the new bars, the
new bars winning on any date both contain.

Usage:
    python rh_import.py dump1.json [dump2.json ...]
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


def main(paths):
    for path in paths:
        with open(path) as f:
            results = json.load(f)["data"]["results"]
        for r in results:
            sym, new = r["symbol"], bars_to_frame(r["bars"])
            if new.empty:
                print(f"{sym}: no bars")
                continue
            out = f"data/{sym}_daily.csv"
            if os.path.exists(out):
                old = pd.read_csv(out, index_col="date")
                new = pd.concat([old[~old.index.isin(new.index)], new]).sort_index()
            new[COLS].to_csv(out)
            print(f"{sym}: {len(new)} bars, {new.index[0]} to {new.index[-1]}")


if __name__ == "__main__":
    main(sys.argv[1:])

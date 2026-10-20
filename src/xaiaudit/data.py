"""Load daily price and volume from Coin Metrics Community Data CSVs.

Source: https://github.com/coinmetrics/data (csv/<asset>.csv). Columns used:
  PriceUSD                      daily USD reference price (close-like, 00:00 UTC)
  volume_reported_spot_usd_1d   reported spot trading volume in USD
Only close prices are available in this dataset (no open/high/low), so all
features are built from close and volume.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_coin(coin: str, raw_dir: str, start: str, end: str) -> pd.DataFrame:
    path = Path(raw_dir) / f"{coin}.csv"
    df = pd.read_csv(path, parse_dates=["time"])
    if "PriceUSD" not in df.columns:
        raise ValueError(f"{path} has no PriceUSD column")
    df = df[["time", "PriceUSD", "volume_reported_spot_usd_1d"]].rename(
        columns={"time": "date", "PriceUSD": "price", "volume_reported_spot_usd_1d": "volume"}
    )
    df = df[(df["date"] >= start) & (df["date"] <= end)].dropna()
    df = df[(df["price"] > 0) & (df["volume"] > 0)]
    df = df.set_index("date").sort_index()
    if df.index.has_duplicates:
        raise ValueError(f"duplicate dates in {coin}")
    gaps = (df.index.to_series().diff().dt.days > 1).sum()
    if gaps:
        raise ValueError(f"{coin}: {gaps} missing days in range; refusing to forward-fill silently")
    return df

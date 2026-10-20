"""Causal feature engineering.

Every feature on row t uses information available at the END of day t only.
The target on row t is the direction of the NEXT day's return (t -> t+1).
`tests/test_features.py` checks this by truncating the series and confirming
that past feature values never change when future data is removed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "ret_lag1",     # log return on day t
    "ret_lag2",     # log return on day t-1
    "ret_lag3",     # log return on day t-2
    "ret_5d",       # cumulative log return, last 5 days
    "ret_10d",      # cumulative log return, last 10 days
    "rsi_14",       # Wilder RSI, 14 days, scaled to [0, 1]
    "macd",         # (EMA12 - EMA26) / price
    "macd_signal",  # EMA9 of macd
    "macd_hist",    # macd - macd_signal  (exactly collinear with the two above)
    "vol_7",        # 7-day std of log returns
    "vol_30",       # 30-day std of log returns (also the regime variable)
    "volume_ratio", # log(volume / trailing 30-day mean volume)
]

# Groups of features derived from the same quantity; used in the report to
# discuss credit-splitting between correlated features.
FEATURE_GROUPS = {
    "momentum_returns": ["ret_lag1", "ret_lag2", "ret_lag3", "ret_5d", "ret_10d"],
    "oscillators": ["rsi_14", "macd", "macd_signal", "macd_hist"],
    "volatility": ["vol_7", "vol_30"],
    "volume": ["volume_ratio"],
}


def _rsi(price: pd.Series, n: int = 14) -> pd.Series:
    delta = price.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    avg_loss = loss.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 1 - 1 / (1 + rs)
    return rsi.fillna(1.0).where(avg_gain.notna())


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a frame with FEATURES, `target` (0/1) and `next_ret`."""
    p, v = df["price"], df["volume"]
    r = np.log(p).diff()
    out = pd.DataFrame(index=df.index)
    out["ret_lag1"] = r
    out["ret_lag2"] = r.shift(1)
    out["ret_lag3"] = r.shift(2)
    out["ret_5d"] = r.rolling(5).sum()
    out["ret_10d"] = r.rolling(10).sum()
    out["rsi_14"] = _rsi(p, 14)
    ema12 = p.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = p.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = (ema12 - ema26) / p
    out["macd"] = macd
    out["macd_signal"] = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    out["vol_7"] = r.rolling(7).std()
    out["vol_30"] = r.rolling(30).std()
    out["volume_ratio"] = np.log(v / v.rolling(30).mean())
    out["next_ret"] = r.shift(-1)                      # the ONLY forward-looking column
    out["target"] = (out["next_ret"] > 0).astype(float)
    out.loc[out["next_ret"].isna(), "target"] = np.nan
    # EWM warm-up: drop the first 60 rows so early EMA values are not biased
    out = out.iloc[60:]
    return out.dropna()

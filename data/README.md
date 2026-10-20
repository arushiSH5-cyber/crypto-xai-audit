# Data

`data/raw/*.csv` are unmodified daily files from **Coin Metrics Community Data**
(<https://github.com/coinmetrics/data>, `csv/<asset>.csv`) for BTC, ETH, ADA, LINK and LTC.
The pipeline uses two columns: `PriceUSD` and `volume_reported_spot_usd_1d`.

These files are © Coin Metrics and keep **Coin Metrics' own license terms**. They are *not*
covered by this repository's MIT license, which applies to the code only. Coin Metrics
publishes Community Data under a Creative Commons Attribution-NonCommercial license (see their
repository for the current terms). If you reuse the data, credit Coin Metrics and check those
terms first.

There is no processed-data folder in the repo: features are computed in memory by
`src/xaiaudit/features.py`.

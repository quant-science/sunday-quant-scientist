# The Quant Science Newsletter
# QS 028: Build a Strategy with a 471.9% return

# READY TO LEARN ALGORITHMIC TRADING WITH US?
# Register for our Course Waitlist: 
# https://learn.quantscience.io/python-algorithmic-trading-course-waitlist

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import vectorbt as vbt
import warnings
warnings.filterwarnings("ignore")

# Download historical price data for TLT ETF from Yahoo Finance and extract the closing prices

tlt = vbt.YFData.download(
    "TLT", 
    start="2004-01-01",
    end="2024-12-01"
).get("Close").to_frame()
close = tlt.Close

# Set up empty dataframes to hold trading signals for short and long positions

short_entries = pd.DataFrame.vbt.signals.empty_like(close)
short_exits = pd.DataFrame.vbt.signals.empty_like(close)
long_entries = pd.DataFrame.vbt.signals.empty_like(close)
long_exits = pd.DataFrame.vbt.signals.empty_like(close)

# Generate short entry signals on the first day of each new month

short_entry_mask = ~tlt.index.tz_convert(None).to_period("M").duplicated()
short_entries.iloc[short_entry_mask] = True

# Generate short exit signals five days after short entry

short_exit_mask = short_entries.shift(5).fillna(False)
short_exits.iloc[short_exit_mask] = True

# Generate long entry signals seven days before the end of each month

long_entry_mask = short_entries.shift(-7).fillna(False)
long_entries.iloc[long_entry_mask] = True

# Generate long exit signals one day before the end of each month

long_exit_mask = short_entries.shift(-1).fillna(False)
long_exits.iloc[long_exit_mask] = True

# Run the simulation and calculate the Sharpe ratio for the trading strategy

pf = vbt.Portfolio.from_signals(
    close=close,
    entries=long_entries,
    exits=long_exits,
    short_entries=short_entries,
    short_exits=short_exits,
    freq="1d"
)
pf.stats()

# Generate a plot with the strategy's performance.

pf.plot().show()


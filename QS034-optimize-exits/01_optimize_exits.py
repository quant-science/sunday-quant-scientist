# The Quant Science Newsletter
# QS 034: Optimize your strategy to find profitable exits

# READY TO LEARN ALGORITHMIC TRADING WITH US?
# Register for our Course Waitlist: 
# https://learn.quantscience.io/python-algorithmic-trading-course-waitlist

import pytz
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import vectorbt as vbt

# Define stock symbols to be analyzed

symbols = [
    "META",
    "AMZN",
    "AAPL",
    "NFLX",
    "GOOG",
]

# Define the start and end dates for historical data download

start_date = datetime(2018, 1, 1, tzinfo=pytz.utc)
end_date = datetime(2021, 1, 1, tzinfo=pytz.utc)

# Define various simulation parameters including number of stocks to trade, window length, seed, window count, exit types, and stop values

traded_count = 3
window_len = timedelta(days=12 * 21)

seed = 42
window_count = 400
exit_types = ["SL", "TS", "TP"]
stops = np.arange(0.01, 1 + 0.01, 0.01)

# Download historical stock data using vectorbt's YFData module and concatenate the data into a single DataFrame

yfdata = vbt.YFData.download(symbols, start=start_date, end=end_date)
ohlcv = yfdata.concat()

# Split the OHLCV data into windows defined by the window length and count

split_ohlcv = {}

for k, v in ohlcv.items():
    split_df, split_indexes = v.vbt.range_split(
        range_len=window_len.days, n=window_count
    )
    split_ohlcv[k] = split_df
ohlcv = split_ohlcv

# Calculate the momentum as the mean percentage change of the closing prices, then select the top stocks based on momentum

momentum = ohlcv["Close"].pct_change().mean()

sorted_momentum = (
    momentum
    .groupby(
        "split_idx", 
        group_keys=False, 
        sort=False
    )
    .apply(
        pd.Series.sort_values
    )
    .groupby("split_idx")
    .head(traded_count)
)

# Select the OHLCV data for the stocks with the highest momentum

selected_open = ohlcv["Open"][sorted_momentum.index]
selected_high = ohlcv["High"][sorted_momentum.index]
selected_low = ohlcv["Low"][sorted_momentum.index]
selected_close = ohlcv["Close"][sorted_momentum.index]

# Initialize entry signals to be true on the first day of each window

entries = pd.DataFrame.vbt.signals.empty_like(selected_open)
entries.iloc[0, :] = True

# Define stop loss exits using vectorbt's OHLCSTX module

sl_exits = vbt.OHLCSTX.run(
    entries,
    selected_open,
    selected_high,
    selected_low,
    selected_close,
    sl_stop=list(stops),
    stop_type=None,
    stop_price=None,
).exits

# Define trailing stop exits using vectorbt's OHLCSTX module

ts_exits = vbt.OHLCSTX.run(
    entries,
    selected_open,
    selected_high,
    selected_low,
    selected_close,
    sl_stop=list(stops),
    sl_trail=True,
    stop_type=None,
    stop_price=None,
).exits

# Define take profit exits using vectorbt's OHLCSTX module

tp_exits = vbt.OHLCSTX.run(
    entries,
    selected_open,
    selected_high,
    selected_low,
    selected_close,
    tp_stop=list(stops),
    stop_type=None,
    stop_price=None,
).exits

# Rename and drop levels for the different exit types to standardize the DataFrame structure

sl_exits.vbt.rename_levels({"ohlcstx_sl_stop": "stop_value"}, inplace=True)
ts_exits.vbt.rename_levels({"ohlcstx_sl_stop": "stop_value"}, inplace=True)
tp_exits.vbt.rename_levels({"ohlcstx_tp_stop": "stop_value"}, inplace=True)
ts_exits.vbt.drop_levels("ohlcstx_sl_trail", inplace=True)

# Ensure the last day in the window is always an exit signal for all exit types

sl_exits.iloc[-1, :] = True
ts_exits.iloc[-1, :] = True
tp_exits.iloc[-1, :] = True

# Convert exits into first exit signals based on entries, allowing gaps

sl_exits = sl_exits.vbt.signals.first(reset_by=entries, allow_gaps=True)
ts_exits = ts_exits.vbt.signals.first(reset_by=entries, allow_gaps=True)
tp_exits = tp_exits.vbt.signals.first(reset_by=entries, allow_gaps=True)

# Concatenate all exit signals into a single DataFrame for further analysis

exits = pd.DataFrame.vbt.concat(
    sl_exits,
    ts_exits,
    tp_exits,
    keys=pd.Index(exit_types, name="exit_type"),
)

# Create a portfolio using the selected close prices, entries, and exits

portfolio = vbt.Portfolio.from_signals(selected_close, entries, exits)

# Calculate the total return of the portfolio

total_return = portfolio.total_return()

# Unstack the total returns by exit type for visualization

total_return_by_type = total_return.unstack(level="exit_type")[exit_types]

# Plot histograms of the total returns for each exit type

total_return_by_type[exit_types].vbt.histplot(
    xaxis_title="Total return",
    xaxis_tickformat="%",
    yaxis_title="Count",
)

# Plot boxplots of the total returns for each exit type

total_return_by_type.vbt.boxplot(
    yaxis_title='Total return',
    yaxis_tickformat='%'
)

# Provide descriptive statistics for the total returns by exit type

total_return_by_type.describe(percentiles=[])

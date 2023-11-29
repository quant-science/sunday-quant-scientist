# The Quant Science Newsletter
# QS 007: Quantstats for Portfolio Analysis
# Copyright: Quant Science, LLC

# Libraries
from openbb_terminal.sdk import openbb
import quantstats as qs
import pandas as pd
import pytimetk as tk
import matplotlib.pyplot as plt

# Load the data
data = openbb.stocks.load("AAPL", start_date="2012-06-01", end_date="2022-06-30")

# Visaulize the data
data \
    .reset_index() \
    .plot_timeseries(
        date_column="date",
        value_column="Adj Close",
        title="AAPL Adj Close",
        x_lab="Date",
        y_lab="Adj Close",
    )

# Calculate the returns
aapl_returns = data['Adj Close'].pct_change()

# Calculate the metrics
qs.reports.metrics(
    aapl_returns,
    mode="full"
)

# Plot the snapshot
qs.plots.snapshot(aapl_returns)


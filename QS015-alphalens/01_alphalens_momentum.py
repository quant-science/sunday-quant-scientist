# The Quant Science Newsletter
# QS 015: Factor Analysis with Alphalens

# READY TO LEARN ALGORITHMIC TRADING WITH US?
# Register for our Course Waitlist: 
# https://learn.quantscience.io/python-algorithmic-trading-course-waitlist

import yfinance as yf
import pandas as pd
import numpy as np
import alphalens as al

# Fetch historical stock data
tickers = ["AAPL", "GOOGL", "MSFT"]
start_date = "2020-01-01"
end_date = "2023-01-01"

# Using yfinance to download data
data = yf.download(tickers, start=start_date, end=end_date)

# Selecting Adjusted Close prices and forward-filling any missing data
prices = data['Adj Close'].ffill()

# Calculate momentum
momentum_window = 90
momentum = prices.pct_change(periods=momentum_window).shift(-momentum_window)

# Prepare the factor data for Alphalens
factor_data = {
    (date, ticker): value
    for date in momentum.index
    for ticker, value in momentum.loc[date].items() if np.isfinite(value)
}
factor_index = pd.MultiIndex.from_tuples(factor_data.keys(), names=['date', 'asset'])
factor_values = pd.Series(factor_data.values(), index=factor_index, name='momentum')

# Prepare price data for Alphalens
aligned_prices = prices.stack().reindex(factor_index)

# Run Alphalens analysis
factor_data_al = al.utils.get_clean_factor_and_forward_returns(
    factor=factor_values,
    prices=aligned_prices.unstack(),
    periods=[1, 5, 10],
    max_loss=0.5
)

# Returns Tear Sheet
al.tears.create_returns_tear_sheet(factor_data_al)
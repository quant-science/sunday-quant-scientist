# The Quant Science Newsletter
# QS 032: How to measure your trading skill with the Information Ratio

# READY TO LEARN ALGORITHMIC TRADING WITH US?
# Register for our Course Waitlist: 
# https://learn.quantscience.io/python-algorithmic-trading-course-waitlist

import pandas as pd
import yfinance as yf

# Download historical price data for QQQ, AAPL, and AMZN from Yahoo Finance

data = yf.download(["QQQ", "AAPL", "AMZN"], start="2020-01-01", end="2022-07-31")

# Extract adjusted close prices for the downloaded data

closes = data['Adj Close']
benchmark_returns = closes.QQQ.pct_change()

# Construct a simple portfolio with equal shares of AAPL and AMZN

aapl_position = closes.AAPL * 50
amzn_position = closes.AMZN * 50

# Compute the portfolio value over time by summing the positions

portfolio_value = aapl_position + amzn_position

# Calculate the portfolio's daily profit and loss (PnL)

portfolio_pnl = (
    (aapl_position - aapl_position.shift()) 
    + (amzn_position - amzn_position.shift())
)

# Compute the portfolio's daily return by dividing PnL by the portfolio value

portfolio_returns = (portfolio_pnl / portfolio_value)
portfolio_returns.name = "Port"

# Create cumulative returns for both the portfolio and the benchmark

portfolio_cumulative_returns = (portfolio_returns.fillna(0.0) + 1).cumprod()
benchmark_cumulative_returns = (benchmark_returns.fillna(0.0) + 1).cumprod()

# Plot the cumulative returns of the portfolio against the benchmark

portfolio_cumulative_returns = (portfolio_returns.fillna(0.0) + 1).cumprod()
benchmark_cumulative_returns = (benchmark_returns.fillna(0.0) + 1).cumprod()

pd.concat([portfolio_cumulative_returns, benchmark_cumulative_returns], axis=1).plot()

def information_ratio(portfolio_returns, benchmark_returns):
    """
    Determines the information ratio of a strategy.
    
    Parameters
    ----------
    portfolio_returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
    benchmark_returns : int, float
        Daily returns of the benchmark or factor, noncumulative.

    Returns
    -------
    information_ratio : float

    Note
    -----
    See https://en.wikipedia.org/wiki/Information_ratio for more details.
    """
    
    # Calculate active return by subtracting benchmark returns from portfolio returns
    active_return = portfolio_returns - benchmark_returns

    # Calculate tracking error as the standard deviation of active returns
    tracking_error = active_return.std()

    # Return the information ratio, which is the mean active return divided by the tracking error
    return active_return.mean() / tracking_error

# Calculate the information ratio of the portfolio relative to the benchmark

information_ratio(portfolio_returns, benchmark_returns)



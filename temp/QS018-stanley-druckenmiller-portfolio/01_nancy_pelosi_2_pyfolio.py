# The Quant Science Newsletter
# QS 018: What can we learn about Nancy Pelosi's Portfolio - Part 2 (Pyfolio Analysis)?

# READY TO LEARN ALGORITHMIC TRADING WITH US?
# Register for our Course Waitlist: 
# https://learn.quantscience.io/python-algorithmic-trading-course-waitlist

# Libraries
import riskfolio as rp
import pandas as pd
import yfinance as yf
import pyfolio as pf

assets = [
    "PANW",
    "NVDA",
    "AAPL",
    "MSFT",
    "GOOG",
    "TSLA",
    "AB",
    "DIS",
    "AXP",
    "^GSPC", # SP500 benchmark
]

# Step 1: Collect and format data
data = yf.download(assets, start = "2018-01-01", end = "2024-07-10")
data = data.loc[:, "Adj Close"]
data

# Step 2: Get returns
returns = data.pct_change().dropna()
returns_bench = returns.pop("^GSPC").to_frame()

# Step 3: Create a Portfolio (Max Sharpe)
port = rp.Portfolio(returns)

w = port.optimization(
    model = "Classic",
    rm = "CVaR",
    obj = "Sharpe",
    hist = True,
    rf = 0,
    l = 0
)
w

# Step 4: Calculate the portfolio returns
portfolio_returns = (returns * w.weights).sum(axis=1)

portfolio_returns

# Step 5: Analyze Returns with Pyfolio

pf.show_perf_stats(portfolio_returns)

pf.plot_drawdown_periods(portfolio_returns)

pf.plot_drawdown_underwater(portfolio_returns)

pf.plot_rolling_sharpe(portfolio_returns)

pf.create_full_tear_sheet(
    portfolio_returns, 
    benchmark_rets=returns_bench['^GSPC']
)


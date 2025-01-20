# The Quant Science Newsletter
# QS 016: What can we learn about Nancy Pelosi's Portfolio (Riskfolio Analysis)?

# READY TO LEARN ALGORITHMIC TRADING WITH US?
# Register for our Course Waitlist: 
# https://learn.quantscience.io/python-algorithmic-trading-course-waitlist

# Source: 
#   8 Top Nancy Pelosi Stocks to Buy
#   https://money.usnews.com/investing/articles/top-nancy-pelosi-stocks-to-buy

# Libraries
import riskfolio as rp
import pandas as pd
import yfinance as yf

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

# Collect and format data
data = yf.download(assets, start = "2018-01-01", end = "2024-12-31")
data = data.loc[:, "Adj Close"]
data

# Get returns
returns = data.pct_change().dropna()
returns_bench = returns.pop("^GSPC").to_frame()

# Riskfolio

port = rp.Portfolio(returns)
port.assets_stats(
    method_mu="hist",
    method_cov="hist",
    d =0.94,
)
port.benchindex = returns_bench

# Max Sharpe Portfolio
w = port.optimization(
    model = "Classic",
    rm = "CVaR",
    obj = "Sharpe",
    hist = True,
    rf = 0,
    l = 0
)
w

rp.plot_pie(
    w = w, 
    others = 0.05, 
    nrow = 25, 
    cmap = "tab20",
    height = 6,
    width = 10,
    ax = None
)

rp.plot_series(returns=returns, w=w)

rp.plot_drawdown(returns, w=w)

rp.plot_table(returns, w)

# Can we do better? Efficient Frontier

wsim = port.efficient_frontier(model='Classic', rm = "MV", points=20, rf=0, hist=True)
wsim

ax = rp.plot_frontier(wsim, mu = port.mu, cov = port.cov, returns = returns)

ax = rp.plot_series(returns=returns,
                    w=wsim,
                    cmap='tab20',
                    height=6,
                    width=10,
                    ax=None)

# Max Return Portfolio

w_maxret = port.optimization(
    model = "Classic",
    rm = "CVaR",
    obj = "MaxRet",
    hist = True,
    rf = 0,
    l = 0
)


ax = rp.plot_pie(
    w = w_maxret, 
    others = 0.05, 
    nrow = 25, 
    cmap = "tab20",
    height = 6,
    width = 10,
    ax = None
)

rp.plot_series(returns=returns, w=w_maxret)

rp.plot_drawdown(returns, w_maxret)

rp.plot_table(returns, w_maxret)

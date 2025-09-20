# The Quant Science Newsletter
# QS 048: Build a dollar neutral trading strategy

# READY TO LEARN ALGORITHMIC TRADING WITH US?
# Register for our Course Waitlist: 
# https://learn.quantscience.io/python-algorithmic-trading-course-waitlist


import warnings
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import riskfolio as rp

warnings.filterwarnings("ignore")


# We set the date range and select the stocks for our portfolio.

start = "2016-01-01"
end = "2024-12-30"

tickers = [
    "JCI", "TGT", "CMCSA", "CPB", "MO", "APA", "MMC", "JPM", "ZION", "PSA",
    "BAX", "BMY", "LUV", "PCAR", "TXT", "TMO", "DE", "MSFT", "HPQ", "SEE",
    "VZ", "CNP", "NI", "T", "BA"
]
tickers.sort()


# We download historical stock data and calculate returns.

data = yf.download(tickers, start=start, end=end).Close
Y = data.pct_change().dropna()


# We build a portfolio object and set optimization parameters.

port = rp.Portfolio(returns=Y)

port.assets_stats(method_mu="hist", method_cov="ledoit")

port.sht = True
port.uppersht = 1
port.upperlng = 1
port.budget = 0
port.upperdev = 0.20 / 252**0.5

w = port.optimization(model="Classic", rm="CVaR", obj="Sharpe", hist=True)


# We create pie and bar charts to display the optimized portfolio weights.

title = "Max Return Dollar Neutral with Variance Constraint"
ax = rp.plot_pie(
    w=w, title=title, others=0.05, nrow=25, cmap="tab20", height=7, width=10, ax=None
)

ax = rp.plot_bar(
    w,
    title="Max Return Dollar Neutral with Variance Constraint",
    kind="v",
    others=0.05,
)

# We use the riskfolio library to create two visualizations of our optimized portfolio. The pie chart shows the allocation of weights across different assets, while the bar chart provides a vertical representation of the same information. These visualizations help us understand the composition of our optimized portfolio at a glance.

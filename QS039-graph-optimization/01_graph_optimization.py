# The Quant Science Newsletter
# QS 039: Using graphs to improve your Sharpe ratio

# READY TO LEARN ALGORITHMIC TRADING WITH US?
# Register for our Course Waitlist:
# https://learn.quantscience.io/python-algorithmic-trading-course-waitlist

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import riskfolio as rp
import yfinance as yf

warnings.filterwarnings("ignore")
pd.options.display.float_format = "{:.4%}".format

# Date range
start = "2016-01-01"
end = "2019-12-30"

# Tickers of assets. Use whatever you want.
assets = [
    "JCI",
    "TGT",
    "CMCSA",
    "CPB",
    "MO",
    "APA",
    "MMC",
    "JPM",
    "ZION",
    "PSA",
    "BAX",
    "BMY",
    "LUV",
    "PCAR",
    "TXT",
    "TMO",
    "DE",
    "MSFT",
    "HPQ",
    "SEE",
    "VZ",
    "CNP",
    "NI",
    "T",
    "BA",
]
assets.sort()

# Downloading data
data = yf.download(assets, start=start, end=end).Close

Y = data.pct_change().dropna()

# Building the portfolio object
port = rp.Portfolio(returns=Y)

# Calculating optimal portfolio

# Select method and estimate input parameters:

method_mu = "hist"  # Method to estimate expected returns based on historical data.
method_cov = "hist"  # Method to estimate covariance matrix based on historical data.

port.assets_stats(method_mu=method_mu, method_cov=method_cov)

# Estimate optimal portfolio:

model = "Classic"  # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
rm = "MV"  # Risk measure used, this time will be variance
obj = "MinRisk"  # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
hist = True  # Use historical scenarios for risk measures that depend on scenarios
rf = 0  # Risk free rate
l = 0  # Risk aversion factor, only useful when obj is 'Utility'

w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)

# Plotting the composition of the portfolio in MST
fig, ax = plt.subplots(2, 1, figsize=(10, 16))
ax = np.ravel(ax)

ax[0] = rp.plot_network_allocation(
    returns=Y,
    w=w,
    codependence="pearson",
    linkage="ward",
    alpha_tail=0.05,
    node_labels=True,
    leaf_order=True,
    kind="kamada",
    seed=123,
    ax=ax[0],
)

# Plotting the composition of the portfolio in the Dendrogram Cluster Network
ax[1] = rp.plot_clusters_network_allocation(
    returns=Y, w=w, codependence="pearson", linkage="ward", k=None, max_k=10, ax=ax[1]
)

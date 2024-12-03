# The Quant Science Newsletter
# QS 022: Hierarchical Risk Parity (HRP) Portfolio Optimization

# READY TO LEARN ALGORITHMIC TRADING WITH US?
# Register for our Course Waitlist: 
# https://learn.quantscience.io/python-algorithmic-trading-course-waitlist

import pandas as pd
import riskfolio as rp
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

# Define a list of asset symbols for which historical price data will be retrieved

assets = [
    "XLE", "XLF", "XLU", "XLI", "GDX", 
    "XLK", "XLV", "XLY", "XLP", "XLB", 
    "XOP", "IYR", "XHB", "ITB", "VNQ", 
    "GDXJ", "IYE", "OIH", "XME", "XRT", 
    "SMH", "IBB", "KBE", "KRE", "XTL", 
]

# Fetch historical price data for the specified assets and pivot the data into a DataFrame

data = (
    yf
    .download(assets)["Adj Close"]
)

# Calculate percentage returns from the historical price data and drop any missing values

returns = data.pct_change().dropna()

# Plot a dendrogram to visualize hierarchical clustering of asset returns using Pearson correlation

ax = rp.plot_dendrogram(
    returns=returns,
    codependence="pearson",
    linkage="single",
    k=None,
    max_k=10,
    leaf_order=True,
    ax=None,
)

# Create an instance of HCPortfolio with the calculated returns for portfolio optimization

port = rp.HCPortfolio(returns=returns)

# Optimize the portfolio using Hierarchical Risk Parity (HRP) with specified parameters

w = port.optimization(
    model="HRP",
    codependence="pearson",
    rm="MV",
    rf=0.05,
    linkage="single",
    max_k=10,
    leaf_order=True,
)

# Plot a pie chart to visualize the portfolio allocation resulting from the HRP optimization

ax = rp.plot_pie(
    w=w,
    title="HRP Naive Risk Parity",
    others=0.05,
    nrow=25,
    cmap="tab20",
    height=8,
    width=10,
    ax=None,
)

# Plot the risk contributions of each asset in the optimized portfolio

ax = rp.plot_risk_con(
    w=w,
    cov=returns.cov(),
    returns=returns,
    rm="MV",
    rf=0,
    alpha=0.05,
    color="tab:blue",
    height=6,
    width=10,
    t_factor=252,
    ax=None,
)

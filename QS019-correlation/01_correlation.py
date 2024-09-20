# The Quant Science Newsletter
# QS 019: Correlation Analysis

# READY TO LEARN ALGORITHMIC TRADING WITH US?
# Register for our Course Waitlist: 
# https://learn.quantscience.io/python-algorithmic-trading-course-waitlist


# * 1.0 Libraries & Data 

import riskfolio as rp
import pandas as pd
import yfinance as yf
import seaborn as sns
import riskfolio as rp
import pyfolio as pf

assets = [
    "PANW",
    "NVDA",
    "AAPL",
    "MSFT",
    "GOOG",
    "TSLA",
    "DIS",
    "AXP",
    "GLD", # S
    "^GSPC", # SP500 benchmark
]

# Collect and format data
data = yf.download(assets, start = "2018-01-01", end = "2024-08-08")
data = data.loc[:, "Adj Close"]
data

# Returns
returns = data.pct_change().dropna()
returns

returns.median().sort_values(ascending=False).to_frame(name="median_return")

# * 2.0 Correlations

# Get Correlations
corr_df = returns.corr()
corr_df

# Visualize the Correlations
sns.heatmap(
    corr_df, 
    annot=True, 
    cmap="coolwarm"
)

# Cluster the Correlations
sns.clustermap(
    corr_df, 
    cmap="coolwarm", 
    metric = "correlation", 
    annot = True
)

# Clustering Correlations
rp.plot_clusters(
    returns = returns,
    codependence='pearson',
    linkage='ward',
    k=None,
    max_k=10,
    leaf_order=True,
    dendrogram=True,
    ax=None
)

# * 3.0 Constructing the Portfolio using Nested Clustered Optimization (NCO)
port = rp.HCPortfolio(returns=returns)

w = port.optimization(
    model = 'NCO',
    codependence='pearson',
    method_cov = 'hist',
    obj='Sharpe',
    rm='MV',
    rf=0,
    l=2, 
    linkage='ward',
    max_k=10,
    leaf_order=True
)

w

# * 4.0 Portfolio Analysis

rp.plot_pie(w)

rp.plot_risk_con(returns=returns, w=w, cov=port.cov)

rp.plot_drawdown(returns=returns, w=w)

rp.plot_table(returns=returns, w=w)

# * BONUS: Portfolio Performance Comparison vs Benchmark with Pyfolio

portfolio_returns = (returns * w.weights).sum(axis=1)
portfolio_returns.name = "portfolio_returns"
portfolio_returns

pf.create_simple_tear_sheet(returns=portfolio_returns, benchmark_rets=returns['^GSPC'])

# Benchmark:
pf.create_simple_tear_sheet(returns=returns['^GSPC'])
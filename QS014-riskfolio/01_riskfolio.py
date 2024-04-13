# The Quant Science Newsletter
# QS 013: Portfolio Optimization with Riskfolio

# READY TO LEARN ALGORITHMIC TRADING WITH US?
# Register for our Course Waitlist: 
# https://learn.quantscience.io/python-algorithmic-trading-course-waitlist

# Libraries
import numpy as np
import pandas as pd
import yfinance as yf
import riskfolio as rp

# Date range
start = '2018-01-01'
end = '2023-12-31'

# Tickers of assets
assets = ['JCI', 'TGT', 'CMCSA', 'CPB', 'MO', 'APA', 'MMC', 'JPM',
          'ZION', 'PSA', 'BAX', 'BMY', 'LUV', 'PCAR', 'TXT', 'TMO',
          'DE', 'MSFT', 'HPQ', 'SEE', 'VZ', 'CNP', 'NI', 'T', 'NVDA']
assets.sort()

# Tickers of factors
factors = ['MTUM', 'QUAL', 'VLUE', 'SIZE', 'USMV']
factors.sort()

tickers = assets + factors
tickers.sort()

# Downloading the data
data = yf.download(tickers, start = start, end = end)
data = data.loc[:,('Adj Close', slice(None))]
data.columns = tickers
returns = data.pct_change().dropna()

Y = returns[assets]
X = returns[factors]

# 1.0 Creating the Portfolio Object
port = rp.Portfolio(returns=Y)

# Choose the risk measure
rm = 'MSV'  # Semi Standard Deviation

# Estimate inputs of the model (historical estimates)
method_mu='hist' # Method to estimate expected returns based on historical data.
method_cov='hist' # Method to estimate covariance matrix based on historical data.

port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)

mu = port.mu
cov = port.cov

# 2.0 Estimate the portfolio that maximizes the risk adjusted return ratio
w1 = port.optimization(model='Classic', rm=rm, obj='Sharpe', rf=0.0, l=0, hist=True)

# 3.0 Estimate points in the efficient frontier mean - semi standard deviation
ws = port.efficient_frontier(model='Classic', rm=rm, points=20, rf=0, hist=True)

# 4.0 Portfolio Cumulative Returns
ax = rp.plot_series(returns=Y,
                    w=ws,
                    cmap='tab20',
                    height=6,
                    width=10,
                    ax=None)

ax = rp.plot_series(returns=Y,
                    w=w1,
                    cmap='tab20',
                    height=6,
                    width=10,
                    ax=None)

# 5.0 Efficient Frontier
label = 'Max Risk Adjusted Return Portfolio'
mu = port.mu
cov = port.cov
returns = port.returns

ax = rp.plot_frontier(w_frontier=ws,
                      mu=mu,
                      cov=cov,
                      returns=Y,
                      rm=rm,
                      rf=0,
                      alpha=0.05,
                      cmap='viridis',
                      w=w1,
                      label=label,
                      marker='*',
                      s=16,
                      c='r',
                      height=6,
                      width=10,
                      t_factor=252,
                      ax=None)

# 6.0 Portfolio Donut Chart
ax = rp.plot_pie(w=w1,
                 title='Portfolio',
                 height=6,
                 width=10,
                 cmap="tab20",
                 ax=None)

# 7.0 Plot Table
ax = rp.plot_table(returns=Y,
                   w=w1,
                   MAR=0,
                   alpha=0.05,
                   ax=None)

# 8.0 Plot Risk Contribution
ax = rp.plot_risk_con(w=w1,
                      cov=cov,
                      returns=Y,
                      rm=rm,
                      rf=0,
                      alpha=0.05,
                      color="tab:blue",
                      height=6,
                      width=10,
                      t_factor=252,
                      ax=None)

# 9.0 Excel Reports
rp.excel_report(returns,
                w1,
                rf=0,
                alpha=0.05,
                t_factor=252,
                ini_days=1,
                days_per_year=252,
                name="QS014-riskfolio/excel-report")
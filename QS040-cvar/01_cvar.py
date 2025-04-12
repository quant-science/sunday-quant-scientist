# The Quant Science Newsletter
# QS 040: Use CVaR to keep the money you make trading

# READY TO LEARN ALGORITHMIC TRADING WITH US?
# Register for our Course Waitlist: 
# https://learn.quantscience.io/python-algorithmic-trading-course-waitlist

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Define a list of stock tickers representing the portfolio components

oex = ['MMM','T','ABBV','ABT','ACN','ALL','GOOGL','GOOG','MO','AMZN','AXP','AIG','AMGN','AAPL','BAC',
       'BRK-B','BIIB','BLK','BA','BMY','CVS','COF','CAT','CVX','CSCO','C','KO','CL','CMCSA',
       'COP','DHR','DUK','DD','EMC','EMR','EXC','XOM','META','FDX','F','GD','GE','GM','GILD',
       'GS','HAL','HD','HON','INTC','IBM','JPM','JNJ','KMI','LLY','LMT','LOW','MA','MCD','MDT','MRK',
       'MET','MSFT','MS','NKE','NEE','OXY','ORCL','PYPL','PEP','PFE','PM','PG','QCOM',
       'SLB','SPG','SO','SBUX','TGT','TXN','BK','USB','UNP','UPS','UNH','VZ','V','WMT',
       'WBA','DIS','WFC']

# Count the number of stocks in the portfolio

num_stocks = len(oex)

# Download historical stock data for the defined period

data = yf.download(oex, start='2014-01-01', end='2016-04-04')

# Calculate daily returns and de-mean the returns by subtracting the mean

returns = data.Close.pct_change(fill_method=None)
returns = returns - returns.mean(skipna=True)


def scale(x):
    return x / np.sum(np.abs(x))

# Generate random weights for the portfolio and scale them

weights = scale(np.random.random(num_stocks))
plt.bar(np.arange(num_stocks), weights)

# Define a function to calculate Value at Risk (VaR)

def value_at_risk(value_invested, returns, weights, alpha=0.95, lookback_days=500):
    # Fill missing values in returns with zero and calculate portfolio returns
    returns = returns.fillna(0.0)
    portfolio_returns = returns.iloc[-lookback_days:].dot(weights)

    # Calculate the VaR as the percentile of portfolio returns
    return np.percentile(portfolio_returns, 100 * (1 - alpha)) * value_invested

# Define parameters for lookback days and confidence level

# Define a function to calculate Conditional Value at Risk (CVaR)

def cvar(value_invested, returns, weights, alpha=0.95, lookback_days=500):
    # Calculate VaR and portfolio returns for the specified lookback period
    var = value_at_risk(value_invested, returns, weights, alpha, lookback_days=lookback_days)
    
    returns = returns.fillna(0.0)
    portfolio_returns = returns.iloc[-lookback_days:].dot(weights)
    var_pct_loss = var / value_invested
    
    # Calculate the mean of returns below the VaR threshold
    return np.nanmean(portfolio_returns[portfolio_returns < var_pct_loss]) * value_invested

# Calculate CVaR using the defined function

value_invested = 100_000
cvar(value_invested, returns, weights, lookback_days=500)

# Calculate VaR again for consistency

value_at_risk(value_invested, returns, weights, lookback_days=500)

# Calculate portfolio returns using historical data and weights

lookback_days = 500

portfolio_returns = returns.fillna(0.0).iloc[-lookback_days:].dot(weights)

# Calculate VaR and CVaR and express them as returns

portfolio_VaR = value_at_risk(value_invested, returns, weights)
portfolio_VaR_return = portfolio_VaR / value_invested

portfolio_CVaR = cvar(value_invested, returns, weights)
portfolio_CVaR_return = portfolio_CVaR / value_invested

# Plot histogram of portfolio returns, marking VaR and CVaR on the plot

plt.hist(portfolio_returns[portfolio_returns > portfolio_VaR_return], bins=20)
plt.hist(portfolio_returns[portfolio_returns < portfolio_VaR_return], bins=10)
plt.axvline(portfolio_VaR_return, color='red', linestyle='solid')
plt.axvline(portfolio_CVaR_return, color='red', linestyle='dashed')
plt.legend(['VaR', 'CVaR', 'Returns', 'Returns < VaR'])
plt.title('Historical VaR and CVaR')
plt.xlabel('Return')
plt.ylabel('Observation Frequency')



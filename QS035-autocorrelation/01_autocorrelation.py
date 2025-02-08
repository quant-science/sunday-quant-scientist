# The Quant Science Newsletter
# QS 035: Use autocorrelation to find a trend

# READY TO LEARN ALGORITHMIC TRADING WITH US?
# Register for our Course Waitlist: 
# https://learn.quantscience.io/python-algorithmic-trading-course-waitlist

import yfinance as yf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Download historical monthly price data for the E-mini S&P 500 Futures

prices = yf.download("ES=F", start="2022-01-01", interval="1mo")

# Compute the percentage change in closing prices to determine the returns

returns = prices.Close.pct_change().dropna()

# Create plots to visualize the autocorrelation and partial autocorrelation of the returns

plot_acf(returns, lags=12);

# The `plot_acf` function generates a plot of the autocorrelation function for the returns. The plot shows how the returns are correlated with their own past values at different lags, up to 12 months. Each vertical bar represents the correlation at a specific lag, and the shaded area indicates the confidence interval. If a bar extends beyond the shaded area, it suggests a statistically significant correlation at that lag. This helps in identifying any patterns or dependencies in the return series.

plot_pacf(returns, lags=12);

# The `plot_pacf` function generates a plot of the partial autocorrelation function for the returns. This plot shows the direct correlation between the returns and their lagged values, excluding the influence of intermediate lags. Like the autocorrelation plot, the partial autocorrelation plot includes bars for each lag and shaded confidence intervals. By examining these plots, we can identify significant autoregressive patterns that might be useful for modeling and forecasting future returns.

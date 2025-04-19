# The Quant Science Newsletter
# QS 041: Trend or momentum? Use the Hurst Exponent to find out

# READY TO LEARN ALGORITHMIC TRADING WITH US?
# Register for our Course Waitlist:
# https://learn.quantscience.io/python-algorithmic-trading-course-waitlist

import pandas as pd
import numpy as np
import yfinance as yf

# Load historical S&P 500 data from 2000 to 2019 using the OpenBB SDK and select the adjusted close prices

df = yf.download("^GSPC", start="2000-01-01", end="2019-12-31").Close

# Plot the S&P 500 adjusted close prices to visualize the historical data

df.plot(title="S&P 500")


def get_hurst_exponent(ts, max_lag=20):
    # Define the range of lags to be used in the calculation
    lags = range(2, max_lag)

    # Calculate the standard deviation of differences for each lag
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]

    # Perform a linear fit to estimate the Hurst exponent
    return np.polyfit(np.log(lags), np.log(tau), 1)[0]


# Calculate and print the Hurst exponent for various lags using the full dataset

for lag in [20, 100, 250, 500, 1000]:
    hurst_exp = get_hurst_exponent(df.values, lag)
    print(f"{lag} lags: {hurst_exp:.4f}")


# Select a shorter series from 2005 to 2007 and calculate the Hurst exponent for various lags

shorter_series = df.loc["2005":"2007"].values
for lag in [20, 100, 250, 500]:
    hurst_exp = get_hurst_exponent(shorter_series, lag)
    print(f"{lag} lags: {hurst_exp:.4f}")


# Calculate rolling volatility using a 30-day window and plot the results to observe changes over time

rv = df.rolling(30).apply(np.std)
rv.plot()


# Calculate and print the Hurst exponent for various lags using the rolling volatility data

for lag in [20, 100, 250, 500, 1000]:
    hurst_exp = get_hurst_exponent(rv.dropna().values, lag)
    print(f"{lag} lags: {hurst_exp:.4f}")

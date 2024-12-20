# The Quant Science Newsletter
# QS 030: Using the Omega ratio for responsible algo trading

# READY TO LEARN ALGORITHMIC TRADING WITH US?
# Register for our Course Waitlist: 
# https://learn.quantscience.io/python-algorithmic-trading-course-waitlist

import yfinance as yf
import numpy as np

# Download the stock data for AAPL from Yahoo Finance for the specified date range

data = yf.download("AAPL", start="2020-01-01", end="2021-12-31")

returns = data["Adj Close"].pct_change()

# Calculate the Omega ratio of a strategy's returns

def omega_ratio(returns, required_return=0.0):
    """Determines the Omega ratio of a strategy.
    
    Parameters
    ----------
    returns : pd.Series or np.ndarray
        Daily returns of the strategy, noncumulative.
    required_return : float, optional
        Minimum acceptance return of the investor. Threshold over which to
        consider positive vs negative returns. It will be converted to a
        value appropriate for the period of the returns. E.g. An annual minimum
        acceptable return of 100 will translate to a minimum acceptable
        return of 0.018.

    Returns
    -------
    omega_ratio : float

    Note
    -----
    See https://en.wikipedia.org/wiki/Omega_ratio for more details.
    """

    # Convert the required return to a daily return threshold
    return_threshold = (1 + required_return) ** (1 / 252) - 1

    # Calculate the difference between returns and the return threshold
    returns_less_thresh = returns - return_threshold

    # Calculate the numerator as the sum of positive returns above the threshold
    numer = sum(returns_less_thresh[returns_less_thresh > 0.0])

    # Calculate the denominator as the absolute sum of negative returns below the threshold
    denom = -1.0 * sum(returns_less_thresh[returns_less_thresh < 0.0])

    # Return the Omega ratio if the denominator is positive; otherwise, return NaN
    if denom > 0.0:
        return numer / denom
    else:
        return np.nan

# Calculate the Omega ratio for the given returns and required return

omega_ratio(returns, 0.07)

# Compute and plot the rolling 30-day Omega ratio of the returns

returns.rolling(30).apply(omega_ratio).plot()

# Plot a histogram of the daily returns to visualize their distribution

returns.hist(bins=50)

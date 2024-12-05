# The Quant Science Newsletter
# QS 024: Downside Deviation vs Volatility

# READY TO LEARN ALGORITHMIC TRADING WITH US?
# Register for our Course Waitlist: 
# https://learn.quantscience.io/python-algorithmic-trading-course-waitlist

import numpy as np
import yfinance as yf

data = yf.download("AAPL")
returns = data["Adj Close"].pct_change()


def downside_deviation(returns):
    # Initialize an empty array to store downside deviation values
    out = np.empty(returns.shape[1:])

    # Clip returns at zero to focus on negative returns
    downside_diff = np.clip(returns, np.NINF, 0)

    # Square the clipped values to calculate the squared deviations
    np.square(downside_diff, out=downside_diff)

    # Calculate the mean of squared deviations ignoring NaNs
    np.nanmean(downside_diff, axis=0, out=out)

    # Take the square root of the mean squared deviations
    np.sqrt(out, out=out)

    # Annualize the downside deviation by multiplying by the square root of 252
    np.multiply(out, np.sqrt(252), out=out)

    # Return the annualized downside deviation as a single value
    return out.item()

dd = downside_deviation(returns)

vol = np.sqrt(np.square(returns).mean()) * np.sqrt(252)

dd / vol 

# If dd/vol is greater than 1, then the downside deviation is greater than the volatility

# If dd/vol is less than 1, then the downside deviation is less than the volatility

# If dd/vol is equal to 1, then the downside deviation is equal to the volatility
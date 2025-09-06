# The Quant Science Newsletter
# QS 047: Measure fat tail returns with skew and kurtosis

# READY TO LEARN ALGORITHMIC TRADING WITH US?
# Join our free webinar where we will give you a sneak peek into the strategies and Python code that power our hedge fund:
# https://learn.quantscience.io/qs-register

# These libraries help us fetch financial data, manipulate it, and visualize the results

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# We'll get SPY data for the last 5 years and calculate its daily returns.

spy = yf.Ticker("SPY")
data = spy.history(period="5y")
returns = data['Close'].pct_change().dropna()

# We use yfinance to download SPY stock data for the past 5 years. Then, we calculate the daily percentage change in closing prices. This gives us the daily returns. We remove any missing values to ensure our data is clean and ready for analysis.

# Now we'll compute the skewness and kurtosis of the returns.

skewness = returns.skew()
kurtosis = returns.kurtosis()

print(f"Skewness: {skewness:.4f}")
print(f"Kurtosis: {kurtosis:.4f}")

# We calculate two important measures of the returns distribution. Skewness tells us about the symmetry of the distribution. Kurtosis gives us information about the tails of the distribution. These values help us understand how the returns are distributed compared to a normal distribution.

# Let's create a histogram of the returns and overlay a normal distribution for comparison.

plt.figure(figsize=(10, 6))
returns.hist(bins=50, density=True, alpha=0.7)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, returns.mean(), returns.std())
plt.plot(x, p, 'k', linewidth=2)
plt.title("SPY Daily Returns Distribution")
plt.xlabel("Returns")
plt.ylabel("Frequency")
plt.show()

# We create a histogram of the returns to visually represent their distribution. We set the number of bins to 50 for a detailed view. We also calculate and plot a normal distribution curve using the mean and standard deviation of our returns. This allows us to compare our actual returns distribution to what a normal distribution would look like.

# <a href="https://pyquantnews.com/">PyQuant News</a> is where finance practitioners level up with Python for quant finance, algorithmic trading, and market data analysis. Looking to get started? Check out the fastest growing, top-selling course to <a href="https://gettingstartedwithpythonforquantfinance.com/">get started with Python for quant finance</a>. For educational purposes. Not investment advise. Use at your own risk.

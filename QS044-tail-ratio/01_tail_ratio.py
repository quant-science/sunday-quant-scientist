# The Quant Science Newsletter
# QS 044: The metrics that power our algorithmic trading strategies

# READY TO LEARN ALGORITHMIC TRADING WITH US?
# Join our free webinar where we will give you a sneak peek into the strategies and Python code that power our hedge fund:
# https://learn.quantscience.io/qs-register


# These libraries help us download financial data, analyze it, and create visualizations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# We download stock data for NVDA and AMD, calculate daily returns, and prepare it for analysis

df = (
    yf.download(["NVDA", "AMD"], start="2020-01-01")
    .Close.pct_change(fill_method=None)
    .dropna()
)

# This code fetches the closing prices for NVIDIA and AMD stocks from January 1, 2020, to the present. It then calculates the daily percentage returns and removes any rows with missing data. The resulting dataframe contains the daily returns for both stocks, which we'll use for our analysis.

# ### Calculate tail ratios

# We define a function to compute the tail ratio and apply it to our stock data

def tail_ratio(returns):
    return abs(np.percentile(returns, 95)) / abs(np.percentile(returns, 5))

tail_ratio_a = tail_ratio(df.AMD)
tail_ratio_b = tail_ratio(df.NVDA)

print(f"Tail Ratio for AMD: {tail_ratio_a:.4f}")
print(f"Tail Ratio for NVDA: {tail_ratio_b:.4f}")

# The tail ratio is a measure of the relative size of the positive and negative tails of a distribution. Our function calculates this by dividing the absolute value of the 95th percentile by the absolute value of the 5th percentile. We then apply this function to both AMD and NVIDIA returns and print the results. A higher tail ratio suggests more extreme positive returns relative to negative ones.

# ### Visualize return distributions

# We create a histogram to compare the return distributions of AMD and NVIDIA

plt.figure(figsize=(10, 6))
plt.hist(df.AMD, bins=50, alpha=0.5, label="AMD")
plt.hist(df.NVDA, bins=50, alpha=0.5, label="NVDA")
plt.axvline(np.percentile(df.AMD, 5), color="r", linestyle="dashed", linewidth=2)
plt.axvline(np.percentile(df.AMD, 95), color="r", linestyle="dashed", linewidth=2)
plt.axvline(np.percentile(df.NVDA, 5), color="g", linestyle="dashed", linewidth=2)
plt.axvline(np.percentile(df.NVDA, 95), color="g", linestyle="dashed", linewidth=2)
plt.legend()
plt.title("Return Distributions with 5th and 95th Percentiles")
plt.xlabel("Returns")
plt.ylabel("Frequency")
plt.show()

# This visualization creates overlapping histograms of the daily returns for AMD and NVIDIA. We use 50 bins to show the distribution of returns for each stock. The 5th and 95th percentiles are marked with dashed lines for each stock. This helps us visually compare the return distributions and see how the tail ratios we calculated earlier relate to the actual data.
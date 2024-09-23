# The Quant Science Newsletter
# QS 020: Financial Functions for Python

# READY TO LEARN ALGORITHMIC TRADING WITH US?
# Register for our Course Waitlist: 
# https://learn.quantscience.io/python-algorithmic-trading-course-waitlist

# Libraries and Data
import ffn
import pandas as pd

prices = ffn.get("AAPL, GOOGL, MSFT, JPM, NVDA", start="2023-01-01", end="2024-09-20")

prices


# Performance Analysis
perf = prices.calc_stats()
perf.display()

# Lookback Returns
perf.display_lookback_returns()

# Monthly Returns
for asset in prices.columns:
    print(f"\nMonthly Returns for {asset.upper()}:")
    perf[asset].display_monthly_returns()

# Performance Plot
perf.plot()

# Correlations
perf.plot_correlation()

# Drawdowns
drawdowns = perf.prices.to_drawdown_series()
drawdowns.plot()

# BONUS: Get all of the stats as a data frame
df = pd.DataFrame()
for asset in list(perf.keys()):
    stats = perf[asset].stats
    stats.name = asset
    df = pd.concat([df, stats], axis= 1)
df
    
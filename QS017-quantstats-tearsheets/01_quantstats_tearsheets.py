# The Quant Science Newsletter
# QS 017: How to make awesome tear sheets in Python

# READY TO LEARN ALGORITHMIC TRADING WITH US?
# Register for our Course Waitlist: 
# https://learn.quantscience.io/python-algorithmic-trading-course-waitlist


import quantstats as qs

# fetch the daily returns for a stock
stock = qs.utils.download_returns('META')

# show sharpe ratio
qs.stats.sharpe(stock)

# Performance Snapshot Plot
qs.plots.snapshot(stock, title='Facebook Performance', show=True)

# Create an HTML Tear Sheet
qs.reports.html(stock, "SPY")

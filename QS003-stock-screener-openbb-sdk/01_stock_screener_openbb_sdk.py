# The Quant Science Newsletter
# QS 003: Make a Stock Screener with Python
# Copyright: Quant Science, LLC

# Requires openbb==^3.2.4

from openbb_terminal.sdk import openbb
import pandas as pd

# 1.0 RUN A STOCK SCREENER ----

# Stock screening with FinViz
new_highs = openbb.stocks.screener.screener_data("new_high")
new_highs

# Filter Price > $15.00 and Country == USA
portfolio_data = new_highs[
    (new_highs.Price > 15) &
    (new_highs.Country == "USA")
]

# Save screener portfolio
portfolio_data.to_csv("QS003-openbb-sdk/new_highs.csv")

# Load screener portfolio
portfolio_data = pd.read_csv("QS003-openbb-sdk/new_highs.csv")

portfolio_data

# Get the tickers as a list
tickers = portfolio_data.Ticker.tolist()

# 2.0 GET STOCK DATA FOR SCREENED TICKERS----

# Get Stock Data for each Ticker
#   NOTE: Takes a minute to run...
stock_data = openbb.economy.index(
    tickers, 
    start_date="2016-01-01", 
    end_date="2019-12-30"
)

# Save
stock_data.to_csv("QS003-openbb-sdk/stock_data.csv")

# Load
stock_data = pd.read_csv(
    "QS003-openbb-sdk/stock_data.csv",     
    index_col="Date"
)

stock_data

# Visualize a sample of the stocks
stock_data['DELL'].plot(title="Dell")

stock_data[['DELL','GOOG', 'V']].plot(
    subplots=True
)


# WANT MORE ALGORITHMIC TRADING HELP?
# Register for our Course Waitlist: 
# https://learn.quantscience.io/python-algorithmic-trading-course-waitlist


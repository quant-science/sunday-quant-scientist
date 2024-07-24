# The Quant Science Newsletter
# QS 002: Using Python to Automate Trades
# Copyright: Quant Science, LLC

# Requires openbb==^3.2.4

import pandas as pd
from openbb_terminal.sdk import openbb
import riskfolio as rp

# GET NEW HIGHS
new_highs = openbb.stocks.screener.screener_data("new_high")

new_highs = pd.read_csv("QS002-automate-trades/new_highs.csv")

port_data = new_highs[(new_highs.Price>15) & (new_highs.Country == "USA")]

port_data

tickers = port_data.Ticker.tolist()
tickers

# GET PRICES & RETURNS

data = openbb.economy.index(tickers, start_date = "2016-01-01", end_date="2019-12-30")

data

returns = data.pct_change()[1:]

returns.dropna(how="any", axis=1, inplace=True)

returns

# RISKFOLIO

port = rp.Portfolio(returns=returns)

port.assets_stats(method_mu='hist', method_cov='hist', d=0.94)

port.lowerret = 0.0008

w_rp_c = port.rp_optimization(
    model = "Classic",
    rm = "MV",
    hist = True,
    rf = 0,
    b = None
)

w_rp_c

port_val = 10_000

w_rp_c["invest_amt"] = w_rp_c * port_val

w_rp_c["last_price"] = data.iloc[-1]

w_rp_c["shares"] = (w_rp_c.invest_amt / w_rp_c.last_price).astype(int)

w_rp_c

# WANT MORE ALGORITHMIC TRADING HELP?
# Register for our Course Waitlist: 
# https://learn.quantscience.io/python-algorithmic-trading-course-waitlist

# The Quant Science Newsletter
# QS 018: What can we learn about Stanley Druckenmiller's Portfolio (Pyfolio Analysis)?

# READY TO LEARN ALGORITHMIC TRADING WITH US?
# Register for our Course Waitlist: 
# https://learn.quantscience.io/python-algorithmic-trading-course-waitlist

# https://x.com/marketplunger1/status/1813311433075286424 


# Libraries

import pandas as pd
import pandas as pd
import yfinance as yf
import riskfolio as rp
import pyfolio as pf

# DATA FROM 13F FILING ----

data = {
    "stock": ["IWM", "MSFT", "CPNG", "TECK", "VST", "NTRA", "NVDA", "COHR", "GE", "WWD", "STX", "ANET", "FLEX", "ZI", "NWSA", "DFS", "VRT", "KMI", "FCX", "KBR", "MRVL", "WAB", "LLY", "PANW", "CHX"],
    "sector": ["FINANCE", "INFORMATION TECHNOLOGY", "CONSUMER DISCRETIONARY", "MATERIALS", "ENERGY", "HEALTH CARE", "INFORMATION TECHNOLOGY", "INDUSTRIALS", "INDUSTRIALS", "INDUSTRIALS", "INFORMATION TECHNOLOGY", "INFORMATION TECHNOLOGY", "INFORMATION TECHNOLOGY", "COMMUNICATIONS", "COMMUNICATIONS", "FINANCE", "FINANCE", "UTILITIES AND TELECOMMUNICATIONS", "MATERIALS", "INDUSTRIALS", "INFORMATION TECHNOLOGY", "INDUSTRIALS", "HEALTH CARE", "INFORMATION TECHNOLOGY", "ENERGY"],
    "shares_held_or_principal_amt": [3157900, 1112270, 22452850, 452247, 2625231, 1929380, 1759430, 2525070, 1068928.04, 954230, 1438983, 428252, 3863155, 5818906, 3312025, 645205, 998510, 3880500, 1378875, 981425, 822875, 386835, 61464, 138718, 848710],
    "market_value": [664106000, 467954000, 399561000, 208396000, 182847000, 176461000, 158975000, 153070000, 149744000, 147066000, 133897000, 124185000, 110525000, 94287000, 86709000, 84580000, 81548000, 71168000, 64835000, 62478000, 58325000, 56354000, 47972000, 39414000, 36766000],
    "pct_of_portfolio": [15.12, 10.65, 9.10, 4.72, 4.16, 4.02, 3.62, 3.49, 3.41, 3.35, 3.05, 2.83, 2.52, 2.15, 1.97, 1.93, 1.86, 1.55, 1.48, 1.42, 1.33, 1.28, 1.09, 0.90, 0.84],
    "previous_pct_of_portfolio": [0, 12.19, 11.07, 0, 2.74, 1.67, 9.12, 0, 2.80, 1.64, 5.39, 1.65, 1.03, 0.51, 2.96, 0, 3.32, 0, 0.60, 1.92, 0.65, 0, 7.00, 0.57, 1.71],
    "rank": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25],
    "change_in_shares": [3157900, 26150, -455990, -972375, 240170, 1036350, -4415510, 2525070, 146691, 54595, -677125, 194067, 2728755, 4957086, -733095, 645205, -1314990, 3880500, 907340, -177110, 459290, 386835, -340886, 74043, -483915],
    "pct_change": ["New", 2.41, -1.98, -17.71, 10.07, 116.05, -71.51, "New", 15.89, 135.82, -32.00, 82.87, 240.55, 536.50, -18.12, "New", -56.84, "New", 192.42, -15.29, 126.32, "New", -84.68, 114.35, -36.31]
}

df = pd.DataFrame(data)

df

# PORTFOLIO ANALYSIS AND OPTIMIZATION ---- 

# Step 1: Collect and Format Data

stock_data = yf.download(df['stock'].tolist(), start = '2023-07-16', end = '2024-07-16')

stock_data = stock_data.loc[:, "Adj Close"]

stock_data

stock_data.dropna().plot()

# Step 2: Get Returns

returns = stock_data.pct_change().dropna()

returns

# Step 3: Get Portfolio Returns

w = df[['stock', 'pct_of_portfolio']].set_index('stock').sort_index()

portfolio_returns = (returns * w.pct_of_portfolio).sum(axis=1)

portfolio_returns.cumsum().plot()

portfolio_returns.plot()

pf.show_perf_stats(portfolio_returns)

pf.plot_drawdown_periods(portfolio_returns)

pf.plot_rolling_sharpe(portfolio_returns)


# Step 4: Optimization

port = rp.Portfolio(returns)

w = port.optimization(
    model = "Classic",
    rm = "CVaR",
    obj = "Sharpe",
    hist = True,
    rf = 0,
    l = 0
)
w


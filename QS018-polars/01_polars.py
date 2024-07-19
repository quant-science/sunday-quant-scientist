# The Quant Science Newsletter
# QS 018: Algorithmic Trading and Quantitative Finance Analysis with Polars

# READY TO LEARN ALGORITHMIC TRADING WITH US?
# Register for our Course Waitlist: 
# https://learn.quantscience.io/python-algorithmic-trading-course-waitlist


# LIBRARIES

import polars as pl
import pandas as pd
import yfinance as yf

# STANLEY DRUCKENMILLER'S FUND DATA FROM 13F FILING ----

stock_list = ["IWM", "MSFT", "CPNG", "TECK", "VST", "NTRA", "NVDA", "COHR", "GE", "WWD", "STX", "ANET", "FLEX", "ZI", "NWSA", "DFS", "VRT", "KMI", "FCX", "KBR", "MRVL", "WAB", "LLY", "PANW", "CHX"]

# COLLECT STOCK DATA

stock_data = yf.download(stock_list, start = '2023-07-16', end = '2024-07-16')

stock_data = stock_data.loc[:, "Adj Close"].dropna()

stock_data    

# CONVERT TO POLARS

stock_data_pl = pl.DataFrame(stock_data.reset_index())

stock_data_pl

# PIVOT TO LONG FORMAT

stock_data_long_pl = stock_data_pl.melt(
    id_vars="Date", 
    value_name="Price",
    variable_name="Stock"
)

stock_data_long_pl

# PLOTTING STOCK PRICES

stock_data_long_pl.plot.line(x="Date", y = "Price", by = "Stock")

# MOVING AVERAGES

moving_average_pl = stock_data_long_pl.with_columns([
    pl.col("Price").rolling_mean(10).over("Stock").alias("Price_MA10"),
    pl.col("Price").rolling_mean(50).over("Stock").alias("Price_MA50"),
])

moving_average_pl

# VISUALIZE ONE OF THE STOCKS

STOCK_SYMBOL = "NVDA"

moving_average_pl.filter(
        pl.col("Stock") == STOCK_SYMBOL
    ).plot.line(
        x = "Date",
        y = ["Price", "Price_MA10", "Price_MA50"],
        by = "Stock",
        groupby = "Stock",
        # subplots = True,
    )

# ADD RETURNS AND ROLLING SHARPE BY GROUP

window_size = 50

rolling_sharpe_pl = moving_average_pl.with_columns(
    (pl.col("Price") / pl.col("Price").shift(1) - 1).over("Stock").alias("Return")
).with_columns(
    (pl.col("Return").rolling_mean(window_size) / pl.col("Return").rolling_std(window_size)).over("Stock").alias("Rolling_Sharpe")
)

rolling_sharpe_pl.plot.line(
    x = "Date",
    y = ["Rolling_Sharpe"],
    by = "Stock",
    title = "50-Day Rolling Sharpe Ratio",
    height = 700,
)





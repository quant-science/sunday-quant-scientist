# The Quant Science Newsletter
# QS 046: How to download free market data

# READY TO LEARN ALGORITHMIC TRADING WITH US?
# Join our free webinar where we will give you a sneak peek into the strategies and Python code that power our hedge fund:
# https://learn.quantscience.io/qs-register

# Import the two libraries we will use
import pandas as pd
import yfinance as yf

# Define the stock symbols we want to download data for
stock_symbols = ["AAPL", "MSFT", "GOOGL"]

# Download the market data for the stock symbols
market_data = {
    symbol: yf.Ticker(symbol).history(period="1y") for symbol in stock_symbols
}
options_data = {symbol: yf.Ticker(symbol).option_chain() for symbol in stock_symbols}
financials_data = {symbol: yf.Ticker(symbol).financials for symbol in stock_symbols}

# Organize and standardize the data
standardized_market_data = []
for symbol, df in market_data.items():
    if not df.empty:
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df = df.dropna()
        df = df.astype(float)
        df["Symbol"] = symbol
        standardized_market_data.append(df)
market_df = pd.concat(standardized_market_data).reset_index()

# Organize and standardize the options data
standardized_options_data = []
for symbol, option_chain in options_data.items():
    calls = option_chain.calls
    puts = option_chain.puts
    for opt_type, opt_df in [("call", calls), ("put", puts)]:
        if not opt_df.empty:
            opt_df = opt_df[
                [
                    "contractSymbol",
                    "strike",
                    "lastPrice",
                    "bid",
                    "ask",
                    "volume",
                    "openInterest",
                ]
            ]
            opt_df = opt_df.dropna()
            opt_df = opt_df.astype(
                {
                    "strike": float,
                    "lastPrice": float,
                    "bid": float,
                    "ask": float,
                    "volume": float,
                    "openInterest": float,
                }
            )
            opt_df["Type"] = opt_type
            opt_df["Symbol"] = symbol
            standardized_options_data.append(opt_df)
options_df = pd.concat(standardized_options_data).reset_index(drop=True)

# Organize and standardize the financials data
standardized_financials_data = []
for symbol, df in financials_data.items():
    if not df.empty:
        df = df.transpose()
        df["Symbol"] = symbol
        df = df.dropna(axis=1, how="all")
        standardized_financials_data.append(df)
financials_df = pd.concat(standardized_financials_data).reset_index()

# Drop any rows with null values
if market_df.isnull().values.any():
    market_df = market_df.dropna()
if options_df.isnull().values.any():
    options_df = options_df.dropna()
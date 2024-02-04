# The Quant Science Newsletter
# QS 012: PYTIMETK Finance Module

# READY TO LEARN ALGORITHMIC TRADING WITH US?
# Register for our Course Waitlist: 
# https://learn.quantscience.io/python-algorithmic-trading-course-waitlist


# Libraries and Data
import pandas as pd
import yfinance as yf
import pytimetk as tk

raw_df = yf.download(['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA'], start='2015-01-01', end='2023-12-31', progress=False)

df = raw_df \
    .stack() \
    .reset_index(level=1, names=["", "Symbol"]) \
    .sort_values(by="Symbol") \
    .reset_index() 

df.glimpse()

# MACD (Polars Backend)

macd_df = df \
    .groupby('Symbol') \
    .augment_macd(
        date_column = 'Date', 
        close_column = 'Close', 
        fast_period = 12, 
        slow_period = 26, 
        signal_period = 9, 
        engine = "polars"
    )
    
macd_df.glimpse()

# Bollinger Bands (Polars Backend)

bbands_df = df \
    .groupby('Symbol') \
    .augment_bbands(
        date_column = 'Date', 
        close_column = 'Close', 
        periods = [20, 40, 60], 
        std_dev = 2, 
        engine = "polars"
    )
    
bbands_df.glimpse()

# CHAINING FEATURES (Polars Backend)

features_df = df \
    .groupby('Symbol') \
    .augment_macd(
        date_column = 'Date', 
        close_column = 'Close', 
        fast_period = 12, 
        slow_period = 26, 
        signal_period = 9, 
        engine = "polars"
    ) \
    .groupby('Symbol') \
    .augment_bbands(
        date_column = 'Date', 
        close_column = 'Close', 
        periods = [20, 40, 60], 
        std_dev = 2, 
        engine = "polars"
    ) \
    .groupby('Symbol') \
    .augment_cmo(
        date_column = 'Date', 
        close_column = 'Close', 
        periods = [14,28], 
        engine = "polars"
    ) \
    .augment_timeseries_signature(
        date_column = 'Date',
    )

features_df.glimpse()
    


# READY TO LEARN ALGORITHMIC TRADING WITH US?
# Register for our Course Waitlist: 
# https://learn.quantscience.io/python-algorithmic-trading-course-waitlist


# The Quant Science Newsletter
# QS 009: Average True Range (ATR)

# STEP 1: Load the data

from openbb_terminal.sdk import openbb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pytimetk as tk

import matplotlib.pyplot as plt
plt.rcdefaults()


SYMBOL = "SPY"
START = "2021-09-30"
END = "2023-12-13"

# Load the data
df = openbb.stocks.load(SYMBOL, start_date=START, end_date=END)

df \
    .reset_index() \
    .plot_timeseries(
        date_column="date",
        value_column="Close",
        title="SPY Close",
        x_lab="Date",
        y_lab="Close",
    )

# STEP 2: Apply ATR
def calculate_atr(df, period=14):
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=period).mean()
    df.drop(['H-L', 'H-PC', 'L-PC', 'TR'], axis=1, inplace=True)
    return df

df_with_atr = calculate_atr(df, period = 14)

# STEP 3: Plotting

# Plotting both the ATR and the original series (High, Low, Close) on the same chart
plt.figure(figsize=(10, 8))

# Plotting High, Low, and Close
plt.subplot(2, 1, 1)
plt.plot(df.index, df['High'], label='High', color='green')
plt.plot(df.index, df['Low'], label='Low', color='red')
plt.plot(df.index, df['Close'], label='Close', color='blue')
plt.title('Stock Prices (High, Low, Close)')
plt.ylabel('Price')
plt.legend()

# Plotting ATR
plt.subplot(2, 1, 2)
plt.plot(df_with_atr.index, df_with_atr['ATR'], label='ATR', color='orange')
plt.title('Average True Range (ATR)')
plt.xlabel('Date')
plt.ylabel('ATR Value')
plt.legend()

plt.tight_layout()
plt.show()

# WANT MORE ALGORITHMIC TRADING HELP?
# Register for our Course Waitlist: 
# https://learn.quantscience.io/python-algorithmic-trading-course-waitlist
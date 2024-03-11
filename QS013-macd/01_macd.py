# The Quant Science Newsletter
# QS 012: Moving Average Convergence Divergence (MACD)

# STEP 1: Load the data

from openbb_terminal.sdk import openbb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pytimetk as tk

import matplotlib.pyplot as plt
plt.rcdefaults()


SYMBOL = "NVDA"
START = "2021-09-30"
END = "2024-03-01"

# Load the data
df = openbb.stocks.load(SYMBOL, start_date=START, end_date=END)

df \
    .reset_index() \
    .plot_timeseries(
        date_column="date",
        value_column="Close",
        title=f"{SYMBOL} Close",
        x_lab="Date",
        y_lab="Close",
    )
    
# STEP 2: Calculate MACD
def calculate_macd(df, short_period=12, long_period=26, signal_period=9):
    df['EMA_short'] = df['Close'].ewm(span=short_period, adjust=False).mean()
    df['EMA_long'] = df['Close'].ewm(span=long_period, adjust=False).mean()
    df['MACD'] = df['EMA_short'] - df['EMA_long']
    df['Signal_Line'] = df['MACD'].ewm(span=signal_period, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
    df['Bullish_Crossover'] = (df['MACD'] > df['Signal_Line']) & (df['MACD'].shift(1) <= df['Signal_Line'].shift(1))
    df['Bearish_Crossover'] = (df['MACD'] < df['Signal_Line']) & (df['MACD'].shift(1) >= df['Signal_Line'].shift(1))

    return df

df_with_macd = calculate_macd(df)

df_with_macd.glimpse()

# STEP 3: Plotting

plt.figure(figsize=(10, 12)) # Increase the figure size to accommodate the new subplot

# Plotting Close Price
plt.subplot(3, 1, 1) # Change subplot grid to 3 rows, 1 column, and this is the 1st subplot
plt.plot(df.index, df['Close'], label='Close', color='blue')
plt.title(f'{SYMBOL} Close Price')
plt.ylabel('Price')
plt.legend()

# Plotting MACD and Signal Line
plt.subplot(3, 1, 2) # This is the 2nd subplot
plt.plot(df_with_macd.index, df_with_macd['MACD'], label='MACD', color='green')
plt.plot(df_with_macd.index, df_with_macd['Signal_Line'], label='Signal Line', color='red')

# Highlight Bullish Crossovers
plt.scatter(df_with_macd[df_with_macd['Bullish_Crossover']].index, df_with_macd[df_with_macd['Bullish_Crossover']]['MACD'], color='blue', label='Bullish Crossover', marker='^', alpha=1, s=100)

# Highlight Bearish Crossovers
plt.scatter(df_with_macd[df_with_macd['Bearish_Crossover']].index, df_with_macd[df_with_macd['Bearish_Crossover']]['MACD'], color='red', label='Bearish Crossover', marker='v', alpha=1, s=100)

plt.title('Moving Average Convergence Divergence (MACD) and Signal Line')
plt.ylabel('MACD Value')
plt.legend()

# Plotting MACD Histogram
plt.subplot(3, 1, 3) # This is the 3rd subplot
plt.bar(df_with_macd.index, df_with_macd['MACD_Histogram'], label='MACD Histogram', color='purple')
plt.title('MACD Histogram')
plt.xlabel('Date')
plt.ylabel('Histogram Value')
plt.legend()

plt.tight_layout() # Adjust layout to not overlap
plt.show()


# ===
# Correlation Analysis

# Make forward 5-day returns
df_with_macd['5-day-forward-return'] = df_with_macd['Close'].shift(-5) / df_with_macd['Close'] - 1

# Calculate rolling correlation
df_with_macd['rolling_corr'] = df_with_macd['5-day-forward-return'] \
    .rolling(window=30) \
    .corr(df_with_macd['MACD']) 
    
df_with_macd \
    .reset_index() \
    .plot_timeseries(
        "date", "rolling_corr"
    )

df_with_macd['rolling_corr'].describe()
# The Quant Science Newsletter
# QS 010: Relative Strength Index (RSI)

# WANT TO LEARN ALGORITHMIC TRADING?
# Register for our Course Waitlist: 
# https://learn.quantscience.io/python-algorithmic-trading-course-waitlist

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
    
# STEP 2: Apply RSI

# Calculate the daily price changes
delta = df['Close'].diff()

# Separate gains and losses
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

# Calculate the exponential moving average (EMA) of gains and losses
window_length = 14
avg_gain = gain.rolling(window=window_length, min_periods=window_length).mean()
avg_loss = loss.rolling(window=window_length, min_periods=window_length).mean()

# Calculate RS
RS = avg_gain / avg_loss

# Calculate RSI
RSI = 100 - (100 / (1 + RS))

# Add RSI to the data frame
df['RSI'] = RSI

# Print the data frame
df

# STEP 3: Plotting

# Create a figure and a set of subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

# Plot Closing Prices
ax1.plot(df['Close'], color='blue')
ax1.set_title(f'{SYMBOL} Closing Price')
ax1.set_ylabel('Price')

# Plot RSI
ax2.plot(df['RSI'], color='green')
ax2.set_title('Relative Strength Index (RSI)')
ax2.set_ylabel('RSI')
ax2.set_ylim([0, 100])  # RSI ranges from 0 to 100
ax2.axhline(70, color='red', linestyle='--')  # Overbought line
ax2.axhline(30, color='red', linestyle='--')  # Oversold line

# Show the plot
plt.tight_layout()
plt



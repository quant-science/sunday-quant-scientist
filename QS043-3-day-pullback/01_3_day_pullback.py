# The Quant Science Newsletter
# QS 043: A new strategy with 77% win rate (with Python code)

# READY TO LEARN ALGORITHMIC TRADING WITH US?
# Register for our Course Waitlist:
# https://learn.quantscience.io/python-algorithmic-trading-course-waitlist


import pandas as pd
import numpy as np
import yfinance as yf

# Define the ticker symbol and download historical data from Yahoo Finance

ticker = "SPY"
data = yf.download(ticker).Close[["SPY"]]
data.columns = ["close"]

# Calculate daily returns by finding the difference between consecutive closing prices

data["return"] = data.close.diff()

# Identify days with negative returns to find losing days in the dataset

data["down"] = data["return"] < 0

# Identify 3-day losing streaks by checking for three consecutive losing days

data["3_day_losing_streak"] = (
    data["down"] & data["down"].shift(1) & data["down"].shift(2)
)

# Initialize a column to track the number of days since the last 3-day losing streak

data["days_since_last_streak"] = np.nan

# Iterate over the data to calculate the days since the last 3-day losing streak

last_streak_day = -np.inf  # Initialize with a very large negative value

for i in range(len(data)):
    if data["3_day_losing_streak"].iloc[i]:
        if i - last_streak_day >= 42:  # Check if it's been at least 42 trading days
            data.loc[data.index[i], "days_since_last_streak"] = i - last_streak_day
        last_streak_day = i

# Filter the data to show only the occurrences that meet the criteria

result = data.dropna(subset=["days_since_last_streak"]).copy()

# Calculate future returns following the identified streaks

result["next_1_day_return"] = data.close.shift(-1) / data.close - 1
result["next_5_day_return"] = data.close.shift(-5) / data.close - 1
result["next_10_day_return"] = data.close.shift(-10) / data.close - 1
result["next_21_day_return"] = data.close.shift(-21) / data.close - 1

# Print the mean future returns for different time horizons

cols = [
    "next_1_day_return",
    "next_5_day_return",
    "next_10_day_return",
    "next_21_day_return"
]
print(result[cols].mean())

# Plot the proportion of positive returns for the different time horizons

result[cols].gt(0).mean().plot.bar()

# Display the proportion of positive returns for the different time horizons

result[cols].gt(0).mean()



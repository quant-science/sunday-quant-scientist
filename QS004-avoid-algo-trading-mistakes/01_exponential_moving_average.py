# The Quant Science Newsletter
# QS 004: Avoid Mistakes in Algorithmic Trading
# Copyright: Quant Science, LLC

# 1.0 IMPORT LIBRARIES ----

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 2.0 CREATE STOCK PRICE DATA ----

data = {
    'date': pd.date_range('2020-01-01', periods=10, freq='B'),
    'price': [100, 103, 107, 105, 108, 107, 115, 110, 108, 112]
}
df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['date']) 
df.set_index('date', inplace=True)

df

# MOVING AVERAGES ----

# Window size for moving averages
window_size = 3

# Simple Moving Average (SMA)
df['sma'] = df['price'].rolling(window=window_size, min_periods=1).mean()

# Exponential Moving Average (EMA)
df['ema'] = df['price'].ewm(span=window_size, adjust=False).mean()

# KALMAN FILTER ----

# Kalman Filter Initialization
initial_state = df['price'][0]
state_estimate = initial_state

# Kalman Filter Parameters
# - Can be tuned to find the optimal filter
estimation_error = 0.5  # Initial estimation error.
process_variance = 0.1  # The process variance.
measurement_variance = 0.1  # The measurement variance.

kf = []

for price in df['price']:
    # Prediction
    prediction = state_estimate  # In a 1D constant model, the prediction is the previous state estimate.
    prediction_error = estimation_error + process_variance
    
    # Update
    kalman_gain = prediction_error / (prediction_error + measurement_variance)
    state_estimate = prediction + kalman_gain * (price - prediction)
    estimation_error = (1 - kalman_gain) * prediction_error
    
    kf.append(state_estimate)

df['kf'] = kf  # Adding Kalman Filter estimates to the DataFrame.

df

# 3.0 PLOT ----

# Matplotlib Plot
plt.style.use('dark_background')
plt.figure(figsize=(10,6))
plt.plot(df.index, df['price'], label='Original Price', marker='o', alpha=0.5, color='lime')
plt.plot(df.index, df['sma'], label=f'{window_size}-Day SMA', linestyle='dashed', color='cyan')
plt.plot(df.index, df['ema'], label=f'{window_size}-Day EMA', linestyle='dotted', color='magenta')
plt.plot(df.index, df['kf'], label='Kalman Filter', linestyle='solid', color='yellow')
plt.title('Price with SMA, EMA and Kalman Filter', color='white')
plt.xlabel('Date', color='white')
plt.ylabel('Price', color='white')
plt.legend()
plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

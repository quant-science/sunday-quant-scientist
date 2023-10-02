import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = {
    'date': pd.date_range('2020-01-01', periods = 10, freq = 'B'), # business day date offset
    'price': [100, 103, 107, 105, 108, 107, 115, 110, 108, 112]
}

df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df.date)
df.set_index('date', inplace = True)

df

# MOVING AVERAGES ----

window_size = 3

# Simple MA
df['sma'] = df['price'].rolling(window = window_size, min_periods=1).mean()
# Exponential MA
df['ema'] = df['price'].ewm(span = window_size, adjust = False).mean()

df

# KALMAN FILTER ----

initial_state = df['price'][0]
state_estimate = initial_state

# Filter parameters --> tuned to find optimal filter
estimation_error = 0.5
process_variance = 0.1
measurement_variance = 0.1

kf = []

for price in df['price']:
    prediction = state_estimate
    prediction_error = estimation_error + process_variance
    
    kalman_gain = prediction_error / (prediction_error + measurement_variance)
    state_estimate

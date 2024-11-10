import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn import hmm

# Download historical price data for SPY from Yahoo Finance

data = yf.download("SPY")

# Calculate log returns of the closing prices

returns = np.log(data.Close / data.Close.shift(1))

# Calculate the range as the difference between high and low prices

range = (data.High - data.Low)

# Concatenate returns and range into a single DataFrame and drop any missing values

features = pd.concat([returns, range], axis=1).dropna()
features.columns = ["returns", "range"]

# Initialize a Gaussian Hidden Markov Model with 3 states and fit it to the features

model = hmm.GaussianHMM(
    n_components=3,
    covariance_type="full",
    n_iter=1000,
)
model.fit(features)

# Predict the hidden states for the given features and store them in a Series

states = pd.Series(model.predict(features), index=data.index[1:])
states.name = "state"

# Plot a histogram of the hidden states

states.hist()

# Define a color map for the different states

color_map = {
    0.0: "green",
    1.0: "orange",
    2.0: "red"
}

# Concatenate the closing prices and the states, drop missing values, 
# set state as a hierarchical index, unstack the state index, and plot the closing prices with different colors for each state

(
    pd.concat([data.Close, states], axis=1)
    .dropna()
    .set_index("state", append=True)
    .Close
    .unstack("state")
    .plot(color=color_map)
)

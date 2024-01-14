


import yfinance as yf
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import sklearn.decomposition as PCA


# Load the data
symbols = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "LQD", "GLD", "USO", "VNQ"]
start_date = "2021-01-31"
end_date = "2023-12-28"

data = yf.download(symbols, start=start_date, end=end_date, auto_adjust=True)

portfolio_returns = data["Close"].pct_change().dropna()

portfolio_returns.plot(title="Portfolio Returns", figsize=(12, 6))

pca = PCA.PCA(n_components=3)
pca.fit(portfolio_returns)

pct = pca.explained_variance_ratio_
pct

sum(pct)

pca_components = pca.components_
pca_components

# Factor Returns

X = np.array(portfolio_returns)

factor_returns = X.dot(pca_components.T)

factor_returns = pd.DataFrame(
    columns=["factor_1", "factor_2", "factor_3"],
    index=portfolio_returns.index,
    data=factor_returns,
)
    

# Factor Exposures

factor_exposures = pd.DataFrame(
    data=pca_components,
    index=["factor_1", "factor_2", "factor_3"],
    columns=portfolio_returns.columns,
).T

# Mean Portfolio Returns by date

mean_portfolio_returns = portfolio_returns.mean(axis=1)

# Regress the portfolio returns on the factor exposures

from statsmodels.api import OLS

model = OLS(mean_portfolio_returns, factor_returns).fit()

model.summary()

beta = model.params


hedged_portfolio_returns = -1 * beta * factor_returns + mean_portfolio_returns
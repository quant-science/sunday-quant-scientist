# The Quant Science Newsletter
# QS 011: SKFOLIO Mean Risk Optimization

# READY TO LEARN ALGORITHMIC TRADING?
# Register for our Course Waitlist: 
# https://learn.quantscience.io/python-algorithmic-trading-course-waitlist


# Libraries and Data
import pandas as pd
import yfinance as yf

from sklearn.model_selection import train_test_split

from skfolio import Population, RiskMeasure
from skfolio.optimization import InverseVolatility, RiskBudgeting
from skfolio.preprocessing import prices_to_returns

prices = yf.download(['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA'], start='2015-01-01', end='2023-12-31', progress=False)

prices = prices['Adj Close']

# Train Test Split

X = prices_to_returns(prices)
X_train, X_test = train_test_split(X, test_size=0.75, shuffle=False)

X_train.head()

# Risk Parity

model = RiskBudgeting(
    risk_measure=RiskMeasure.VARIANCE,
    portfolio_params=dict(name="Risk Parity - Variance"),
)
model.fit(X_train)
model.weights_

# Benchmark

benchmark = InverseVolatility(
    portfolio_params=dict(name="Inverse Volatility")
)
benchmark.fit(X_train)
benchmark.weights_


# Predictions AND Evaluation

pred_model = model.predict(X_test)
pred_bench = benchmark.predict(X_test)

print(pred_model.annualized_sharpe_ratio)
print(pred_bench.annualized_sharpe_ratio)

# Visualizations

population = Population([pred_model, pred_bench])

population.plot_composition()

population.plot_cumulative_returns()

population.summary()

pred_model.plot_contribution(RiskMeasure.ANNUALIZED_VARIANCE)
pred_bench.plot_contribution(RiskMeasure.ANNUALIZED_VARIANCE)

# READY TO LEARN ALGORITHMIC TRADING?
# Register for our Course Waitlist: 
# https://learn.quantscience.io/python-algorithmic-trading-course-waitlist


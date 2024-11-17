import yfinance as yf
import riskfolio as rf
import pandas as pd
import warnings
pd.options.display.float_format = "{:.4%}".format
warnings.filterwarnings("ignore")


mag_7 = [
    "AMZN",
    "AAPL",
    "NVDA",
    "META",
    "TSLA",
    "MSFT",
    "GOOG",
]

factors = ["MTUM", "QUAL", "VLUE", "SIZE", "USMV"]

start = "2020-01-01"
end = "2024-07-31"


port_returns = (
    yf
    .download(
        mag_7, 
        start=start, 
        end=end
    )["Adj Close"]
    .pct_change()
    .dropna()
)

factor_returns = (
    yf
    .download(
        factors, 
        start=start, 
        end=end
    )["Adj Close"]
    .pct_change()
    .dropna()
)


port = rf.Portfolio(returns=port_returns)

port.assets_stats(method_mu="hist", method_cov="ledoit")

port.lowerret = 0.00056488 * 1.5

loadings = rf.loadings_matrix(
    X=factor_returns,
    Y=port_returns, 
    feature_selection="PCR",
    n_components=0.95
)

loadings.style.format("{:.4f}").background_gradient(cmap='RdYlGn')


port.factors = factor_returns
port.factors_stats(
    method_mu="hist",
    method_cov="ledoit",
    dict_load=dict(
        n_components=0.95  # 95% of explained variance.
    )
)

w = port.optimization(
    model="FM",  # Factor model
    rm="MV",  # Risk measure used, this time will be variance
    obj="Sharpe",  # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
    hist=False,  # Use risk factor model for expected returns
)

ax = rf.plot_pie(
    w=w,
    title='Sharpe FM Mean Variance',
    others=0.05,
    nrow=25,
    cmap="tab20"
)

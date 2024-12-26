# The Quant Science Newsletter
# QS 031: Use the Kelly criterion for optimal position sizing

# READY TO LEARN ALGORITHMIC TRADING WITH US?
# Register for our Course Waitlist: 
# https://learn.quantscience.io/python-algorithmic-trading-course-waitlist

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.integrate import quad
from scipy.stats import norm
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

# Fetch annual returns for the S&P 500 index since 1950 and compute rolling mean and standard deviation over a 25-year window

annual_returns = (
    yf.download(["^GSPC"])[["Adj Close"]]
    .resample("YE")
    .last()
    .pct_change()
    .dropna()
    .rename(columns={"Adj Close": "^GSPC"})
)

return_params = annual_returns["^GSPC"].rolling(25).agg(["mean", "std"]).dropna()

# Define a function to calculate the negative value of the expected log return

def norm_integral(f, mean, std):
    """Calculates the negative expected log return
    
    Parameters
    ----------
    f : float
        Leverage factor
    mean : float
        Mean return
    std : float
        Standard deviation of returns
    
    Returns
    -------
    float
        Negative expected log return
    """
    val, er = quad(
        lambda s: np.log(1 + f * s) * norm.pdf(s, mean, std),
        mean - 3 * std,
        mean + 3 * std,
    )
    return -val

# Define a function to optimize the Kelly fraction using the minimize_scalar method

def get_kelly(data):
    """Optimizes the Kelly fraction
    
    Parameters
    ----------
    data : pd.Series
        Contains mean and standard deviation of returns
    
    Returns
    -------
    float
        Optimal Kelly fraction
    """
    solution = minimize_scalar(
        norm_integral, args=(data["mean"], data["std"]), bounds=[0, 2], method="bounded"
    )
    return solution.x

# Calculate the Kelly fraction for each rolling window and add it to the annual returns DataFrame. Then visualize the cumulative compounded returns using the Kelly strategy

annual_returns["f"] = return_params.apply(get_kelly, axis=1)

(
    annual_returns[["^GSPC"]]
    .assign(kelly=annual_returns["^GSPC"].mul(annual_returns.f.shift()))
    .dropna()
    .loc["1900":]
    .add(1)
    .cumprod()
    .sub(1)
    .plot(lw=2)
)

# Pick an arbitrary point for mean and standard deviation to calculate optimal Kelly fraction. Optimize the Kelly fraction for the given mean and standard deviation. This formula can result in Kelly fractions higher than 1. In this case, it is theoretically advantageous to use leverage to purchase additional securities on margin.

m = .058
s = .216

sol = minimize_scalar(norm_integral, args=(m, s), bounds=[0.0, 2.0], method="bounded")
print("Optimal Kelly fraction: {:.4f}".format(sol.x))

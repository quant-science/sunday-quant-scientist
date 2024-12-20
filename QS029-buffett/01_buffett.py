# The Quant Science Newsletter
# QS 029: Use Python to save 197,291 financial ratios like Warren Buffet

# READY TO LEARN ALGORITHMIC TRADING WITH US?
# Register for our Course Waitlist: 
# https://learn.quantscience.io/python-algorithmic-trading-course-waitlist

import io
import os
import time
import requests
import pandas as pd
import arcticdb as adb

# First, we establish a connection to the ArcticDB database and define a helper function to construct API URLs.

arctic = adb.Arctic("lmdb://fundamantals")
lib = arctic.get_library("financial_ratios", create_if_missing=True)

def build_fmp_url(request, period, year):
    apikey = os.environ.get("FMP_API_KEY")
    return f"https://financialmodelingprep.com/api/v4/{request}?year={year}&period={period}&apikey={apikey}"

# Next, we define a function that retrieves financial data from the FMP API and converts it into a pandas DataFrame.

def get_fmp_data(request, period, year):
    url = build_fmp_url(request, period, year)
    response = requests.get(url)
    csv = response.content.decode("utf-8")
    return pd.read_csv(io.StringIO(csv), parse_dates=True)

ratios = get_fmp_data("ratios-bulk", "quarter", "2020")
ratios

# Now, we iterate over multiple years to store their financial ratios into the ArcticDB.

for year in [2020, 2021, 2022]:
    ratios = get_fmp_data("ratios-bulk", "quarter", year)
    adb_sym = f"financial_ratios/{year}"
    adb_fcn = lib.update if lib.has_symbol(adb_sym) else lib.write
    adb_fcn(adb_sym, ratios)
    time.sleep(10)

# Finally, we define a function to filter financial data based on specific criteria and return the results as a pandas DataFrame.

def filter_by_year(year):
    cols = [
        "symbol",
        "period",
        "date",
        "debtEquityRatio", 
        "currentRatio", 
        "priceToBookRatio", 
        "returnOnEquity", 
        "returnOnAssets", 
        "interestCoverage"
    ]
    q = adb.QueryBuilder()
    filter = (
        (q["debtEquityRatio"] < 0.5)
        & (
            (q["currentRatio"] > 1.5) & (q["currentRatio"] < 2.5)
        )
        & (q["priceToBookRatio"] < 1.5)
        & (q["returnOnEquity"] > 0.08)
        & (q["returnOnAssets"] > 0.06)
        & (q["interestCoverage"] > 5)
    )
    q = q[filter]
    return lib.read(f"financial_ratios/{year}", query_builder=q).data[cols].set_index("symbol")

filter_by_year("2020")

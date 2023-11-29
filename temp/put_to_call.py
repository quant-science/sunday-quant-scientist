

from openbb_terminal.sdk import openbb
import pandas as pd
import pytimetk as tk

SYMBOL = "NVDA"
START = pd.Timestamp.today() - pd.Timedelta(days=365)
END = pd.Timestamp.today()

prices_df = openbb.stocks.load(SYMBOL, start_date=START, end_date=END)

vix = openbb.stocks.load("^VIX", start_date=START, end_date=END)

pcr = openbb.stocks.options.pcr(SYMBOL)

chains = openbb.stocks.options.chains(SYMBOL, "YahooFinance") 

pcr.reset_index() \
    .plot_timeseries(
        date_column="Date",
        value_column="PCR",
        smooth_frac=0.05,
        title = f"{SYMBOL}: Put/Call Ratio",
        x_axis_date_labels = '%Y-%m-%d',
    )
    


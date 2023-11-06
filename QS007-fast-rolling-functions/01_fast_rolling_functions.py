# The Quant Science Newsletter
# QS 007: Pytimetk Algorithmic Trading System
# Copyright: Quant Science, LLC

# 1.0 IMPORT LIBRARIES ----

# Note: I'm using pytimetk==0.2.0 (just released)

import pandas as pd
import pytimetk as tk

# 2.0 GET STOCK PRICE DATA ----

stocks_df = tk.load_dataset("stocks_daily")
stocks_df['date'] = pd.to_datetime(stocks_df['date'])

stocks_df.groupby('symbol').plot_timeseries(
    date_column = 'date',
    value_column = 'adjusted',
    facet_ncol = 2,
    width = 1100,
    height = 800,
    title = 'Stock Prices'
)

# 3.0 Fast Rolling Computations ----

# 3.1 Pandas Lambda Functions ----
%%timeit 
rolled_df_pandas_slow = stocks_df \
    .groupby('symbol') \
    .augment_rolling(
        date_column = 'date',
        value_column = 'adjusted',
        window = [20, 50, 200],
        window_func = ('mean', lambda x: x.mean()),
        engine = 'pandas',
        show_progress = False
    )
# 288 ms ± 19.5 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# 3.2 Pandas Fast Rolling Functions ----
%%timeit 
rolled_df_pandas_fast = stocks_df \
    .groupby('symbol') \
    .augment_rolling(
        date_column = 'date',
        value_column = 'adjusted',
        window = [20, 50, 200],
        window_func = 'mean',
        engine = 'pandas',
        show_progress = False
    )
# 21.1 ms ± 2.84 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

# 3.3 Polars Fast Rolling Functions ----
#  32X Faster than Pandas Slow
#  2.4X Faster than Pandas Fast
%%timeit 
rolled_df_polars = stocks_df \
    .groupby('symbol') \
    .augment_rolling(
        date_column = 'date',
        value_column = 'adjusted',
        window = [20, 50, 200],
        window_func = 'mean',
        engine = 'polars',
        show_progress = False
    )
# 8.86 ms ± 322 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)


# Bonus - plotting the results

rolled_df_pandas_fast[['symbol', 'date', 'adjusted','adjusted_rolling_mean_win_20','adjusted_rolling_mean_win_50','adjusted_rolling_mean_win_200']] \
    .melt(
        id_vars = ['symbol', 'date'],
        var_name = 'type',
        value_name = 'value'
    ) \
    .groupby('symbol') \
    .plot_timeseries(
        date_column = 'date',
        value_column = 'value',
        color_column = 'type',
        facet_ncol = 2,
        smooth = False,
        width = 1100,
        height = 800,
        title = 'Stock Prices with Rolling Means'
    )
        
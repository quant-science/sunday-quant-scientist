# The Quant Science Newsletter
# QS 005: Pytimetk Algorithmic Trading
# Copyright: Quant Science, LLC

# 1.0 IMPORT LIBRARIES ----

# Note: I'm using the development version of pytimetk==0.1.0.9000
# pip install git+https://github.com/business-science/pytimetk.git

import pandas as pd
import pytimetk as tk

# 2.0 GET STOCK PRICE DATA ----

stocks_df = tk.load_dataset("stocks_daily")
stocks_df['date'] = pd.to_datetime(df['date'])

stocks_df.glimpse()

# 3.0 ADD MOVING AVERAGES ----

# Add 2 moving averages (10-day, 50-Day, 200-Day)
windows = [10, 50, 200]
window_funcs = ['mean', 'median']

sma_df = stocks_df[['symbol', 'date', 'adjusted']] \
    .groupby('symbol') \
    .augment_rolling(
        date_column = 'date',
        value_column = 'adjusted',
        window = windows,
        window_func = window_funcs,
        center = False,
    )

sma_df.glimpse()

# 4.0 VISUALIZE ----

# Mean ----
(sma_df 

    # zoom in on dates
    .query('date >= "2023-01-01"') 

    # Convert to long format
    .melt(
        id_vars = ['symbol', 'date'],
        value_vars = [
            "adjusted", 
            "adjusted_rolling_mean_win_10", "adjusted_rolling_mean_win_50",
            "adjusted_rolling_mean_win_200",
        ]
    ) 

    # Group on symbol and visualize
    .groupby("symbol") 
    .plot_timeseries(
        date_column = 'date',
        value_column = 'value',
        color_column = 'variable',
        title = "Mean: 10, 50, 200 Day Moving Averages",
        smooth = False, 
        facet_ncol = 2,
        width = 800,
        height = 500,
        engine = "plotly"
    )
)


(sma_df 

    # zoom in on dates
    .query('date >= "2023-01-01"') 

    # Convert to long format
    .melt(
        id_vars = ['symbol', 'date'],
        value_vars = [
            "adjusted", 
            "adjusted_rolling_median_win_10", 
            "adjusted_rolling_median_win_50",
            "adjusted_rolling_median_win_200",
        ]
    ) 

    # Group on symbol and visualize
    .groupby("symbol") 
    .plot_timeseries(
        date_column = 'date',
        value_column = 'value',
        color_column = 'variable',
        title = "Median: 10, 50, 200 Day Moving Medians",
        smooth = False, 
        facet_ncol = 2,
        width = 800,
        height = 500,
        engine = "plotly"
    )
)

# 5.0 BONUS: BOLINGER BANDS ----

bollinger_df = stocks_df[['symbol', 'date', 'adjusted']] \
    .groupby('symbol') \
    .augment_rolling(
        date_column = 'date',
        value_column = 'adjusted',
        window = 20,
        window_func = ['mean', 'std'],
        center = False
    ) \
    .assign(
        upper_band = lambda x: x['adjusted_rolling_mean_win_20'] + 2*x['adjusted_rolling_std_win_20'],
        lower_band = lambda x: x['adjusted_rolling_mean_win_20'] - 2*x['adjusted_rolling_std_win_20']
    )

# Visualize
(bollinger_df

    # zoom in on dates
    .query('date >= "2023-01-01"') 

    # Convert to long format
    .melt(
        id_vars = ['symbol', 'date'],
        value_vars = ["adjusted", "adjusted_rolling_mean_win_20", "upper_band", "lower_band"]
    ) 

    # Group on symbol and visualize
    .groupby("symbol") 
    .plot_timeseries(
        date_column = 'date',
        value_column = 'value',
        color_column = 'variable',
        title = "Bolinger Bands: 20-Day Moving Average",
        # Adjust colors for Bollinger Bands
        color_palette =["#2C3E50", "#E31A1C", '#18BC9C', '#18BC9C'],
        smooth = False, 
        facet_ncol = 2,
        width = 800,
        height = 500,
        engine = "plotly" 
    )
)

# CONCLUSIONS ----
# - pytimetk is a great tool for adding features like moving averages to stock price data
# - pytimetk is also great for visualizing stock price data
# - we are just scratching the surface of what pytimetk can do

# WANT TO SEE WHAT PYTIMETK CAN DO FOR ALGO TRADING AND FINANCE?: ----
# - FREE TRAINING ON TUESDAY, OCTOBER 17TH, 2023: https://us02web.zoom.us/webinar/register/1716838099992/WN_QKYacsmkSryYuYvyUXkW9g


# WANT MORE ALGORITHMIC TRADING HELP?
# Register for our Course Waitlist: 
# https://learn.quantscience.io/python-algorithmic-trading-course-waitlist


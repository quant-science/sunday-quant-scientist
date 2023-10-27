# The Quant Science Newsletter
# QS 006: Pytimetk Algorithmic Trading
# Copyright: Quant Science, LLC

# 1.0 IMPORT LIBRARIES ----

# Note: I'm using the development version of pytimetk==0.1.0.9000
# pip install git+https://github.com/business-science/pytimetk.git

import pandas as pd
import pytimetk as tk

# 2.0 GET STOCK PRICE DATA ----

stocks_df = tk.load_dataset("stocks_daily")
stocks_df['date'] = pd.to_datetime(stocks_df['date'])

stocks_df.glimpse()


stocks_filtered_df = stocks_df.query('date >= "2021-01-01"')

# 3.0 ANOMALY DETECTION: IDENTIFY BUY/SELL REGIONS ----

stocks_anomalized_df = stocks_filtered_df[['symbol', 'date', 'adjusted']] \
    .groupby('symbol') \
    .anomalize(
        date_column = "date",
        value_column = "adjusted",
        method = 'stl',
        iqr_alpha = 0.10
        
    )
    
stocks_anomalized_df.glimpse()

# 4.0 VISUALIZE ----

stocks_anomalized_df \
    .groupby('symbol') \
    .plot_anomalies(
        date_column = "date",
        facet_ncol = 2,
        width = 1100,
        height = 800,
    )

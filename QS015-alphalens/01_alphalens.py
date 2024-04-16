import alphalens as al
import pandas as pd

# Example factor data
dates = pd.date_range('2023-01-01', periods=5, tz='UTC')
assets = ['AAPL', 'GOOGL', 'MSFT']
data = {
    ('2023-01-01', 'AAPL'): 1.5,
    ('2023-01-01', 'GOOGL'): 1.1,
    ('2023-01-01', 'MSFT'): 1.3,
    ('2023-01-02', 'AAPL'): 1.6,
    ('2023-01-02', 'GOOGL'): 1.2,
    ('2023-01-02', 'MSFT'): 1.1,
    ('2023-01-03', 'AAPL'): 1.5,
    ('2023-01-03', 'GOOGL'): 1.1,
    ('2023-01-03', 'MSFT'): 1.3,
    ('2023-01-04', 'AAPL'): 1.2,
    ('2023-01-04', 'GOOGL'): 1.1,
    ('2023-01-04', 'MSFT'): 1.5,
    ('2023-01-05', 'AAPL'): 1.2,
    ('2023-01-05', 'GOOGL'): 1.1,
    ('2023-01-05', 'MSFT'): 1.5,
}
factor_index = pd.MultiIndex.from_tuples(data.keys(), names=['date', 'asset'])
factor_data = pd.DataFrame(list(data.values()), index=factor_index, columns=['factor'])
factor_data.index = pd.MultiIndex.from_tuples([(pd.to_datetime(date), asset) for date, asset in factor_data.index], names=['date', 'asset'])
factor_data.index.set_levels(factor_data.index.levels[0].tz_localize('UTC'), level=0, inplace=True)

# Check for duplicates in the factor_data index
if factor_data.index.duplicated().any():
    print("Duplicate entries found in factor_data index. Attempting to drop duplicates...")
    factor_data = factor_data[~factor_data.index.duplicated(keep='first')]

# Example price data
price_data = pd.DataFrame({
    'AAPL': [150, 152, 154, 155, 157],
    'GOOGL': [1200, 1205, 1210, 1215, 1220],
    'MSFT': [220, 222, 224, 226, 228]
}, index=dates)

# Ensure price data has unique index if extending is necessary
if price_data.index.duplicated().any():
    print("Duplicate entries found in price_data index. Data may not have been extended properly.")

# Extend price data to ensure coverage for forward returns calculation
last_factor_date = factor_data.index.get_level_values('date').max()
additional_dates = pd.date_range(start=last_factor_date + pd.Timedelta(days=1), periods=5, tz='UTC')
extended_prices = pd.DataFrame(index=additional_dates, columns=price_data.columns, data=[[157, 1225, 230]]*5)
price_data = pd.concat([price_data, extended_prices]).drop_duplicates()


factor_data = al.utils.get_clean_factor_and_forward_returns(
    factor=factor_data['factor'],
    prices=price_data,
    periods=(1, 5),
    max_loss=0.8
)


factor_data

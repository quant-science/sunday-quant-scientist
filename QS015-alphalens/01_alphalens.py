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
}
factor_index = pd.MultiIndex.from_tuples(data.keys(), names=['date', 'asset'])
factor_data = pd.DataFrame(list(data.values()), index=factor_index, columns=['factor'])
factor_data.index = pd.MultiIndex.from_tuples([(pd.to_datetime(date), asset) for date, asset in factor_data.index], names=['date', 'asset'])
factor_data.index.set_levels(factor_data.index.levels[0].tz_localize('UTC'), level=0, inplace=True)

factor_data

# Example price data
price_data = pd.DataFrame({
    'AAPL': [150, 152, 154, 155, 157],
    'GOOGL': [1200, 1205, 1210, 1215, 1220],
    'MSFT': [220, 222, 224, 226, 228]
}, index=dates)

price_data

# Convert the factor data to work with Alphalens
factor_data = al.utils.get_clean_factor_and_forward_returns(
    factor=factor_data['factor'],
    prices=price_data,
    periods=(1, 5)
)



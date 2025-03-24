import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# This function will create a simple trading strategy based on moving averages
def trading_strategy(ticker='AAPL', start='2020-01-01',):
    df = yf.download(ticker, start=start)
    
    # Drop the 'Ticker' level from columns
    df.columns = df.columns.droplevel(1)  # Remove 'AAPL' level
    close_col = 'Close'  # Now we can use simple column name
    
    df['SMA20'] = df[close_col].rolling(window=20).mean()
    df['SMA50'] = df[close_col].rolling(window=50).mean()
    
    # Detect crossovers
    df['Signal'] = 0
    df['Prev_SMA20'] = df['SMA20'].shift(1)
    df['Prev_SMA50'] = df['SMA50'].shift(1)
    
    # Buy signal: SMA20 crosses above SMA50
    df.loc[(df['SMA20'] > df['SMA50']) & (df['Prev_SMA20'] <= df['Prev_SMA50']), 'Signal'] = 1
    
    # Sell signal: SMA20 crosses below SMA50
    df.loc[(df['SMA20'] < df['SMA50']) & (df['Prev_SMA20'] >= df['Prev_SMA50']), 'Signal'] = -1
    
    return df

result = trading_strategy()
result


# Plotting the trading strategy
def plot_trading_strategy(df, ticker='AAPL'):
    plt.figure(figsize=(12, 6))
    
    # Plot price
    plt.plot(df.index, df['Close'], label='Close Price', alpha=0.5)
    
    # Plot SMAs
    plt.plot(df.index, df['SMA20'], label='SMA20', alpha=0.8)
    plt.plot(df.index, df['SMA50'], label='SMA50', alpha=0.8)
    
    # Plot buy signals
    plt.scatter(df.index[df['Signal'] == 1], 
                df['Close'][df['Signal'] == 1], 
                color='green', 
                label='Buy', 
                marker='^', 
                s=100)
    
    # Plot sell signals
    plt.scatter(df.index[df['Signal'] == -1], 
                df['Close'][df['Signal'] == -1], 
                color='red', 
                label='Sell', 
                marker='v', 
                s=100)
    
    plt.title(f'{ticker} Trading Strategy - SMA Crossover')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_trading_strategy(result)
import yfinance as yf
import mplfinance as mpf
import warnings
warnings.filterwarnings('ignore')

data = yf.download("AAPL", start="2022-01-01", end="2022-06-30")

mpf.plot(data)

mpf.plot(data, type="candle")

mpf.plot(data, type="line")

mpf.plot(data, type="renko")

mpf.plot(data, type="ohlc", mav=15)

mpf.plot(data, type="candle", mav=(7, 14, 21))

mpf.plot(data, type="candle", mav=(7, 14, 21), volume=True)

mpf.plot(
    data, 
    type="candle", 
    mav=(7, 14, 21), 
    volume=True, 
    show_nontrading=True
)

intraday = yf.download(tickers="PLTR", period="5d", interval="1m")
iday = intraday.iloc[-100:, :]
mpf.plot(iday, type="candle", mav=(7, 12), volume=True)



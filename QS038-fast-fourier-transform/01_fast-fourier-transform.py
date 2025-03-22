# The Quant Science Newsletter
# QS 038: Using the Fast Fourier Transformation

# READY TO LEARN ALGORITHMIC TRADING WITH US?
# Register for our Course Waitlist: 
# https://learn.quantscience.io/python-algorithmic-trading-course-waitlist

import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Download the price data from Yahoo
prices = yf.download("GS", start="2020-01-01").Close

# Fit the transformation and add the absolute value and angle
close_fft = np.fft.fft(prices.GS.values)
fft_df = pd.DataFrame({'fft': close_fft})
fft_df['absolute'] = fft_df['fft'].apply(lambda x: np.abs(x))
fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))

# +
# Plot across a series of transformations
plt.figure(figsize=(14, 6))
fft_list = np.asarray(fft_df['fft'].values)

for num_ in [3, 6, 9, 100]:
    fft_list_m10 = np.copy(fft_list)
    fft_list_m10[num_:-num_] = 0
    plt.plot(np.fft.ifft(fft_list_m10), label=f'Fourier transform with {num_} components')

plt.plot(prices.GS.values, label='Real')
plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Goldman Sachs (close) stock prices & Fourier transforms')
plt.legend()
plt.show()

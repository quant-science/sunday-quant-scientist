# The Quant Science Newsletter
# QS 023: K-Means Clustering for Algorithmic Trading

# READY TO LEARN ALGORITHMIC TRADING WITH US?
# Register for our Course Waitlist: 
# https://learn.quantscience.io/python-algorithmic-trading-course-waitlist

from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import yfinance as yf

# Load Dow Jones data
dji = pd.read_html('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average')[2]
symbols = dji.Symbol.tolist()
data = yf.download(symbols, start="2020-01-01", end="2024-12-31")["Adj Close"]

# Calculate returns and volatility (annualized)
moments = (
    data
    .pct_change()
    .describe()
    .T[["mean", "std"]]
    .rename(columns={"mean": "returns", "std": "vol"})
) * [252, sqrt(252)]

# Plot Elbow Curve to find optimal number of clusters
sse = []
for k in range(2, 15):
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(moments)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(2, 15), sse, marker='o', linestyle='-', color='b')
plt.title("Elbow Curve for KMeans Clustering")
plt.xlabel("Number of Clusters")
plt.ylabel("Sum of Squared Errors (SSE)")
plt.grid(True)
plt.show()

# Perform KMeans clustering with k=5
kmeans = KMeans(n_clusters=5, n_init=10).fit(moments)
labels = kmeans.labels_

# Scatter plot of Dow Jones stocks by returns and volatility
plt.figure(figsize=(10, 7))
scatter = plt.scatter(
    moments.returns, 
    moments.vol, 
    c=labels, 
    cmap="rainbow",
    edgecolor='k',
    s=100
)
plt.title("Dow Jones Stocks by Return and Volatility (K=5)")
plt.xlabel("Annualized Returns")
plt.ylabel("Annualized Volatility")

# Annotate each point with the stock symbol
for i, symbol in enumerate(moments.index):
    plt.annotate(
        symbol, 
        (moments.returns[i], moments.vol[i]), 
        textcoords="offset points", 
        xytext=(5, 5),  # Offset text to avoid overlap with point
        ha='center',
        fontsize=8,
        weight='bold'
    )

# Add color bar to indicate clusters
plt.colorbar(scatter, label="Cluster Label")
plt.grid(True)
plt.show()

# Annotate points with stock symbols and cluster labels
for i, txt in enumerate(moments.index):
    plt.annotate(f"{txt} ({labels[i]})", 
                 (moments.returns[i], moments.vol[i] + 0.05),
                 fontsize=8,
                 ha="center")

# Add a color bar to show cluster groups
plt.colorbar(scatter, label="Cluster Label")
plt.grid(True)
plt.show()




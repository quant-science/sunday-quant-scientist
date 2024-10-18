from math import sqrt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import yfinance as yf



dji = (
    pd.read_html('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average')[2]
)

symbols = dji.Symbol.tolist()
data = yf.download(
    symbols, 
    start="2020-01-01",
    end="2022-12-31"
)["Adj Close"]



moments = (
    data
    .pct_change()
    .describe()
    .T[["mean", "std"]]
    .rename(columns={"mean": "returns", "std": "vol"})
) * [252, sqrt(252)]



sse = []
for k in range(2, 15):
    kmeans = KMeans(n_clusters=k, n_init=10)
    kmeans.fit(moments)
    sse.append(kmeans.inertia_)

plt.plot(range(2, 15), sse)
plt.title("Elbow Curve");



kmeans = KMeans(n_clusters=5, n_init=10).fit(moments)
plt.scatter(
    moments.returns, 
    moments.vol, 
    c=kmeans.labels_, 
    cmap="rainbow",
);


plt.title("Dow Jones stocks by return and volatility (K=5)")
for i in range(len(moments.index)):
    txt = f"{moments.index[i]} ({kmeans.labels_[i]})"
    xy = tuple(moments.iloc[i, :] + [0, 0.01])
    plt.annotate(txt, xy)



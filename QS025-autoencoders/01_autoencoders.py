# The Quant Science Newsletter
# QS 025: Autoencoders For Algorithmic Trading

# READY TO LEARN ALGORITHMIC TRADING WITH US?
# Register for our Course Waitlist: 
# https://learn.quantscience.io/python-algorithmic-trading-course-waitlist

import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


symbols = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "TSLA", "BRK-B", "V", "JNJ", "WMT", "JPM",
    "MA", "PG", "UNH", "DIS", "NVDA", "HD", 
    "PYPL", "BAC", "VZ", "ADBE", "CMCSA", "NFLX",
    "KO", "NKE", "MRK", "PEP", "T", "PFE", "INTC",
]

stock_data = yf.download(
    symbols, 
    start="2020-01-01", 
    end="2023-12-31"
)["Adj Close"]


log_returns = np.log(stock_data / stock_data.shift(1))
moving_avg = stock_data.rolling(window=22).mean()
volatility = stock_data.rolling(window=22).std()

features = pd.concat([log_returns, moving_avg, volatility], axis=1).dropna()
processed_data = (features - features.mean()) / features.std()


tensor = torch.tensor(processed_data.values, dtype=torch.float32)
dataset = TensorDataset(tensor)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

class StockAutoencoder(nn.Module):
    """Autoencoder neural network for stock data embedding
    
    This class defines an autoencoder with an encoder and decoder
    to compress and reconstruct stock data.
    
    Parameters
    ----------
    feature_dim : int
        The dimensionality of the input features
    """
    
    def __init__(self, feature_dim):
        super(StockAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10),  # Latent space
        )
        self.decoder = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train(model, data_loader, epochs=100):
    """Train the autoencoder model
    
    This function trains the autoencoder using MSE loss and Adam optimizer
    over a specified number of epochs.
    
    Parameters
    ----------
    model : nn.Module
        The autoencoder model to be trained
    data_loader : DataLoader
        DataLoader object to iterate through the dataset
    epochs : int, optional
        Number of epochs to train the model (default is 100)
    """
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for epoch in range(epochs):
        for data in data_loader:
            inputs = data[0]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

feature_dim = processed_data.shape[1]
model = StockAutoencoder(feature_dim)
train(model, data_loader)

def extract_embeddings(model, data_loader):
    """Extract embeddings from the trained autoencoder model
    
    This function extracts embeddings by passing data through the encoder
    part of the autoencoder.
    
    Parameters
    ----------
    model : nn.Module
        The trained autoencoder model
    data_loader : DataLoader
        DataLoader object to iterate through the dataset
    
    Returns
    -------
    embeddings : torch.Tensor
        Tensor containing the extracted embeddings
    """
    
    model.eval()
    embeddings = []
    with torch.no_grad():
        for data in data_loader:
            inputs = data[0]
            encoded = model.encoder(inputs)
            embeddings.append(encoded)
    return torch.vstack(embeddings)


embeddings = extract_embeddings(model, data_loader)

kmeans = KMeans(n_clusters=5, random_state=42).fit(embeddings.numpy())
clusters = kmeans.labels_

pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings.numpy())

plt.figure(figsize=(12, 8))
scatter = sns.scatterplot(
    x=embeddings_2d[:, 0],
    y=embeddings_2d[:, 1],
    hue=clusters,
    palette=sns.color_palette("hsv", len(set(clusters))),
    s=100,  # Increase marker size
    edgecolor='k'
)
plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimension 2")
plt.title("PCA Plot of Stock Embeddings with Clusters")
plt.legend(title="Cluster")
plt.grid(True)

# Adding stock symbols with enhanced readability
for i, symbol in enumerate(symbols):
    plt.annotate(
        symbol, 
        (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
        textcoords="offset points", 
        xytext=(8, 8),  # Increase offset to reduce overlap
        ha='center',
        fontsize=9,
        fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7, edgecolor="gray")
    )

plt.show()



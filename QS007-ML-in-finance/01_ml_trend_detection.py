# The Quant Science Newsletter
# QS 007: Machine Learning for SPY Trend Detection
# Credit: Danny Groves for his awesome
# https://twitter.com/drdanobi/status/1729469353282744515?s=46&t=npiSgI5uPxafM5JqdAQNDw

# Libraries
from openbb_terminal.sdk import openbb
import pandas as pd
import pytimetk as tk
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score

# INPUTS

SYMBOL = "SPY"
START = "2015-09-30"
END = "2023-09-30"
TRAIN_UNTIL = "2018-09-30"

# Load the data
data = openbb.stocks.load(SYMBOL, start_date=START, end_date=END)

df = data.reset_index()[['date', 'Open', 'High', 'Low', 'Close']]
df

df \
    .plot_timeseries(
        date_column="date",
        value_column="Close",
        title="SPY Close",
        x_lab="Date",
        y_lab="Close",
    )

# Feature Engineering

# Distance from Moving Averages
for m in [10, 20, 30, 50, 100]:
    df[f'feat_dist_from_ma_{m}'] = df['Close']/df['Close'].rolling(m).mean() - 1

# Distance from n day max/min
for m in [6, 10, 15, 20, 30, 50, 100]:
    df[f'feat_dist_from_max_{m}'] = df['Close']/df['High'].rolling(m).max() - 1
    df[f'feat_dist_from_min_{m}'] = df['Close']/df['Low'].rolling(m).min() - 1 

# Price Distance
for m in [6, 10, 15, 20, 30, 50, 100]:
    df[f'feat_price_dist_{m}'] = df['Close']/df['Close'].shift(m) - 1

df.glimpse()

# Target Variable (Predict price above 20SMA in 5 days)

df['target_ma'] = df['Close'].rolling(20).mean()
df['price_above_ma'] = df['Close'] > df['target_ma']
df['target'] = df['price_above_ma'].astype(int).shift(-5)

df.glimpse()

# Clean and Train Test Split

df = df.dropna()

feat_cols = [col for col in df.columns if 'feat' in col]
train_until = TRAIN_UNTIL

x_train = df[df['date'] <= train_until][feat_cols]
y_train = df[df['date'] <= train_until]['target']

x_test = df[df['date'] > train_until][feat_cols]
y_test = df[df['date'] > train_until]['target']

# Train Model

clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    random_state=42,
    class_weight='balanced'
)

clf.fit(x_train, y_train)

y_train_pred = clf.predict(x_train)
y_test_pred = clf.predict(x_test)

print(f"Train Accuracy: {accuracy_score(y_train, y_train_pred)}")
print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred)}")

print(f"Train Precision: {precision_score(y_train, y_train_pred)}")
print(f"Test Precision: {precision_score(y_test, y_test_pred)}")

print(f"Train ROC AUC: {roc_auc_score(y_train, clf.predict_proba(x_train)[:, 1])}")
print(f"Test ROC AUC: {roc_auc_score(y_test, clf.predict_proba(x_test)[:, 1])}")

# Visualize

df_test = df[df['date'] > train_until].reset_index(drop=True)
df_test['pred_prob'] = clf.predict_proba(x_test)[:, 1]
df_test['pred'] = df_test['pred_prob'] > 0.5

fig = df_test \
    .plot_timeseries(
        date_column="date",
        value_column="Close",
        title=f"{SYMBOL} Price with Predicted Patterns",
        x_lab="Date",
        y_lab="Close",
    )

fig.add_trace(
    go.Line(
        x=df_test['date'],
        y=df_test['target_ma'],
        name="Target 20SMA"
    )
)

df_pattern = (
    df_test[df_test['pred']]
        .groupby((~df_test['pred']).cumsum())['date']
        .agg(['first', 'last'])
)

for idx, row in df_pattern.iterrows():
    fig.add_vrect(
        x0=row['first'],
        x1=row['last'],
        line_width=0,
        fillcolor="green",
        opacity=0.2,
    )
    
fig.update_layout(
    width = 800,
    height = 600,
    xaxis_rangeslider_visible=True,
)

fig.show()

# WANT MORE ALGORITHMIC TRADING HELP?
# Register for our Course Waitlist: 
# https://learn.quantscience.io/python-algorithmic-trading-course-waitlist


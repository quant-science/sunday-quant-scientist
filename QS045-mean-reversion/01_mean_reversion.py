# The Quant Science Newsletter
# QS 045: How to backtest a mean reversion strategy

# READY TO LEARN ALGORITHMIC TRADING WITH US?
# Join our free webinar where we will give you a sneak peek into the strategies and Python code that power our hedge fund:
# https://learn.quantscience.io/qs-register


import pandas as pd
import warnings

from zipline import run_algorithm
from zipline.pipeline.factors import Returns, VWAP
from zipline.pipeline import CustomFactor, Pipeline
from zipline.api import (
    calendars,
    attach_pipeline,
    schedule_function,
    date_rules,
    time_rules,
    pipeline_output,
    record,
    order_target_percent,
    get_datetime
)

warnings.filterwarnings("ignore")

# ## Define our custom factor

# We create a custom factor to calculate mean reversion scores for stocks.

class MeanReversion(CustomFactor):
    inputs = [Returns(window_length=21)]
    window_length = 21

    def compute(self, today, assets, out, monthly_returns):
        df = pd.DataFrame(monthly_returns)
        out[:] = df.iloc[-1].sub(df.mean()).div(df.std())

# This code defines a custom factor called MeanReversion. It uses a 21-day window of stock returns to calculate a mean reversion score. The score is based on how far the most recent return is from the average, relative to the standard deviation. This helps identify stocks that may be overbought or oversold.

# ## Set up our trading pipeline

# We create a pipeline to select stocks based on our mean reversion factor.

def compute_factors():
    mean_reversion = MeanReversion()
    vwap = VWAP(window_length=21)
    pipe = Pipeline(
        columns={
            "longs": mean_reversion.bottom(5),
            "shorts": mean_reversion.top(5),
            "ranking": mean_reversion.rank(),
        },
        screen=vwap > 15.0
    )
    return pipe


pipe = compute_factors()
pipe.show_graph()

# Our pipeline selects stocks for long and short positions based on their mean reversion scores. We choose the bottom 5 stocks for long positions and the top 5 for short positions. We also include a ranking of all stocks. The pipeline screens out low-priced stocks by only considering those with a 21-day volume-weighted average price above $15.

# ## Implement our trading algorithm

# We define functions to handle trading logic and portfolio rebalancing.

def before_trading_start(context, data):
    context.factor_data = pipeline_output("factor_pipeline")
    record(factor_data=context.factor_data.ranking)

    assets = context.factor_data.index
    record(prices=data.current(assets, "price"))


def rebalance(context, data):
    factor_data = context.factor_data
    assets = factor_data.index

    longs = assets[factor_data.longs]
    shorts = assets[factor_data.shorts]
    divest = context.portfolio.positions.keys() - longs.union(shorts)

    print(
        f"{get_datetime().date()} | Longs {len(longs)} | Shorts {len(shorts)} | {context.portfolio.portfolio_value}"
    )

    exec_trades(data, assets=divest, target_percent=0)

    exec_trades(
        data, assets=longs, target_percent=1 / len(longs) if len(longs) > 0 else 0
    )

def exec_trades(data, assets, target_percent):
    for asset in assets:
        if data.can_trade(asset):
            order_target_percent(asset, target_percent)


def initialize(context):
    attach_pipeline(compute_factors(), "factor_pipeline")
    schedule_function(
        rebalance,
        date_rules.month_end(),
        time_rules.market_open(),
        calendar=calendars.US_EQUITIES,
    )

# These functions form the core of our trading algorithm. We gather factor data before trading starts, rebalance our portfolio at the end of each month, and execute trades to achieve our target allocations. The algorithm aims to go long on stocks with low mean reversion scores and short those with high scores.

# ## Run our backtest

# We set up and run a backtest of our trading strategy.

start = pd.Timestamp("2020-01-01")
end = pd.Timestamp("2024-07-01")
capital_base = 25_000

perf = run_algorithm(
    start=start,
    end=end,
    initialize=initialize,
    capital_base=capital_base,
    before_trading_start=before_trading_start,
    bundle="quotemedia",
)

perf.portfolio_value.plot()

# We run a backtest of our strategy from January 2020 to July 2024 with an initial capital of $25,000. The algorithm uses the Quotemedia data bundle for historical stock prices. After running the backtest, we plot the portfolio value over time to visualize the strategy's performance.

# The Quant Science Newsletter
# QS 037: A new library that uses reinforcement learning for trading

# READY TO LEARN ALGORITHMIC TRADING WITH US?
# Register for our Course Waitlist: 
# https://learn.quantscience.io/python-algorithmic-trading-course-waitlist

import yfinance
import pandas_ta  #noqa


TICKER = 'NVDA'  # TODO: replace this with your own ticker
TRAIN_START_DATE = '2021-01-01'  # TODO: replace this with your own start date
TRAIN_END_DATE = '2022-12-31'  # TODO: replace this with your own end date
EVAL_START_DATE = '2023-01-01'  # TODO: replace this with your own end date
EVAL_END_DATE = '2023-01-31'  # TODO: replace this with your own end date

yf_ticker = yfinance.Ticker(ticker=TICKER)

df_training = yf_ticker.history(start=TRAIN_START_DATE, end=TRAIN_END_DATE)
df_training.drop(['Dividends', 'Stock Splits'], axis=1, inplace=True)
df_training["Volume"] = df_training["Volume"].astype(int)
df_training.ta.log_return(append=True, length=16)
df_training.ta.rsi(append=True, length=14)
df_training.ta.macd(append=True, fast=12, slow=26)
df_training.to_csv('training.csv', index=False)

df_evaluation = yf_ticker.history(start=EVAL_START_DATE, end=EVAL_END_DATE)
df_evaluation.drop(['Dividends', 'Stock Splits'], axis=1, inplace=True)
df_evaluation["Volume"] = df_evaluation["Volume"].astype(int)
df_evaluation.ta.log_return(append=True, length=16)
df_evaluation.ta.rsi(append=True, length=14)
df_evaluation.ta.macd(append=True, fast=12, slow=26)
df_evaluation.to_csv('evaluation.csv', index=False)


import pandas as pd
from tensortrade.feed.core import DataFeed, Stream
from tensortrade.oms.instruments import Instrument
from tensortrade.oms.exchanges import Exchange, ExchangeOptions
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Wallet, Portfolio
import tensortrade.env.default as default

def create_env(config):
    dataset = pd.read_csv(filepath_or_buffer=config["csv_filename"], parse_dates=['Datetime']).fillna(method='backfill').fillna(method='ffill')
    ttse_commission = 0.0035  # TODO: adjust according to your commission percentage, if present
    price = Stream.source(list(dataset["Close"]), dtype="float").rename("USD-TTRD")
    ttse_options = ExchangeOptions(commission=ttse_commission)
    ttse_exchange = Exchange("TTSE", service=execute_order, options=ttse_options)(price)

 # Instruments, Wallets and Portfolio
    USD = Instrument("USD", 2, "US Dollar")
    TTRD = Instrument("TTRD", 2, "TensorTrade Corp")
    cash = Wallet(ttse_exchange, 1000 * USD)  # This is the starting cash we are going to use
    asset = Wallet(ttse_exchange, 0 * TTRD)  # And we will start owning 0 stocks of TTRD
    portfolio = Portfolio(USD, [cash, asset])

    # Renderer feed
    renderer_feed = DataFeed([
        Stream.source(list(dataset["Datetime"])).rename("date"),
        Stream.source(list(dataset["Open"]), dtype="float").rename("open"),
        Stream.source(list(dataset["High"]), dtype="float").rename("high"),
        Stream.source(list(dataset["Low"]), dtype="float").rename("low"),
        Stream.source(list(dataset["Close"]), dtype="float").rename("close"),
        Stream.source(list(dataset["Volume"]), dtype="float").rename("volume")
    ])

    features = []
    for c in dataset.columns[1:]:
        s = Stream.source(list(dataset[c]), dtype="float").rename(dataset[c].name)
        features += [s]
    feed = DataFeed(features)
    feed.compile()

    reward_scheme = default.rewards.SimpleProfit(window_size=config["reward_window_size"])
    action_scheme = default.actions.BSH(cash=cash, asset=asset)
    
    env = default.create(
            feed=feed,
            portfolio=portfolio,
            action_scheme=action_scheme,
            reward_scheme=reward_scheme,
            renderer_feed=renderer_feed,
            renderer=[],
            window_size=config["window_size"],
            max_allowed_loss=config["max_allowed_loss"]
        )
    
    return env



import ray
import os
from ray import tune
from ray.tune.registry import register_env

# Let's define some tuning parameters
FC_SIZE = tune.grid_search([[256, 256], [1024], [128, 64, 32]])  # Those are the alternatives that ray.tune will try...
LEARNING_RATE = tune.grid_search([0.001, 0.0005, 0.00001])  # ... and they will be combined with these ones ...
MINIBATCH_SIZE = tune.grid_search([5, 10, 20])  # ... and these ones, in a cartesian product.

# Get the current working directory
cwd = os.getcwd()

# Initialize Ray
ray.init()  # There are *LOTS* of initialization parameters, like specifying the maximum number of CPUs\GPUs to allocate. For now just leave it alone.

# Register our environment, specifying which is the environment creation function
register_env("MyTrainingEnv", create_env)

# Specific configuration keys that will be used during training
env_config_training = {
    "window_size": 14,  # We want to look at the last 14 samples (hours)
    "reward_window_size": 7,  # And calculate reward based on the actions taken in the next 7 hours
    "max_allowed_loss": 0.10,  # If it goes past 10% loss during the iteration, we don't want to waste time on a "loser".
    "csv_filename": os.path.join(cwd, 'training.csv'),  # The variable that will be used to differentiate training and validation datasets
}
# Specific configuration keys that will be used during evaluation (only the overridden ones)
env_config_evaluation = {
    "max_allowed_loss": 1.00,  # During validation runs we want to see how bad it would go. Even up to 100% loss.
    "csv_filename": os.path.join(cwd, 'evaluation.csv'),  # The variable that will be used to differentiate training and validation datasets
}

analysis = tune.run(
    run_or_experiment="PPO",  # We'll be using the builtin PPO agent in RLLib
    name="MyExperiment1",
    metric='episode_reward_mean',
    mode='max',
    stop={
        "training_iteration": 5  # Let's do 5 steps for each hyperparameter combination
    },
    config={
        "env": "MyTrainingEnv",
        "env_config": env_config_training,  # The dictionary we built before
        "log_level": "WARNING",
        "framework": "torch",
        "ignore_worker_failures": True,
        "num_workers": 1,  # One worker per agent. You can increase this but it will run fewer parallel trainings.
        "num_envs_per_worker": 1,
        "num_gpus": 0,  # I yet have to understand if using a GPU is worth it, for our purposes, but I think it's not. This way you can train on a non-gpu enabled system.
        "clip_rewards": True,
        "lr": LEARNING_RATE,  # Hyperparameter grid search defined above
        "gamma": 0.50,  # This can have a big impact on the result and needs to be properly tuned (range is 0 to 1)
        "observation_filter": "MeanStdFilter",
        "model": {
            "fcnet_hiddens": FC_SIZE,  # Hyperparameter grid search defined above
        },
        "sgd_minibatch_size": MINIBATCH_SIZE,  # Hyperparameter grid search defined above
        "evaluation_interval": 1,  # Run evaluation on every iteration
        "evaluation_config": {
            "env_config": env_config_evaluation,  # The dictionary we built before (only the overriding keys to use in evaluation)
            "explore": False,  # We don't want to explore during evaluation. All actions have to be repeatable.
        },
    },
    num_samples=1,  # Have one sample for each hyperparameter combination. You can have more to average out randomness.
    keep_checkpoints_num=10,  # Keep the last 2 checkpoints
    checkpoint_freq=1,  # Do a checkpoint on each iteration (slower but you can pick more finely the checkpoint to use later)
)

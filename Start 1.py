import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from Functions import *
from RLEnvironment import TradingEnv

raw_data = yf.download("AAPL", start="2020-01-01", end="2021-01-01")
raw_data.columns = raw_data.columns.droplevel('Ticker')
df = modelInput(raw_data)

features = [
    "Close", "SMA_10", "SMA_30", "SMA_50", "EMA_30",
    "True_Range", "ATR_SMA", "ATR_EMA",
    "RSI", "Stochastic_Oscillator%K", "Stochastic_Oscillator%D", "OBV"
]

# Create the environment with debug mode on
env = TradingEnv(df, feature_cols=features, verbose=True)

# Reset the environment
obs, _ = env.reset()
print("Initial Observation:", obs)

for episode in range(100):
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = env.action_space.sample()
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward

    print(f"Episode {episode + 1} completed — Total reward: {total_reward:.4f}")

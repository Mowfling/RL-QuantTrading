import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from Functions import *
from RLEnvironment import TradingEnv
import torch
import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise

print(torch.cuda.is_available())  # should be True
print(torch.cuda.get_device_name(0))  # should show your GPU

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
raw_data = yf.download("AAPL", start="2020-01-01", end="2021-01-01")
raw_data.columns = raw_data.columns.droplevel('Ticker')
df = modelInput(raw_data)

features = [
    "Close", "SMA_10", "SMA_30", "SMA_50", "EMA_30",
    "True_Range", "ATR_SMA", "ATR_EMA",
    "RSI", "Stochastic_Oscillator%K", "Stochastic_Oscillator%D", "OBV"
]

# Create the environment with debug mode on
env = TradingEnv(df, feature_cols=features, verbose=False)

n_actions = env.action_space.shape[0]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# Create the model
model = TD3(
    "MlpPolicy",
    env,
    action_noise=action_noise,
    verbose=1,
    tensorboard_log="./td3_logs/",
)

# Train the model
model.learn(total_timesteps=100_000)

# Save the model
model.save("td3_trading_model")

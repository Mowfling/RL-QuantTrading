import numpy as np
import yfinance as yf
from Functions import *
from RLEnvironment import TradingEnv
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from data import features

raw_data = yf.download("SPY", start="2020-01-01", end="2021-01-01")
raw_data.columns = raw_data.columns.droplevel('Ticker')
df = modelInput(raw_data)

model = TD3.load("td3_trading_model")
env = TradingEnv(df, feature_cols=features, verbose=True)
obs, _ = env.reset()


done = False
total_reward = 0
portfolio_values = []


while not done:
    action, _ = model.predict(obs, deterministic=True)  # Use deterministic=True for evaluation
    obs, reward, done, _, _ = env.step(action)
    total_reward += reward

print(f"Total test reward: {total_reward:.4f}")

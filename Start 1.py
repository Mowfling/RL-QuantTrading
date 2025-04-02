import numpy as np
import yfinance as yf
from Functions import *
from RLEnvironment import TradingEnv
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from data import features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
raw_data = yf.download("AAPL", start="2020-01-01", end="2021-01-01")
raw_data.columns = raw_data.columns.droplevel('Ticker')
df = modelInput(raw_data)


env = TradingEnv(df, feature_cols=features, verbose=False)

n_actions = env.action_space.shape[0]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = TD3(
    "MlpPolicy",
    env,
    action_noise=action_noise,
    verbose=1,
    tensorboard_log="./td3_logs/",
)

model.learn(total_timesteps=100_000)
model.save("td3_trading_model")

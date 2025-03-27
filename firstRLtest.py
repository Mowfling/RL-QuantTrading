import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):
    def __init__(self, df):
        self.df = df
        self.current_step = 0
        self.initial_cash = 10000
        self.cash = self.initial_cash
        self.shares_held = 0

        # Define observation space (e.g. 5 indicators)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(df.shape[1],), dtype=np.float32)

        # Define action space: Discrete (0 = hold, 1 = buy, 2 = sell)
        self.action_space = spaces.Discrete(3)

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.cash = self.initial_cash
        self.shares_held = 0
        return self._get_obs(), {}

    def step(self, action):
        price = self.df["Close"].iloc[self.current_step]
        reward = 0

        # Execute action
        if action == 1:  # Buy
            if self.cash > 0:
                self.shares_held = self.cash / price
                self.cash = 0
        elif action == 2:  # Sell
            if self.shares_held > 0:
                self.cash = self.shares_held * price
                self.shares_held = 0

        # Reward = portfolio change (optional: Sharpe, drawdown, etc.)
        portfolio_value = self.cash + self.shares_held * price
        reward = portfolio_value  # or delta from last step

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        return self.df.iloc[self.current_step].values.astype(np.float32)

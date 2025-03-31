import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from data import RED, GREEN

class TradingEnv(gym.Env):
    def __init__(self, df, feature_cols, verbose=False):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.current_step = 0
        self.balance = 1000.0
        self.position = 0
        self.entry_price = 0.0
        self.verbose = verbose

        # Observation space matches the number of features
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.feature_cols),),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # Hold, Buy, Sell

    def _get_obs(self):
        return self.df.loc[self.current_step, self.feature_cols].values.astype(np.float32)

    def reset(self, seed=None, options=None):
        self.current_step = 0
        self.balance = 1000.0
        self.position = 0
        self.entry_price = 0.0
        return self._get_obs(), {}


    def step(self, action):
        current_price = self.df.loc[self.current_step, "Close"]
        prev_portfolio_value = self.balance + self.position * current_price
        reward = 0.0
        action_name = ["Hold", "Buy", "Sell"][action]

        #Execute action
        if action == 1 and self.position == 0:  # Buy
            self.position = 1
            self.entry_price = current_price
            self.balance -= current_price  # spend cash to buy
        elif action == 2 and self.position == 1:  # Sell
            self.position = 0
            self.balance += current_price  # gain cash from sell

        #Advance time
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        next_price = self.df.loc[self.current_step, "Close"]
        new_portfolio_value = self.balance + self.position * next_price

        #Reward
        reward = new_portfolio_value - prev_portfolio_value
        reward /= prev_portfolio_value

        if self.verbose and self.current_step % 10 == 0:
            print(f"Step: {self.current_step}")
            print(f"  Action: {action_name}")
            print(f"  Price: {current_price:.2f} → {next_price:.2f}")
            print(f"  Position: {self.position}")
            print(f"  Balance: {self.balance:.2f}")
            print(f"  Portfolio Value: {prev_portfolio_value:.2f} → {new_portfolio_value:.2f}")
            print(f"  Reward: {reward:.4f}")
            print("─" * 40)


        next_state = self._get_obs()
        terminated = done
        truncated = False  # or True if you want to simulate a time limit
        return next_state, reward, terminated, truncated, {}

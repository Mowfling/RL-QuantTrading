import numpy as np
import yfinance as yf
from Functions import *
from RLEnvironment import TradingEnv
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from data import features
import matplotlib.pyplot as plt


plotData(getHistoricalData("AAPL", start="2020-01-01", end="2021-01-01")['Close'], title="AAPL from 2020 to 2021")
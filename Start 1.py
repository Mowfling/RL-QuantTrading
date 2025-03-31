import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from data import listofsp500
from Functions import *
from backtesting_functions import *

raw_data = yf.download("AAPL", start="2020-01-01", end="2021-01-01")
raw_data.head()
df = modelInput(raw_data)
features = [
    "Close", "SMA_10", "SMA_30", "SMA_50", "EMA_30",
    "True_Range", "ATR_SMA", "ATR_EMA",
    "RSI", "Stochastic_Oscillator", "OBV"
]
assert all(f in df.columns for f in features), "Missing one or more features in df!"

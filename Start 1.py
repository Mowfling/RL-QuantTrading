import pandas as pd
import numpy
import matplotlib
import yfinance as yf
from Functions import fetch_sp500_tickers, fetchStock


test = fetch_sp500_tickers()
fetchStock(test[0], False)





import pandas as pd
import numpy
import matplotlib
import yfinance as yf
from Functions import fetch_sp500_tickers, fetch_stock


test = fetch_sp500_tickers()
fetch_stock(test[0])





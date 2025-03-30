import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from data import listofsp500
from Functions import *
from backtesting_functions import *

#plotData(calculate50DaySMA(getHistoricalData("AAPL", period="20y")), title="Stock price of AAPL", ylabel="Price in USD")
plotData(rsi_sectorrotation(sector_period="30y", rsi_period=365), title="RSI Sector rotation", ylabel="RSI Factor", show_graph=False)
plotData(calculate_rsi(getHistoricalData("XOM")["Close"], period=14), title="RSI of XOM", ylabel="RSI factor", show_graph=False)
plt.show()
#plotData(getHistoricalData("XOM")["Close"], title="Closing price of XOM")>
#data = getHistoricalData("XOM", period="3y")["Close"]
#rsi = calculate_rsi(data)
#PlotDataDualAxis(data, rsi, title="XOM Price vs RSI", ax1_label="Closing price in USD", ax2_label="RSI")


#data = getHistoricalData("SPY", period="20y")["Close"]
#rsi = calculate_rsi(data)

#result = backtest_rsi_normalized(data, rsi, buy_threshold=30, sell_threshold=70)

# Plot the portfolio value over time
#plotData(result[["Portfolio Value"]], title="RSI Strategy Portfolio Value", ylabel="USD")


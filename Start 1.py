import pandas as pd
import numpy 
import matplotlib.pyplot as plt
import yfinance as yf
from data import listofsp500
from Functions import *

#plotData(calculate50DaySMA(getHistoricalData("AAPL", period="20y")), title="Stock price of AAPL", ylabel="Price in USD")


#plotData(rsi_sectorrotation(sector_period="30y", rsi_period=365), title="RSI Sector rotation", ylabel="RSI Factor")
#plotData(calculate_rsi(getHistoricalData("XOM")["Close"], period=14), title="RSI of XOM", ylabel="RSI factor")
#plotData(getHistoricalData("XOM")["Close"], title="Closing price of XOM")

data = getHistoricalData("XOM", period="3y")["Close"]
rsi = calculate_rsi(data)

PlotDataDualAxis(data, rsi, title="XOM Price vs RSI", ax1_label="Closing price in USD", ax2_label="RSI")

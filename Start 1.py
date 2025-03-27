import pandas as pd
import numpy 
import matplotlib.pyplot as plt
import yfinance as yf
from data import listofsp500
from Functions import *

#plotData(calculate50DaySMA(getHistoricalData("AAPL", period="20y")), title="Stock price of AAPL", ylabel="Price in USD")


plotData(rsi_sectorrotation(sector_period="30y", rsi_period=365), title="RSI Sector rotation", ylabel="RSI Factor")


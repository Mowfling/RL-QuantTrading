import pandas as pd
import numpy 
import matplotlib.pyplot as plt
import yfinance as yf
from data import listofsp500
from Functions import *

#plotPrice("AAPL", "4y")
#print(calculate50DaySMA("AAPL"))
#print(calculate200DaySMA("AAPL")) 
#plotData(calculateEMA("AAPL", 9), "APPL 9 day EMA")
plotData(rsi_sectorrotation(), title="Sector rotation")
#print(calculateStochasticOscillator(fetchData("AAPL")))
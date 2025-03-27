import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from data import listofsp500, sectors

__all__ = [
    'fetchStock',
    'fetchData',
    'plotPrice',
    'calculate50DaySMA',
    'calculate200DaySMA',
    'calculateEMA',
    'calculateTrueRange',
    'calculateAverageTrueRangeSMA',
    'calculateAverageTrueRangeEMA',
    'calculateStochasticOscillator',
    'plotData',
    'rsi_sectorrotation'
]

def fetchStock(ticker_symbol, showData = False, period = "1y"):
    ticker = yf.Ticker(ticker_symbol)
    historical_data = ticker.history(period)
    #financials = ticker.financials
    #actions = ticker.actions
    if (showData):
        print(ticker_symbol)
        print("Historical Data:")
        print(historical_data)
    #    print("\nFinancials:")
    #    print(financials)
    #    print("\nStock Actions:")
    #    print(actions)
    return ticker, historical_data

def getHistoricalData(ticker_symbol, period = "1y"):
    ticker = yf.Ticker(ticker_symbol)
    historical_data = ticker.history(period)
    return historical_data

def fetchData(ticker_symbol, period = "1y"):
    ticker = yf.Ticker(ticker_symbol)
    historical_data = ticker.history(period)
    return historical_data

def plotPrice(ticker_symbol, period):
    historical_data = getHistoricalData(ticker_symbol, period)
    plt.figure(figsize=(10, 6))  
    plt.plot(historical_data['Close'], label='Close Price')  
    plt.title(f'Historical Closing Prices of {ticker_symbol}') 
    plt.xlabel('Date')
    plt.ylabel('Close Price (USD)') 
    plt.legend()
    plt.show() 

def calculate50DaySMA(data, sma_period = 50, data_period_days = "200d"):
    return data['Close'].rolling(window=sma_period).mean()

def calculate200DaySMA(data, sma_period = 200, data_period_days = "1y"):
    return data['Close'].rolling(window=sma_period).mean()

def calculateEMA(data, period):
    return data['Close'].ewm(span=period, adjust=False).mean()

def calculateTrueRange(data):
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    true_range = np.maximum.reduce([high_low, high_close, low_close])
    return true_range

def calculateAverageTrueRangeSMA(data, atr_period = 14):
    true_range = calculateTrueRange(data)
    average_true_range = true_range.rolling(window = atr_period).mean()
    return average_true_range

def calculateAverageTrueRangeEMA(data, atr_period = 14):
    true_range = calculateTrueRange(data)
    average_true_range = true_range.ewm(span=atr_period, adjust=False).mean()
    return average_true_range

def calculateStochasticOscillator(data, k_period=14, d_period=3):
    low_min = data['Low'].rolling(window=k_period).min()
    high_max = data['High'].rolling(window=k_period).max()
    #%K calculation
    percent_k = ((data['Close'] - low_min) / (high_max - low_min)) * 100
    # %D calculation (3-period SMA of %K)
    percent_d = percent_k.rolling(window=d_period).mean()
    return percent_k, percent_d

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def rsi_sectorrotation(sector_period= "3mo", rsi_period = 14):
    rsi_data = {}
    for sector, ticker in sectors.items():
        data = fetchData(ticker, sector_period)
        rsi = calculate_rsi(data["Close"], rsi_period)
        rsi_data[sector] = rsi
    rsi_df = pd.DataFrame(rsi_data)
    return rsi_df


def plotData(data, title = "Default title", xlabel = "Date", ylabel = "Price USD"):
    plt.plot(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
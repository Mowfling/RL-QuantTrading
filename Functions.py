import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

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

def plotPrice(ticker_symbol, period):
    ticker, historical_data = fetchStock(ticker_symbol, False, period)
    plt.figure(figsize=(10, 6))  
    plt.plot(historical_data['Close'], label='Close Price')  
    plt.title(f'Historical Closing Prices of {ticker}') 
    plt.xlabel('Date')
    plt.ylabel('Close Price (USD)') 
    plt.legend()
    plt.show() 

def calculate50DaySMA(ticker_symbol, sma_period = 50, data_period_days = "200d"):
    ticker, hdata = fetchStock(ticker_symbol, False, data_period_days)
    return hdata['Close'].rolling(window=sma_period).mean()

def calculate200DaySMA(ticker_symbol, sma_period = 200, data_period_days = "1y"):
    ticker, hdata = fetchStock(ticker_symbol, False, data_period_days)
    return hdata['Close'].rolling(window=sma_period).mean()

def calculateEMA(ticker_symbol, period):
    ticker, hdata = fetchStock(ticker_symbol)
    return hdata['Close'].ewm(span=period, adjust=False).mean()

def calculateTrueRange(ticker_symbol, period):
    ticker = yf.Ticker(ticker_symbol)
    data = ticker.history(period)
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    true_range = np.maximum.reduce([high_low, high_close, low_close])
    return true_range

def calculateAverageTrueRangeSMA(ticker_symbol, data_period, atr_period):
    true_range = calculateTrueRange(ticker_symbol, data_period)
    average_true_range = true_range.rolling(window = atr_period).mean()
    return average_true_range

def calculateAverageTrueRangeEMA(ticker_symbol, data_period, atr_period):
    true_range = calculateTrueRange(ticker_symbol, data_period)
    average_true_range = true_range.ewm(span=atr_period, adjust=False).mean()
    return average_true_range

def plotData(data, title = "Default title", xlabel = "Date", ylabel = "Price USD"):
    plt.plot(data, color='blue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
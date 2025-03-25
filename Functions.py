#Simple way to scrape ticker names for now
import pandas as pd
import numpy
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
        print("\nFinancials:")
        print(financials)
        print("\nStock Actions:")
        print(actions)
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

def calculateSMA(ticker_symbol, sma_period = 50, data_period_days = "200d"):
    ticker, hdata = fetchStock(ticker_symbol, False, data_period_days)
    return hdata['Close'].rolling(window=sma_period).mean()


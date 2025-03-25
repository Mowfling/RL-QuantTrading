#Simple way to scrape ticker names for now
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import yfinance as yf

def fetchStock(ticker_symbol, showData = False, period_in_years = 1):
    ticker = yf.Ticker(ticker_symbol)
    historical_data = ticker.history(period=f"{period_in_years}y")
    financials = ticker.financials
    actions = ticker.actions
    if (showData):
        print(ticker_symbol)
        print("Historical Data:")
        print(historical_data)
        print("\nFinancials:")
        print(financials)
        print("\nStock Actions:")
        print(actions)
    return ticker, historical_data, financials, actions

def plotPrice(ticker_symbol, period_in_years):
    ticker, historical_data, financials, actions = fetchStock(ticker_symbol, False, period_in_years)
    plt.figure(figsize=(10, 6))  
    plt.plot(historical_data['Close'], label='Close Price')  
    plt.title(f'Historical Closing Prices of {ticker}') 
    plt.xlabel('Date')
    plt.ylabel('Close Price (USD)') 
    plt.legend()
    plt.show() 

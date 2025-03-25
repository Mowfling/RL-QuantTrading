#Simple way to scrape ticker names for now
import pandas as pd
import numpy
import matplotlib
import yfinance as yf


def fetch_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(url, attrs={'id': 'constituents'})[0]  
    tickers = table['Symbol'].tolist() 
    return tickers


def fetchStock(ticker_symbol, showData = True):
    ticker = yf.Ticker(ticker_symbol)
    historical_data = ticker.history(period="1y")
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
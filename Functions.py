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


def fetch_stock(ticker_symbol):
    print(ticker_symbol)
    ticker = ticker_symbol
    ticker = yf.Ticker(ticker_symbol)
    historical_data = ticker.history(period="1y")
    print("Historical Data:")
    print(historical_data)
    # Fetch basic financials
    financials = ticker.financials
    print("\nFinancials:")
    print(financials)

    # Fetch stock actions like dividends and splits
    actions = ticker.actions
    print("\nStock Actions:")
    print(actions)
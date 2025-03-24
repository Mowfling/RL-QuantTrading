#Simple way to scrape ticker names for now
import pandas as pd

def fetch_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(url, attrs={'id': 'constituents'})[0]  
    tickers = table['Symbol'].tolist() 
    return tickers

sp500_tickers = fetch_sp500_tickers()
print(sp500_tickers)
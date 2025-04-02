import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from data import listofsp500, sectors

__all__ = [
    'fetchStock',
    'plotPrice',
    'calculateSMA',
    'calculateEMA',
    'calculateTrueRange',
    'calculateAverageTrueRangeSMA',
    'calculateAverageTrueRangeEMA',
    'calculateStochasticOscillator',
    'plotData',
    'rsi_sectorrotation',
    'getHistoricalData',
    'calculate_rsi',
    'PlotDataDualAxis',
    'modelInput'
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

def getHistoricalData(ticker_symbol, period = "1y", interval_d = "1d", start="", end=""):
    ticker = yf.Ticker(ticker_symbol)
    if (interval_d != ""):
        historical_data = yf.download(ticker_symbol, start=start, end=end)
    else:
        historical_data = ticker.history(period)
    return historical_data

#Minute data only works for 7 last days
def getMinuteData(ticker_symbol, period = "1d"):
    data = yf.download(ticker_symbol, period, interval="1m")
    return data

def plotPrice(ticker_symbol, period):
    historical_data = getHistoricalData(ticker_symbol, period)
    plt.figure(figsize=(10, 6))  
    plt.plot(historical_data['Close'], label='Close Price')  
    plt.title(f'Historical Closing Prices of {ticker_symbol}') 
    plt.xlabel('Date')
    plt.ylabel('Close Price (USD)') 
    plt.legend()
    plt.show() 

def calculateSMA(data, sma_period = 50):
    return data['Close'].rolling(window=sma_period).mean()

def calculateEMA(data, period = 50):
    return data['Close'].ewm(span=period, adjust=False).mean()

def calculateTrueRange(data):
    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Close'].shift()).abs()
    low_close = (data['Low'] - data['Close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range  # returns a Series

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

def calculate_rsi(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def rsi_sectorrotation(sector_period= "3mo", rsi_period = 14):
    rsi_data = {}
    for sector, ticker in sectors.items():
        data = getHistoricalData(ticker, sector_period)
        rsi = calculate_rsi(data["Close"], rsi_period)
        rsi_data[sector] = rsi
    rsi_df = pd.DataFrame(rsi_data)
    return rsi_df

def plotData(data, title="Default title", xlabel="Date", ylabel="Price USD", show_graph=True):
    plt.figure(figsize=(10, 5))

    if hasattr(data, "columns"):
        for column in data.columns:
            plt.plot(data.index, data[column], label=column)
    else:
        plt.plot(data, label=data.name if data.name else "Series")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if (show_graph):
        plt.show()
    else:
        return plt

def PlotDataDualAxis(dataset1, dataset2, title="Default title", ax1_label = "Default label", ax2_label = "Default label", ax1_color="blue", ax2_color="orange", show_graph=True):
    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(dataset1, color=ax1_color, label=ax1_label)
    ax1.set_ylabel(ax1_label, color=ax1_color)
    ax1.tick_params(axis='y', labelcolor=ax1_color)

    ax2 = ax1.twinx()
    ax2.plot(dataset2, color=ax2_color, label=ax2_label)
    ax2.set_ylabel(ax2_label, color=ax2_color)
    ax2.tick_params(axis='y', labelcolor=ax2_color)

    # Add title and grid
    plt.title(title)
    fig.tight_layout()
    plt.grid(True)
    if (show_graph):
        plt.show()
    else:
        return plt

def calculate_obv(data):
    # Handle multi-ticker data: reduce to single ticker if necessary
    if isinstance(data.columns, pd.MultiIndex):
        raise ValueError("Please provide data for a single ticker (e.g., data.xs('AAPL', axis=1, level=1))")

    obv = [0]
    for i in range(1, len(data)):
        prev_close = data['Close'].iloc[i - 1]
        curr_close = data['Close'].iloc[i]
        volume = data['Volume'].iloc[i]

        if curr_close > prev_close:
            obv.append(obv[-1] + volume)
        elif curr_close < prev_close:
            obv.append(obv[-1] - volume)
        else:
            obv.append(obv[-1])

    return pd.Series(obv, index=data.index)

def calculate_intraday_vwap(data):
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    vwap = (typical_price * data['Volume']).cumsum() / data['Volume'].cumsum()
    return vwap

def modelInput(data):
    df = pd.DataFrame()
    df['Close'] = data['Close']
    df['SMA_10'] = calculateSMA(data, sma_period=10)
    df['SMA_30'] = calculateSMA(data, sma_period=30)
    df['SMA_50'] = calculateSMA(data, sma_period=50)
    df['EMA_30'] = calculateEMA(data, period=30)
    df['True_Range'] = calculateTrueRange(data)
    df['ATR_SMA'] = calculateAverageTrueRangeSMA(data, atr_period=14)
    df['ATR_EMA'] = calculateAverageTrueRangeEMA(data, atr_period=14)
    df['RSI'] = calculate_rsi(data['Close'], period=14)
    df['Stochastic_Oscillator%K'], df['Stochastic_Oscillator%D'] = calculateStochasticOscillator(data, k_period=14, d_period=3)
    df['OBV'] = calculate_obv(data)
    df.dropna(inplace=True)  # modify in place, but don't assign it
    return df

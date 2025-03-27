import pandas as pd
import numpy as np

def rsi_scaling_factor(rsi, threshold, direction="buy"):
    if direction == "buy":
        return 1 / (1 + np.exp((rsi - threshold) / 2))  # tweak divisor to change curve shape
    elif direction == "sell":
        return 1 / (1 + np.exp((threshold - rsi) / 2))
    return 0


def backtest_rsi_normalized(prices, rsi, 
                            buy_threshold=30, sell_threshold=70,
                            max_fraction=0.9,  # max % to buy/sell
                            initial_cash=10000):
    
    position = 0
    cash = initial_cash
    portfolio_values = []

    for i in range(1, len(prices)):
        price = prices.iloc[i]
        rsi_value = rsi.iloc[i]

        # Buy scaled by how low RSI is
        if cash > 0:
            factor = rsi_scaling_factor(rsi_value, buy_threshold, direction="buy")
            if factor > 0.01:  # avoid tiny trades
                buy_amount = cash * factor * max_fraction
                shares_bought = buy_amount / price
                position += shares_bought
                cash -= buy_amount

        # Sell scaled by how high RSI is
        if cash > 0:
            factor = rsi_scaling_factor(rsi_value, sell_threshold, direction="sell")
            if factor > 0.01:  # avoid tiny trades
                shares_to_sell = position * factor * max_fraction
                cash += shares_to_sell * price
                position -= shares_to_sell

        portfolio_value = cash + position * price
        portfolio_values.append(portfolio_value)

    result = pd.DataFrame({
        "Price": prices.iloc[1:], 
        "RSI": rsi.iloc[1:], 
        "Portfolio Value": portfolio_values
    })
    return result

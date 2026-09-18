import numpy as np
import pandas as pd
import ta

FEATURE_COLUMNS = [
    "ret_1d","ret_5d","ret_20d","vol_20d",
    "sma20_ratio","sma50_ratio","ema20_ratio",
    "rsi14","macd","macd_signal","bb_width","volume_change"
]

def add_features(df, horizon=5):
    x = df.copy()
    x["ret_1d"] = x["Close"].pct_change()
    x["ret_5d"] = x["Close"].pct_change(5)
    x["ret_20d"] = x["Close"].pct_change(20)
    x["vol_20d"] = x["ret_1d"].rolling(20).std() * np.sqrt(252)

    sma20 = x["Close"].rolling(20).mean()
    sma50 = x["Close"].rolling(50).mean()
    ema20 = x["Close"].ewm(span=20, adjust=False).mean()
    x["sma20_ratio"] = x["Close"]/sma20 - 1
    x["sma50_ratio"] = x["Close"]/sma50 - 1
    x["ema20_ratio"] = x["Close"]/ema20 - 1

    x["rsi14"] = ta.momentum.RSIIndicator(x["Close"], 14).rsi()
    m = ta.trend.MACD(x["Close"])
    x["macd"] = m.macd()
    x["macd_signal"] = m.macd_signal()
    b = ta.volatility.BollingerBands(x["Close"], 20, 2)
    x["bb_width"] = (b.bollinger_hband()-b.bollinger_lband())/x["Close"]
    x["volume_change"] = x["Volume"].pct_change()

    x["target_return"] = x["Close"].shift(-horizon)/x["Close"] - 1
    return x.replace([np.inf,-np.inf], np.nan).dropna()

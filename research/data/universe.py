from pathlib import Path
import pandas as pd

DEFAULT_UNIVERSE = [
    "RELIANCE.NS","HDFCBANK.NS","ICICIBANK.NS","INFY.NS","TCS.NS",
    "ITC.NS","LT.NS","SBIN.NS","BHARTIARTL.NS","HINDUNILVR.NS",
    "AXISBANK.NS","KOTAKBANK.NS","MARUTI.NS","SUNPHARMA.NS","TITAN.NS",
    "ASIANPAINT.NS","BAJFINANCE.NS","HCLTECH.NS","NTPC.NS","POWERGRID.NS"
]

def load_universe(path="data/universe.csv", max_tickers=None):
    p = Path(path)
    if p.exists():
        tickers = pd.read_csv(p)["ticker"].dropna().astype(str).tolist()
    else:
        tickers = DEFAULT_UNIVERSE
    return tickers[:max_tickers] if max_tickers else tickers

def save_universe(tickers, path="data/universe.csv"):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ticker": tickers}).to_csv(p, index=False)

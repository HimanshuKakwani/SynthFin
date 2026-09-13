import argparse
from pathlib import Path
import pandas as pd
import yfinance as yf
from .universe import load_universe

def download_one(ticker, start, end, raw_dir):
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = raw_dir / f"{ticker.replace('/','_')}.csv"
    df = yf.download(ticker, start=start, end=end, auto_adjust=True,
                     progress=False, threads=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    required = ["Open","High","Low","Close","Volume"]
    if df.empty or not set(required).issubset(df.columns):
        raise ValueError(f"No usable data for {ticker}")
    df = df[required].dropna()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.to_csv(out)
    return df

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--universe", default="data/universe.csv")
    p.add_argument("--max-tickers", type=int, default=None)
    args = p.parse_args()
    tickers = load_universe(args.universe, args.max_tickers)
    for t in tickers:
        try:
            df = download_one(t, args.start, args.end, "data/raw")
            print(f"[OK] {t}: {len(df)} rows")
        except Exception as e:
            print(f"[WARN] {t}: {e}")

if __name__ == "__main__":
    main()

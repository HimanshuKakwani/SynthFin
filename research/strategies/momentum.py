def signal(df, lookback=20): return (df.Close.pct_change(lookback)>0).astype(float)

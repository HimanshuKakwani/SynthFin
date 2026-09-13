import numpy as np, pandas as pd, ta
def signal(df, low=40, high=60):
 r=ta.momentum.RSIIndicator(df.Close,14).rsi(); s=np.where(r<low,1.0,np.where(r>high,0.0,np.nan)); return pd.Series(s,index=df.index).ffill().fillna(0)

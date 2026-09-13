import numpy as np, pandas as pd, ta
def signal(df):
 b=ta.volatility.BollingerBands(df.Close,20,2); s=np.where(df.Close<b.bollinger_lband(),1.0,np.where(df.Close>b.bollinger_hband(),0.0,np.nan)); return pd.Series(s,index=df.index).ffill().fillna(0)

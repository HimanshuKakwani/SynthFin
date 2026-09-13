import pandas as pd
def signal(df, fast=20, slow=50):
 f=df.Close.ewm(span=fast,adjust=False).mean(); s=df.Close.ewm(span=slow,adjust=False).mean(); return (f>s).astype(float)

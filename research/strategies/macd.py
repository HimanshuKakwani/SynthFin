import ta
def signal(df):
 m=ta.trend.MACD(df.Close); return (m.macd()>m.macd_signal()).astype(float)

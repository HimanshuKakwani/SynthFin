import numpy as np
import pandas as pd

def performance(returns, risk_free=0.0):
    r = pd.Series(returns).dropna()
    if len(r)==0: return {}
    wealth=(1+r).cumprod()
    years=len(r)/252
    cagr=wealth.iloc[-1]**(1/years)-1
    vol=r.std()*np.sqrt(252)
    sharpe=((r.mean()*252)-risk_free)/vol if vol>0 else np.nan
    down=r[r<0].std()*np.sqrt(252)
    sortino=((r.mean()*252)-risk_free)/down if down>0 else np.nan
    dd=wealth/wealth.cummax()-1
    return {
        "CAGR":float(cagr), "Volatility":float(vol),
        "Sharpe":float(sharpe), "Sortino":float(sortino),
        "MaxDrawdown":float(dd.min()), "TotalReturn":float(wealth.iloc[-1]-1)
    }

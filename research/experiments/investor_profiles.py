from pathlib import Path
import pandas as pd
from research.risk.risk_profile import PROFILE_LAMBDA

def run():
    out=pd.DataFrame([{"Profile":k,"RiskLambda":v} for k,v in PROFILE_LAMBDA.items()])
    Path("results/tables").mkdir(parents=True,exist_ok=True)
    out.to_csv("results/tables/investor_profiles_setup.csv",index=False)
    print(out)
    return out

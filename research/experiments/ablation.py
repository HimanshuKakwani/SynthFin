from pathlib import Path
import pandas as pd

def run():
    variants=[
        "Risk only",
        "Risk + ML",
        "Risk + ML + Strategy",
        "Risk + ML + Strategy + Rebalancing"
    ]
    out=pd.DataFrame({"Variant":variants})
    Path("results/tables").mkdir(parents=True,exist_ok=True)
    out.to_csv("results/tables/ablation_setup.csv",index=False)
    print(out)
    return out

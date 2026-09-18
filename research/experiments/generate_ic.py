from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


OBS = Path("results/tables/prediction_observations.csv")
OUT = Path("results/tables")


def generate_ic(model):
    df = pd.read_csv(OBS, parse_dates=["Date"])
    df = df[df["Model"] == model].copy()

    rows = []

    for date, group in df.groupby("Date"):
        group = group.dropna(subset=["Prediction", "ActualReturn"])

        if len(group) < 2:
            continue

        # Spearman IC is undefined when predictions or realized
        # returns are constant across the cross-section.
        if group["Prediction"].nunique() < 2:
            ic = np.nan
        elif group["ActualReturn"].nunique() < 2:
            ic = np.nan
        else:
            ic, _ = spearmanr(
                group["Prediction"],
                group["ActualReturn"],
            )

        rows.append({
            "Date": date,
            "IC": ic,
        })

    result = pd.DataFrame(rows, columns=["Date", "IC"])

    output = OUT / f"ic_timeseries_{model}.csv"
    result.to_csv(output, index=False)

    valid = result["IC"].dropna()

    print(
        f"{model}: "
        f"{len(result)} dates | "
        f"{len(valid)} valid ICs | "
        f"mean IC = {valid.mean():.6f} | "
        f"saved to {output}"
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    for model in [
        "Naive",
        "RandomForest",
        "XGBoost",
    ]:
        generate_ic(model)


if __name__ == "__main__":
    main()
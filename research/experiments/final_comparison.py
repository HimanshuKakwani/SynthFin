from pathlib import Path
import pandas as pd

FILES = {
    "FINAL_BACKTEST": "results/tables/FINAL_PORTFOLIO_RESULTS.csv",
    "TURNOVER_AWARE": "results/tables/FINAL_2025_TURNOVER_RESULTS.csv",
}

OUTPUT = Path("results/tables/FINAL_COMPARISON.csv")


def main():
    frames = []

    # Final backtest results
    path = Path(FILES["FINAL_BACKTEST"])
    if path.exists():
        df = pd.read_csv(path)

        df["Source"] = "FinalBacktest"
        frames.append(df)

    # Turnover-aware results
    path = Path(FILES["TURNOVER_AWARE"])
    if path.exists():
        df = pd.read_csv(path)

        # Keep only the selected conservative configuration
        df = df[
            (df["RiskProfile"] == "conservative")
            & (df["Gamma"] == 0.01)
        ].copy()

        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            "No final result files found."
        )

    result = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    # Put important columns first.
    preferred = [
        "Strategy",
        "Model",
        "TopK",
        "RiskProfile",
        "Gamma",
        "CAGR",
        "Volatility",
        "Sharpe",
        "Sortino",
        "MaxDrawdown",
        "TotalReturn",
        "AverageTurnover",
        "TotalTransactionCosts",
        "Observations",
        "Source",
    ]

    columns = [
        c for c in preferred
        if c in result.columns
    ]

    result = result[columns]

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT,
        index=False,
    )

    print("=" * 100)
    print("FINAL STRATEGY COMPARISON")
    print("=" * 100)
    print(result.to_string(index=False))
    print()
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()

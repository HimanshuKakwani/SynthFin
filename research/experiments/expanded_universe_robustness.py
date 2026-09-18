"""50-stock robustness experiment for ICCSDI revision.

Prerequisites:
1. data/universe_50.csv exists.
2. 2018-2025 processed price data for that universe exists.
3. XGB predictions have been generated with:
   python -m research.backtest.walk_forward --universe data/universe_50.csv \
       --output results/tables/walk_forward_predictions_50.csv
"""
from pathlib import Path
import pandas as pd

import research.experiments.final_unified_2025 as unified
PRED = Path("results/tables/walk_forward_predictions_50.csv")
OUT = Path("results/tables/REVISION_50STOCK_2025.csv")


def main():
    predictions = pd.read_csv(
    PRED,
    parse_dates=["Date"]
)

    predictions = predictions[
        predictions["Date"].dt.year == 2025
    ].copy()

    tickers = sorted(
        predictions["Ticker"].unique()
    )

    assert len(tickers) == 50, (
        f"Expected 50-stock robustness universe, "
        f"found {len(tickers)}"
    )

    print(
        f"Expanded universe: "
        f"{len(tickers)} unique stocks"
    )

    unified.DATA_DIR = Path("data/processed_50")

    prices = unified.load_prices(tickers)
    returns = prices.pct_change()

    rows = []
    for strategy, profile in [
    ("BuyHold", None),
    ("EqualWeight", None),
    ("RiskOnly", None),
    ("XGB_EqualWeight", None),
    ("XGB_Risk", "moderate"),
    ("XGB_TurnoverAware", "moderate"),
]:
        h = unified.run_strategy(
            predictions,
            returns,
            strategy,
            profile,
        )
        if h.empty:
            continue
        rows.append({
            "Universe": "Expanded50",
            "NStocksAvailable": len(tickers),
            "Strategy": strategy,
            "RiskProfile": profile or "",
            **unified.calculate_metrics(h),
        })

    result = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT, index=False)
    print(result.to_string(index=False))
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()

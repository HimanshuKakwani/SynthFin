from pathlib import Path

import numpy as np
import pandas as pd

from research.backtest.engine import (
    run_backtest,
    performance_metrics
)

from research.portfolio.optimizer import (
    equal_weight,
    risk_only_weights,
    ml_risk_weights
)


PREDICTION_FILE = (
    "results/tables/"
    "walk_forward_predictions.csv"
)


DATA_DIR = Path(
    "data/processed"
)


def load_returns(tickers):

    frames = []

    for ticker in tickers:

        filename = (
            ticker.replace("/", "_")
            + ".csv"
        )

        path = DATA_DIR / filename

        if not path.exists():
            continue

        df = pd.read_csv(
            path,
            index_col=0,
            parse_dates=True
        )

        df = df.sort_index()

        if "Close" not in df.columns:
            continue

        returns = (
            df["Close"]
            .pct_change()
            .rename(ticker)
        )

        frames.append(
            returns
        )

    if not frames:

        return pd.DataFrame()

    return pd.concat(
        frames,
        axis=1
    ).sort_index()


def make_equal_weight(k):

    def strategy(
        predictions,
        historical_returns
    ):

        selected = (
            predictions
            .sort_values(
                "Prediction",
                ascending=False
            )
            .head(k)
        )

        return equal_weight(
            selected["Ticker"]
        )

    return strategy


def make_risk_only(k):

    def strategy(
        predictions,
        historical_returns
    ):

        return risk_only_weights(
            predictions,
            historical_returns,
            k=k,
            max_weight=0.20
        )

    return strategy


def make_ml_risk(
    k,
    risk_profile
):

    def strategy(
        predictions,
        historical_returns
    ):

        return ml_risk_weights(
            predictions,
            historical_returns,
            k=k,
            risk_profile=risk_profile,
            max_weight=0.25
        )

    return strategy


def run():

    predictions = pd.read_csv(
        PREDICTION_FILE,
        parse_dates=["Date"]
    )

    predictions = predictions.sort_values(
        ["Date", "Ticker"]
    )

    tickers = sorted(
        predictions["Ticker"]
        .unique()
    )

    historical_returns = load_returns(
        tickers
    )

    strategies = {

        "EqualWeight_Top5":
            make_equal_weight(5),

        "EqualWeight_Top10":
            make_equal_weight(10),

        "RiskOnly_Top5":
            make_risk_only(5),

        "RiskOnly_Top10":
            make_risk_only(10),

        "XGB_Conservative":
            make_ml_risk(
                10,
                "conservative"
            ),

        "XGB_Moderate":
            make_ml_risk(
                10,
                "moderate"
            ),

        "XGB_Aggressive":
            make_ml_risk(
                10,
                "aggressive"
            )
    }

    all_results = []

    output_dir = Path(
        "results/tables"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    for name, strategy in strategies.items():

        print(
            f"\nRunning {name}"
        )

        history = run_backtest(
            predictions=predictions,
            historical_returns=historical_returns,
            weight_function=strategy,
            rebalance_every=5,
            initial_capital=1_000_000,
            transaction_cost_bps=10
        )

        metrics = performance_metrics(
            history
        )

        metrics["Strategy"] = name

        all_results.append(
            metrics
        )

        history.to_csv(
            output_dir /
            f"walkforward_{name}.csv",
            index=False
        )

    results = pd.DataFrame(
        all_results
    )

    columns = [
        "Strategy",
        "CAGR",
        "Volatility",
        "Sharpe",
        "Sortino",
        "MaxDrawdown",
        "TotalReturn",
        "AverageTurnover",
        "TotalTransactionCosts"
    ]

    results = results[
        [
            c for c in columns
            if c in results.columns
        ]
    ]

    results.to_csv(
        output_dir /
        "walkforward_portfolio_results.csv",
        index=False
    )

    print(
        "\n"
        + "=" * 90
    )

    print(
        "WALK-FORWARD PORTFOLIO RESULTS"
    )

    print(
        "=" * 90
    )

    print(
        results.to_string(
            index=False
        )
    )

    return results


if __name__ == "__main__":
    run()
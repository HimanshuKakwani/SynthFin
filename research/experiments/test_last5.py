from pathlib import Path

import numpy as np
import pandas as pd

from research.portfolio.optimizer import (
    equal_weight,
    risk_only_weights,
    ml_risk_weights,
)


PREDICTION_FILE = (
    "results/tables/"
    "walk_forward_predictions.csv"
)

DATA_DIR = Path("data/processed")


def load_daily_returns(tickers):

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
            parse_dates=True,
        )

        df = df.sort_index()

        if "Close" not in df.columns:
            continue

        returns = (
            df["Close"]
            .pct_change()
            .rename(ticker)
        )

        frames.append(returns)

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        axis=1,
    ).sort_index()


def turnover(
    previous,
    current,
):

    if previous is None:
        return 1.0

    tickers = set(
        previous.index
    ).union(
        current.index
    )

    previous = previous.reindex(
        tickers,
        fill_value=0.0,
    )

    current = current.reindex(
        tickers,
        fill_value=0.0,
    )

    return float(
        (
            current - previous
        ).abs().sum()
    )


def transaction_cost(
    previous,
    current,
    cost_bps=10,
):

    return (
        turnover(
            previous,
            current,
        )
        *
        cost_bps
        /
        10000.0
    )


def realized_5day_return(
    daily_returns,
    weights,
    start_date,
):

    dates = daily_returns.index

    future_dates = dates[
        dates > start_date
    ][:5]

    if len(future_dates) == 0:
        return np.nan

    future = daily_returns.loc[
        future_dates
    ]

    future = future[
        [
            ticker
            for ticker in weights.index
            if ticker in future.columns
        ]
    ]

    weights = weights.reindex(
        future.columns,
        fill_value=0.0,
    )

    # Portfolio daily returns.
    daily_portfolio_returns = (
        future * weights
    ).sum(axis=1)

    # Compound the five daily returns.
    return (
        (1 + daily_portfolio_returns)
        .prod()
        - 1
    )


def run_strategy(
    predictions,
    daily_returns,
    strategy_name,
    top_k=10,
    risk_profile=None,
    cost_bps=10,
):

    previous_weights = None

    rows = []

    for date in predictions["Date"].unique():

        current = predictions[
            predictions["Date"] == date
        ].copy()

        if current.empty:
            continue

        # ----------------------------
        # Build portfolio
        # ----------------------------

        if strategy_name == "EqualWeight":

            selected = (
                current
                .sort_values(
                    "Prediction",
                    ascending=False,
                )
                .head(top_k)
            )

            weights = equal_weight(
                selected["Ticker"]
            )

        elif strategy_name == "RiskOnly":

            weights = risk_only_weights(
                current,
                daily_returns.loc[
                    :date
                ],
                k=top_k,
                max_weight=0.20,
            )

        elif strategy_name == "XGB_Risk":

            weights = ml_risk_weights(
                current,
                daily_returns.loc[
                    :date
                ],
                k=top_k,
                risk_profile=risk_profile,
                max_weight=0.25,
            )

        else:

            raise ValueError(
                f"Unknown strategy: "
                f"{strategy_name}"
            )

        if weights.empty:
            continue

        cost = transaction_cost(
            previous_weights,
            weights,
            cost_bps,
        )

        realized = realized_5day_return(
            daily_returns,
            weights,
            date,
        )

        if np.isnan(realized):
            continue

        net_return = (
            realized - cost
        )

        rows.append({
            "Date": date,
            "Strategy": strategy_name,
            "TopK": top_k,
            "RiskProfile": (
                risk_profile
                if risk_profile
                else ""
            ),
            "Gross5DReturn": realized,
            "TransactionCost": cost,
            "Net5DReturn": net_return,
            "Turnover": turnover(
                previous_weights,
                weights,
            ),
            "NStocks": len(weights),
        })

        previous_weights = weights

    return pd.DataFrame(rows)


def main():

    predictions = pd.read_csv(
        PREDICTION_FILE,
        parse_dates=["Date"],
    )

    predictions = predictions.sort_values(
        "Date"
    )

    # ------------------------------------------------
    # IMPORTANT:
    # Last five prediction/rebalance dates.
    # ------------------------------------------------

    all_dates = sorted(
        predictions["Date"].unique()
    )

    last_five_dates = all_dates[-5:]

    test = predictions[
        predictions["Date"].isin(
            last_five_dates
        )
    ].copy()

    print()
    print("=" * 80)
    print("LAST 5 PREDICTION-DATE PORTFOLIO TEST")
    print("=" * 80)

    print(
        "Dates:"
    )

    for date in last_five_dates:
        print(
            f"  {pd.Timestamp(date).date()}"
        )

    print()

    tickers = sorted(
        test["Ticker"].unique()
    )

    daily_returns = load_daily_returns(
        tickers
    )

    strategies = [
        (
            "EqualWeight",
            5,
            None,
        ),
        (
            "EqualWeight",
            10,
            None,
        ),
        (
            "RiskOnly",
            5,
            None,
        ),
        (
            "RiskOnly",
            10,
            None,
        ),
        (
            "XGB_Risk",
            10,
            "conservative",
        ),
        (
            "XGB_Risk",
            10,
            "moderate",
        ),
        (
            "XGB_Risk",
            10,
            "aggressive",
        ),
    ]

    results = []

    for (
        strategy,
        k,
        risk_profile,
    ) in strategies:

        print(
            f"Running "
            f"{strategy} "
            f"Top-{k}"
            + (
                f" {risk_profile}"
                if risk_profile
                else ""
            )
        )

        result = run_strategy(
            test,
            daily_returns,
            strategy,
            top_k=k,
            risk_profile=risk_profile,
            cost_bps=10,
        )

        if result.empty:
            continue

        results.append(result)

    if not results:

        print(
            "\nNo results generated."
        )

        return

    results = pd.concat(
        results,
        ignore_index=True,
    )

    print()
    print("=" * 80)
    print("5-DAY TEST RESULTS")
    print("=" * 80)

    print(
        results.to_string(
            index=False
        )
    )

    Path(
        "results/tables"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        "results/tables/"
        "last5_portfolio_test.csv",
        index=False,
    )

    print()
    print(
        "Saved:"
    )

    print(
        "results/tables/"
        "last5_portfolio_test.csv"
    )


if __name__ == "__main__":
    main()
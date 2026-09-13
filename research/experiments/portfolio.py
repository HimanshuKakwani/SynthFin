from pathlib import Path

import numpy as np
import pandas as pd

from research.portfolio.optimizer import (
    build_portfolio,
)


PREDICTION_FILE = (
    "results/tables/"
    "prediction_observations.csv"
)


def portfolio_returns(
    selected,
):
    """
    Calculate the portfolio's realized return
    for one rebalance date.
    """

    if selected.empty:
        return 0.0

    return (
        selected["Weight"]
        *
        selected["ActualReturn"]
    ).sum()


def transaction_cost(
    previous_weights,
    current_weights,
    cost_bps=10,
):
    """
    Turnover-based transaction cost.

    cost_bps=10 means 0.10%.
    """

    if previous_weights is None:

        turnover = 1.0

    else:

        all_tickers = set(
            previous_weights.index
        ).union(
            current_weights.index
        )

        previous = (
            previous_weights
            .reindex(
                all_tickers,
                fill_value=0.0
            )
        )

        current = (
            current_weights
            .reindex(
                all_tickers,
                fill_value=0.0
            )
        )

        turnover = (
            current - previous
        ).abs().sum()

    return turnover * (
        cost_bps / 10000.0
    )


def run_strategy(
    predictions,
    model="XGBoost",
    k=5,
    risk_profile="moderate",
    rebalance_frequency=5,
    cost_bps=10,
):
    """
    Run a simple chronological portfolio simulation.

    Predictions are assumed to represent forward 5-day returns.
    """

    data = predictions[
        predictions["Model"] == model
    ].copy()

    data["Date"] = pd.to_datetime(
        data["Date"]
    )

    data = data.sort_values(
        ["Date", "Ticker"]
    )

    dates = sorted(
        data["Date"].unique()
    )

    portfolio_history = []

    previous_weights = None

    for step, date in enumerate(dates):

        if step % rebalance_frequency != 0:
            continue

        current = data[
            data["Date"] == date
        ].copy()

        if len(current) < k:
            continue

        selected, weights = build_portfolio(
            current,
            k=k,
            risk_profile=risk_profile,
        )

        realized_return = portfolio_returns(
            selected
        )

        cost = transaction_cost(
            previous_weights,
            weights,
            cost_bps=cost_bps,
        )

        net_return = (
            realized_return - cost
        )

        portfolio_history.append({
            "Date": date,
            "GrossReturn": realized_return,
            "TransactionCost": cost,
            "NetReturn": net_return,
            "Turnover": cost / (
                cost_bps / 10000
            )
            if cost_bps > 0
            else 0.0,
            "NStocks": len(selected),
        })

        previous_weights = weights

    return pd.DataFrame(
        portfolio_history
    )


def performance_metrics(
    returns,
):
    """
    Calculate portfolio performance metrics.
    """

    returns = pd.Series(
        returns
    ).dropna()

    if returns.empty:

        return {
            "CAGR": np.nan,
            "Volatility": np.nan,
            "Sharpe": np.nan,
            "Sortino": np.nan,
            "MaxDrawdown": np.nan,
        }

    wealth = (
        1.0 + returns
    ).cumprod()

    years = len(returns) / 252.0

    if years > 0:

        cagr = (
            wealth.iloc[-1]
            ** (1 / years)
        ) - 1

    else:

        cagr = np.nan

    volatility = (
        returns.std(ddof=1)
        *
        np.sqrt(252)
    )

    if volatility > 0:

        sharpe = (
            returns.mean()
            /
            returns.std(ddof=1)
        ) * np.sqrt(252)

    else:

        sharpe = np.nan

    downside = returns[
        returns < 0
    ]

    if len(downside) > 0:

        downside_std = (
            downside.std(ddof=1)
            *
            np.sqrt(252)
        )

        if downside_std > 0:

            sortino = (
                returns.mean()
                *
                252
                /
                downside_std
            )

        else:

            sortino = np.nan

    else:

        sortino = np.nan

    running_max = wealth.cummax()

    drawdown = (
        wealth / running_max
    ) - 1

    max_drawdown = drawdown.min()

    return {
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MaxDrawdown": max_drawdown,
    }


def run():

    path = Path(
        PREDICTION_FILE
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Missing {PREDICTION_FILE}"
        )

    predictions = pd.read_csv(
        path,
        parse_dates=["Date"],
    )

    results = []

    configurations = [
        (
            "XGBoost",
            5,
            "conservative",
        ),
        (
            "XGBoost",
            5,
            "moderate",
        ),
        (
            "XGBoost",
            5,
            "aggressive",
        ),
        (
            "XGBoost",
            10,
            "moderate",
        ),
    ]

    for (
        model,
        k,
        risk_profile,
    ) in configurations:

        print(
            f"\nRunning {model} | "
            f"Top {k} | "
            f"{risk_profile}"
        )

        history = run_strategy(
            predictions,
            model=model,
            k=k,
            risk_profile=risk_profile,
            rebalance_frequency=5,
            cost_bps=10,
        )

        metrics = performance_metrics(
            history["NetReturn"]
        )

        metrics[
            "Model"
        ] = model

        metrics[
            "TopK"
        ] = k

        metrics[
            "RiskProfile"
        ] = risk_profile

        results.append(
            metrics
        )

        filename = (
            f"portfolio_"
            f"{model}_"
            f"top{k}_"
            f"{risk_profile}.csv"
        )

        history.to_csv(
            Path(
                "results/tables"
            ) / filename,
            index=False,
        )

    results_df = pd.DataFrame(
        results
    )

    Path(
        "results/tables"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        "results/tables/"
        "portfolio_metrics.csv",
        index=False,
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "PORTFOLIO RESULTS"
    )

    print(
        "=" * 80
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    return results_df


if __name__ == "__main__":
    run()
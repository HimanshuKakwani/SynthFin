from pathlib import Path

import numpy as np
import pandas as pd

from research.portfolio.turnover_optimizer import (
    optimize_portfolio,
    turnover,
    risk_aversion_for_profile,
)


PREDICTIONS = (
    "results/tables/"
    "walk_forward_predictions.csv"
)

DATA_DIR = Path(
    "data/processed"
)

COST_BPS = 10

HOLDING_DAYS = 5

INITIAL_CAPITAL = 1_000_000


def load_prices(tickers):

    frames = []

    for ticker in tickers:

        path = (
            DATA_DIR
            /
            (
                ticker.replace("/", "_")
                + ".csv"
            )
        )

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

        frames.append(
            df["Close"].rename(
                ticker
            )
        )

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        axis=1,
    ).sort_index()


def calculate_metrics(history):

    if history.empty:
        return {}

    returns = history[
        "NetReturn"
    ]

    equity = history[
        "PortfolioValue"
    ]

    periods_per_year = (
        252 / HOLDING_DAYS
    )

    years = (
        len(returns)
        /
        periods_per_year
    )

    if years <= 0:
        return {}

    cagr = (
        equity.iloc[-1]
        /
        equity.iloc[0]
    ) ** (
        1.0 / years
    ) - 1.0

    volatility = (
        returns.std(ddof=1)
        *
        np.sqrt(
            periods_per_year
        )
    )

    if (
        returns.std(ddof=1)
        > 1e-12
    ):

        sharpe = (
            returns.mean()
            /
            returns.std(ddof=1)
        ) * np.sqrt(
            periods_per_year
        )

    else:

        sharpe = np.nan

    downside = returns[
        returns < 0
    ]

    if len(downside) > 1:

        downside_std = (
            downside.std(ddof=1)
            *
            np.sqrt(
                periods_per_year
            )
        )

        if downside_std > 0:

            sortino = (
                returns.mean()
                *
                periods_per_year
                /
                downside_std
            )

        else:

            sortino = np.nan

    else:

        sortino = np.nan

    running_max = (
        equity.cummax()
    )

    drawdown = (
        equity
        /
        running_max
        -
        1.0
    )

    return {
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MaxDrawdown": drawdown.min(),
        "TotalReturn": (
            equity.iloc[-1]
            /
            equity.iloc[0]
            -
            1.0
        ),
        "AverageTurnover":
            history[
                "Turnover"
            ].mean(),
        "TotalTransactionCosts":
            history[
                "TransactionCost"
            ].sum(),
        "Observations":
            len(history),
    }


def run_strategy(
    predictions,
    returns,
    gamma,
    risk_profile,
    max_turnover=None,
    k=10,
):

    dates = np.array(
        sorted(
            predictions[
                "Date"
            ].unique()
        )
    )

    rebalance_dates = dates[
        ::HOLDING_DAYS
    ]

    previous_weights = None

    capital = (
        INITIAL_CAPITAL
    )

    rows = []

    risk_aversion = (
        risk_aversion_for_profile(
            risk_profile
        )
    )

    for date in rebalance_dates:

        idx = np.where(
            dates == date
        )[0]

        if len(idx) == 0:
            continue

        idx = idx[0]

        future_dates = dates[
            idx + 1:
            idx + 1 + HOLDING_DAYS
        ]

        # Need a complete 5-day
        # forward holding period.
        if len(future_dates) < HOLDING_DAYS:
            continue

        current = predictions[
            predictions["Date"]
            == date
        ].copy()

        if current.empty:
            continue

        # CRITICAL:
        # Only data strictly before the
        # prediction date is available.
        historical = returns.loc[
            returns.index < date
        ]

        if historical.empty:
            continue

        weights = optimize_portfolio(
            predictions=current,
            historical_returns=historical,
            previous_weights=previous_weights,
            k=k,
            risk_aversion=risk_aversion,
            turnover_penalty=gamma,
            max_weight=0.20,
            max_turnover=max_turnover,
        )

        if weights.empty:
            continue

        available = [
            ticker
            for ticker in weights.index
            if ticker in returns.columns
        ]

        weights = weights[
            available
        ]

        period = returns.loc[
            future_dates,
            available,
        ]

        daily_portfolio = (
            period
            *
            weights
        ).sum(axis=1)

        gross_return = (
            1.0
            +
            daily_portfolio
        ).prod() - 1.0

        current_turnover = (
            turnover(
                previous_weights,
                weights,
            )
        )

        transaction_cost = (
            current_turnover
            *
            COST_BPS
            /
            10000.0
        )

        net_return = (
            gross_return
            -
            transaction_cost
        )

        capital *= (
            1.0
            +
            net_return
        )

        rows.append({
            "Date": date,
            "PortfolioValue": capital,
            "GrossReturn": gross_return,
            "TransactionCost":
                transaction_cost,
            "NetReturn": net_return,
            "Turnover":
                current_turnover,
            "NStocks":
                len(weights),
            "Gamma": gamma,
            "RiskProfile":
                risk_profile,
            "MaxTurnover":
                (
                    max_turnover
                    if max_turnover is not None
                    else ""
                ),
        })

        previous_weights = weights

    return pd.DataFrame(rows)


def main():

    predictions = pd.read_csv(
        PREDICTIONS,
        parse_dates=["Date"],
    )

    predictions = predictions.sort_values(
        ["Date", "Ticker"]
    )

    tickers = sorted(
        predictions[
            "Ticker"
        ].unique()
    )

    prices = load_prices(
        tickers
    )

    returns = prices.pct_change()

    output = Path(
        "results/tables"
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------
    # First experiment:
    # explicit turnover penalty
    # --------------------------------------------------

    gammas = [
        0.0,
        0.1,
        0.5,
        1.0,
        2.0,
        5.0,
    ]

    profiles = [
        "conservative",
        "moderate",
        "aggressive",
    ]

    summaries = []

    for profile in profiles:

        for gamma in gammas:

            print(
                f"Running "
                f"{profile:<12} "
                f"| gamma={gamma}"
            )

            history = run_strategy(
                predictions,
                returns,
                gamma=gamma,
                risk_profile=profile,
                max_turnover=None,
                k=10,
            )

            if history.empty:
                continue

            metrics = calculate_metrics(
                history
            )

            metrics[
                "RiskProfile"
            ] = profile

            metrics[
                "Gamma"
            ] = gamma

            metrics[
                "MaxTurnover"
            ] = ""

            summaries.append(
                metrics
            )

    # --------------------------------------------------
    # Second experiment:
    # hard turnover constraints
    # --------------------------------------------------

    turnover_limits = [
        0.25,
        0.50,
        0.75,
        1.00,
    ]

    for profile in profiles:

        for limit in turnover_limits:

            print(
                f"Running "
                f"{profile:<12} "
                f"| max_turnover={limit}"
            )

            history = run_strategy(
                predictions,
                returns,
                gamma=0.0,
                risk_profile=profile,
                max_turnover=limit,
                k=10,
            )

            if history.empty:
                continue

            metrics = calculate_metrics(
                history
            )

            metrics[
                "RiskProfile"
            ] = profile

            metrics[
                "Gamma"
            ] = 0.0

            metrics[
                "MaxTurnover"
            ] = limit

            summaries.append(
                metrics
            )

    results = pd.DataFrame(
        summaries
    )

    columns = [
        "RiskProfile",
        "Gamma",
        "MaxTurnover",
        "CAGR",
        "Volatility",
        "Sharpe",
        "Sortino",
        "MaxDrawdown",
        "TotalReturn",
        "AverageTurnover",
        "TotalTransactionCosts",
        "Observations",
    ]

    results = results[
        columns
    ]

    results.to_csv(
        output
        /
        "PROPER_TURNOVER_EXPERIMENT.csv",
        index=False,
    )

    print()
    print(
        "=" * 120
    )

    print(
        "PROPER TURNOVER-AWARE RESULTS"
    )

    print(
        "=" * 120
    )

    print(
        results.to_string(
            index=False
        )
    )

    print()
    print(
        "Saved:"
    )

    print(
        output
        /
        "PROPER_TURNOVER_EXPERIMENT.csv"
    )


if __name__ == "__main__":
    main()
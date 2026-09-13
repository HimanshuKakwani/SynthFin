import numpy as np
import pandas as pd

from research.backtest.transaction_costs import (
    calculate_transaction_cost
)


def performance_metrics(history):

    if history.empty:

        return {}

    returns = history[
        "NetReturn"
    ].dropna()

    equity = history[
        "PortfolioValue"
    ]

    if len(returns) == 0:
        return {}

    # Each observation represents one 5-trading-day period.
    periods_per_year = 252 / 5

    years = (
        len(returns)
        /
        periods_per_year
    )

    if years > 0:

        cagr = (
            equity.iloc[-1]
            /
            equity.iloc[0]
        ) ** (
            1.0 / years
        ) - 1.0

    else:

        cagr = np.nan

    volatility = (
        returns.std(ddof=1)
        *
        np.sqrt(periods_per_year)
    )

    if returns.std(ddof=1) > 0:

        sharpe = (
            returns.mean()
            /
            returns.std(ddof=1)
        ) * np.sqrt(periods_per_year)

    else:

        sharpe = np.nan

    downside = returns[
        returns < 0
    ]

    if len(downside) > 1:

        downside_std = (
            downside.std(ddof=1)
            *
            np.sqrt(periods_per_year)
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

    running_max = equity.cummax()

    drawdown = (
        equity
        /
        running_max
    ) - 1.0

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
        ) - 1.0,
        "AverageTurnover": history[
            "Turnover"
        ].mean(),
        "TotalTransactionCosts": history[
            "TransactionCost"
        ].sum()
    }


def run_backtest(
    predictions,
    historical_returns,
    weight_function,
    rebalance_every=5,
    initial_capital=1_000_000,
    transaction_cost_bps=10
):
    """
    Run a portfolio backtest.

    predictions:
        Date, Ticker, Prediction, ActualReturn

    historical_returns:
        Date-indexed dataframe of daily stock returns.

    weight_function:
        Function receiving (prediction_slice, historical_returns)
        and returning a weight Series.
    """

    predictions = predictions.copy()

    predictions["Date"] = pd.to_datetime(
        predictions["Date"]
    )

    predictions = predictions.sort_values(
        "Date"
    )

    dates = sorted(
        predictions["Date"].unique()
    )

    # Rebalance every 5 prediction dates.
    rebalance_dates = dates[
        ::rebalance_every
    ]

    portfolio_value = initial_capital

    previous_weights = None

    history = []

    for date in rebalance_dates:

        current = predictions[
            predictions["Date"] == date
        ].copy()

        if current.empty:
            continue

        weights = weight_function(
            current,
            historical_returns
        )

        if weights is None or weights.empty:
            continue

        # Next 5 prediction observations are the
        # holding period.
        date_position = dates.index(
            date
        )

        holding_dates = dates[
            date_position:
            date_position + rebalance_every
        ]

        if len(holding_dates) < 1:
            continue

        holding = predictions[
            predictions["Date"].isin(
                holding_dates
            )
        ].copy()

        # Aggregate each stock's realized target return.
        realized = (
            holding
            .groupby("Ticker")
            ["ActualReturn"]
            .mean()
        )

        realized = realized.reindex(
            weights.index
        ).fillna(0.0)

        gross_return = (
            realized * weights
        ).sum()

        transaction_cost, turnover = (
            calculate_transaction_cost(
                previous_weights,
                weights,
                transaction_cost_bps
            )
        )

        net_return = (
            gross_return
            -
            transaction_cost
        )

        portfolio_value *= (
            1.0 + net_return
        )

        history.append({
            "Date": date,
            "PortfolioValue": portfolio_value,
            "GrossReturn": gross_return,
            "TransactionCost": transaction_cost,
            "NetReturn": net_return,
            "Turnover": turnover,
            "NStocks": len(weights)
        })

        previous_weights = weights

    history = pd.DataFrame(
        history
    )

    return history
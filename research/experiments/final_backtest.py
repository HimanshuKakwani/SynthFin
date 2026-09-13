from pathlib import Path

import numpy as np
import pandas as pd

from research.portfolio.optimizer import (
    equal_weight,
    risk_only_weights,
    ml_equal_weights,
    ml_risk_weights,
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
                +
                ".csv"
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

        prices = (
            df["Close"]
            .rename(ticker)
        )

        frames.append(
            prices
        )

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        axis=1,
    ).sort_index()


def daily_returns(prices):

    return prices.pct_change()


def turnover(
    previous,
    current,
):

    if previous is None:
        return 1.0

    universe = (
        set(previous.index)
        |
        set(current.index)
    )

    old = previous.reindex(
        universe,
        fill_value=0.0,
    )

    new = current.reindex(
        universe,
        fill_value=0.0,
    )

    return float(
        (new - old)
        .abs()
        .sum()
    )


def transaction_cost(
    previous,
    current,
):

    return (
        turnover(
            previous,
            current,
        )
        *
        COST_BPS
        /
        10000.0
    )


def next_holding_period(
    dates,
    date,
):

    positions = np.where(
        dates == date
    )[0]

    if len(positions) == 0:
        return None

    idx = positions[0]

    future = dates[
        idx + 1:
        idx + 1 + HOLDING_DAYS
    ]

    if len(future) < HOLDING_DAYS:
        return None

    return future


def realized_return(
    returns,
    weights,
    future_dates,
):

    available = [
        t for t in weights.index
        if t in returns.columns
    ]

    if not available:
        return np.nan

    w = weights[
        available
    ]

    period = returns.loc[
        future_dates,
        available,
    ]

    daily_portfolio = (
        period * w
    ).sum(axis=1)

    return float(
        (1.0 + daily_portfolio)
        .prod()
        -
        1.0
    )


def metrics(history):

    if history.empty:
        return {}

    r = history[
        "NetReturn"
    ]

    equity = history[
        "PortfolioValue"
    ]

    periods_per_year = (
        252 / HOLDING_DAYS
    )

    annual_return = (
        equity.iloc[-1]
        /
        equity.iloc[0]
    ) ** (
        periods_per_year
        /
        len(r)
    ) - 1

    vol = (
        r.std(ddof=1)
        *
        np.sqrt(
            periods_per_year
        )
    )

    if r.std(ddof=1) > 0:

        sharpe = (
            r.mean()
            /
            r.std(ddof=1)
        ) * np.sqrt(
            periods_per_year
        )

    else:

        sharpe = np.nan

    downside = r[r < 0]

    if len(downside) > 1:

        downside_std = (
            downside.std(ddof=1)
            *
            np.sqrt(
                periods_per_year
            )
        )

        sortino = (
            r.mean()
            *
            periods_per_year
            /
            downside_std
        )

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
        1
    )

    return {
        "CAGR": annual_return,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MaxDrawdown": drawdown.min(),
        "TotalReturn": (
            equity.iloc[-1]
            /
            equity.iloc[0]
            -
            1
        ),
        "AverageTurnover": (
            history[
                "Turnover"
            ].mean()
        ),
        "TotalTransactionCosts": (
            history[
                "TransactionCost"
            ].sum()
        ),
        "Observations": len(history),
    }


def run_strategy(
    name,
    predictions,
    returns,
    universe,
    top_k=10,
    risk_profile=None,
):

    dates = np.array(
        sorted(
            predictions["Date"]
            .unique()
        )
    )

    # Rebalance every five trading dates.
    rebalance_dates = dates[
        ::HOLDING_DAYS
    ]

    capital = (
        INITIAL_CAPITAL
    )

    previous_weights = None

    history = []

    for date in rebalance_dates:

        future_dates = (
            next_holding_period(
                dates,
                date,
            )
        )

        if future_dates is None:
            continue

        current = predictions[
            predictions["Date"]
            == date
        ].copy()

        if current.empty:
            continue

        historical_returns = returns.loc[
            returns.index < date
        ]

        if name == "BuyHold":

            weights = equal_weight(
                universe
            )

        elif name == "EqualWeight":

            weights = equal_weight(
                universe
            )

        elif name == "RiskOnly":

            weights = risk_only_weights(
                universe,
                historical_returns,
                k=top_k,
                max_weight=0.20,
            )

        elif name == "XGB_EqualWeight":

            weights = ml_equal_weights(
                current,
                k=top_k,
            )

        elif name == "XGB_Risk":

            weights = ml_risk_weights(
                current,
                historical_returns,
                k=top_k,
                risk_profile=risk_profile,
                max_weight=0.25,
            )

        else:

            raise ValueError(
                name
            )

        if weights.empty:
            continue

        cost = transaction_cost(
            previous_weights,
            weights,
        )

        gross = realized_return(
            returns,
            weights,
            future_dates,
        )

        if np.isnan(gross):
            continue

        net = (
            gross
            -
            cost
        )

        capital *= (
            1.0 + net
        )

        history.append({
            "Date": date,
            "PortfolioValue": capital,
            "GrossReturn": gross,
            "TransactionCost": cost,
            "NetReturn": net,
            "Turnover": turnover(
                previous_weights,
                weights,
            ),
            "NStocks": len(weights),
        })

        previous_weights = weights

    return pd.DataFrame(
        history
    )


def main():

    predictions = pd.read_csv(
        PREDICTIONS,
        parse_dates=["Date"],
    )

    predictions = predictions.sort_values(
        ["Date", "Ticker"]
    )

    universe = sorted(
        predictions[
            "Ticker"
        ].unique()
    )

    prices = load_prices(
        universe
    )

    returns = daily_returns(
        prices
    )

    strategies = [

        (
            "BuyHold",
            20,
            None,
        ),

        (
            "EqualWeight",
            20,
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
            "XGB_EqualWeight",
            5,
            None,
        ),

        (
            "XGB_EqualWeight",
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

    output = Path(
        "results/tables"
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    summaries = []

    for (
        name,
        k,
        profile,
    ) in strategies:

        print(
            f"\nRunning {name}"
            +
            (
                f" | {profile}"
                if profile
                else ""
            )
        )

        history = run_strategy(
            name,
            predictions,
            returns,
            universe,
            top_k=k,
            risk_profile=profile,
        )

        if history.empty:
            continue

        summary = metrics(
            history
        )

        summary[
            "Strategy"
        ] = name

        summary[
            "TopK"
        ] = k

        summary[
            "RiskProfile"
        ] = (
            profile
            if profile
            else ""
        )

        summaries.append(
            summary
        )

        history.to_csv(
            output
            /
            (
                "final_"
                +
                name
                +
                (
                    "_"
                    +
                    profile
                    if profile
                    else ""
                )
                +
                ".csv"
            ),
            index=False,
        )

    result = pd.DataFrame(
        summaries
    )

    columns = [
        "Strategy",
        "TopK",
        "RiskProfile",
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

    result = result[
        columns
    ]

    result.to_csv(
        output
        /
        "FINAL_PORTFOLIO_RESULTS.csv",
        index=False,
    )

    print()
    print("=" * 100)
    print(
        "FINAL WALK-FORWARD PORTFOLIO RESULTS"
    )
    print("=" * 100)

    print(
        result.to_string(
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
        "FINAL_PORTFOLIO_RESULTS.csv"
    )


if __name__ == "__main__":
    main()
import numpy as np
import pandas as pd


def run_buy_and_hold(
    prices,
    initial_capital=1_000_000,
):
    """
    True buy-and-hold portfolio.

    Equal capital is invested initially.
    No rebalancing occurs afterwards.
    """

    prices = prices.copy()

    prices = prices.sort_index()

    prices = prices.dropna(
        axis=1,
        how="all",
    )

    if prices.empty:
        return pd.DataFrame()

    # First valid price for each asset.
    initial_prices = prices.iloc[0]

    valid = initial_prices.notna()

    prices = prices.loc[
        :,
        valid.index[valid]
    ]

    initial_prices = (
        prices.iloc[0]
    )

    n = len(
        initial_prices
    )

    if n == 0:
        return pd.DataFrame()

    # Equal initial capital.
    capital_per_stock = (
        initial_capital / n
    )

    shares = (
        capital_per_stock
        /
        initial_prices
    )

    portfolio_value = (
        prices
        .multiply(shares, axis=1)
        .sum(axis=1)
    )

    history = pd.DataFrame({
        "PortfolioValue":
            portfolio_value,
    })

    history[
        "DailyReturn"
    ] = (
        history[
            "PortfolioValue"
        ].pct_change()
    )

    return history
import numpy as np
import pandas as pd


def equal_weight(tickers):
    tickers = list(tickers)

    if not tickers:
        return pd.Series(dtype=float)

    return pd.Series(
        1.0 / len(tickers),
        index=tickers,
        dtype=float,
    )


def top_k(predictions, k=5):
    return (
        predictions
        .dropna(subset=["Prediction"])
        .sort_values(
            "Prediction",
            ascending=False,
        )
        .head(k)
        .copy()
    )


def inverse_volatility_weights(
    returns,
    tickers,
    max_weight=0.20,
):
    """
    Risk-only allocation.

    IMPORTANT:
    This function does not use ML predictions.
    """

    available = [
        t for t in tickers
        if t in returns.columns
    ]

    if not available:
        return pd.Series(dtype=float)

    recent = returns[
        available
    ].tail(60)

    volatility = recent.std()

    volatility = volatility.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    volatility = volatility.dropna()

    volatility = volatility[
        volatility > 0
    ]

    if volatility.empty:
        return pd.Series(dtype=float)

    inverse_vol = 1.0 / volatility

    weights = (
        inverse_vol
        /
        inverse_vol.sum()
    )

    return apply_max_weight(
        weights,
        max_weight,
    )


def risk_only_weights(
    available_tickers,
    returns,
    k=10,
    max_weight=0.20,
):
    """
    Select the k lowest-volatility stocks
    from the entire investment universe.
    """

    available_tickers = [
        t for t in available_tickers
        if t in returns.columns
    ]

    if not available_tickers:
        return pd.Series(dtype=float)

    recent = returns[
        available_tickers
    ].tail(60)

    volatility = recent.std()

    volatility = volatility.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    volatility = volatility[
        volatility > 0
    ]

    if volatility.empty:
        return pd.Series(dtype=float)

    selected = (
        volatility
        .sort_values()
        .head(k)
        .index
        .tolist()
    )

    return inverse_volatility_weights(
        returns,
        selected,
        max_weight=max_weight,
    )


def ml_equal_weights(
    predictions,
    k=10,
):
    """
    ML selection + equal weighting.

    This isolates the value of the ML ranking signal.
    """

    selected = top_k(
        predictions,
        k,
    )

    if selected.empty:
        return pd.Series(dtype=float)

    return equal_weight(
        selected["Ticker"]
    )


def ml_risk_weights(
    predictions,
    returns,
    k=10,
    risk_profile="moderate",
    max_weight=0.25,
):
    """
    ML ranking + risk-aware weighting.

    Expected return comes from ML prediction.
    Risk comes from recent realized volatility.
    """

    selected = top_k(
        predictions,
        k,
    )

    if selected.empty:
        return pd.Series(dtype=float)

    tickers = [
        t for t in selected["Ticker"]
        if t in returns.columns
    ]

    if not tickers:
        return pd.Series(dtype=float)

    recent = returns[
        tickers
    ].tail(60)

    volatility = recent.std()

    volatility = volatility.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    volatility = volatility.fillna(
        volatility.median()
    )

    volatility = volatility.clip(
        lower=1e-8
    )

    expected_return = (
        selected
        .set_index("Ticker")
        ["Prediction"]
        .reindex(tickers)
    )

    if risk_profile == "conservative":
        risk_aversion = 5.0

    elif risk_profile == "aggressive":
        risk_aversion = 0.5

    else:
        risk_aversion = 2.0

    score = (
        expected_return
        -
        risk_aversion * volatility
    )

    score = (
        score
        -
        score.min()
        +
        1e-8
    )

    weights = (
        score
        /
        score.sum()
    )

    return apply_max_weight(
        weights,
        max_weight,
    )


def apply_max_weight(
    weights,
    max_weight=0.20,
):
    """
    Iteratively enforce a maximum position size.
    """

    weights = (
        weights
        .copy()
        .astype(float)
    )

    if weights.empty:
        return weights

    weights = (
        weights
        /
        weights.sum()
    )

    for _ in range(100):

        oversized = (
            weights > max_weight
        )

        if not oversized.any():
            break

        excess = (
            weights[oversized]
            -
            max_weight
        ).sum()

        weights[oversized] = max_weight

        remaining = (
            ~oversized
        )

        if not remaining.any():
            break

        base = weights[remaining]

        if base.sum() <= 0:
            break

        weights[remaining] = (
            base
            /
            base.sum()
            *
            (
                1.0
                -
                weights[oversized].sum()
            )
        )

    return (
        weights
        /
        weights.sum()
    )
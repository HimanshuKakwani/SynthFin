import numpy as np
import pandas as pd

from scipy.optimize import minimize


def turnover(previous_weights, new_weights):
    """
    L1 portfolio turnover.

    T = sum_i |w_new_i - w_old_i|

    If previous_weights is None, the portfolio
    is assumed to be initially funded from cash,
    giving turnover = 1.0.
    """

    if previous_weights is None:
        return 1.0

    universe = sorted(
        set(previous_weights.index)
        | set(new_weights.index)
    )

    old = (
        previous_weights
        .reindex(universe, fill_value=0.0)
        .astype(float)
    )

    new = (
        new_weights
        .reindex(universe, fill_value=0.0)
        .astype(float)
    )

    return float(
        np.abs(
            new.values - old.values
        ).sum()
    )


def _build_inputs(
    predictions,
    historical_returns,
    previous_weights,
    k,
):
    """
    Build a FIXED investment universe.

    The full historical-return universe is retained.
    ML predictions determine expected returns but
    do not change the optimization dimension.
    """

    universe = [
        ticker
        for ticker in historical_returns.columns
    ]

    if not universe:
        return None

    recent = (
        historical_returns[universe]
        .tail(60)
        .copy()
    )

    # Require enough historical observations.
    valid = [
        ticker
        for ticker in universe
        if recent[ticker].notna().sum() >= 20
    ]

    if not valid:
        return None

    recent = recent[valid]

    # --------------------------------------------------
    # ML expected returns
    # --------------------------------------------------

    pred = (
        predictions
        .dropna(
            subset=["Prediction"]
        )
        .sort_values(
            "Prediction",
            ascending=False,
        )
        .copy()
    )

    pred = (
        pred
        .drop_duplicates(
            subset=["Ticker"]
        )
        .set_index("Ticker")
    )

    # Start every asset at zero expected return.
    expected_returns = pd.Series(
        0.0,
        index=valid,
        dtype=float,
    )

    # Only top-k receive the ML signal.
    selected = (
        pred
        .loc[
            pred.index.intersection(valid)
        ]
        .head(k)
    )

    if not selected.empty:

        expected_returns.loc[
            selected.index
        ] = (
            selected["Prediction"]
            .astype(float)
        )

    # --------------------------------------------------
    # Covariance
    # --------------------------------------------------

    covariance = (
        recent
        .cov()
        .fillna(0.0)
        .values
    )

    covariance = (
        covariance
        +
        np.eye(
            len(valid)
        ) * 1e-6
    )

    # --------------------------------------------------
    # Previous portfolio
    # --------------------------------------------------

    if previous_weights is None:

        previous = pd.Series(
            0.0,
            index=valid,
            dtype=float,
        )

    else:

        previous = (
            previous_weights
            .reindex(
                valid,
                fill_value=0.0,
            )
            .astype(float)
        )

    return (
        valid,
        expected_returns.values,
        covariance,
        previous.values,
    )


def optimize_portfolio(
    predictions,
    historical_returns,
    previous_weights=None,
    k=10,
    risk_aversion=2.0,
    turnover_penalty=1.0,
    max_weight=0.20,
    max_turnover=None,
):
    """
    Fixed-universe turnover-aware optimizer.

    Objective:

        minimize

        - expected return
        + risk_aversion * variance
        + turnover_penalty * L1 turnover

    Subject to:

        sum(w) = 1
        0 <= w_i <= max_weight

    Optional:

        turnover <= max_turnover

    IMPORTANT:
    The function never silently returns an infeasible
    solution. If the requested constraints cannot be
    satisfied, an empty Series is returned.
    """

    prepared = _build_inputs(
        predictions,
        historical_returns,
        previous_weights,
        k,
    )

    if prepared is None:
        return pd.Series(
            dtype=float
        )

    (
        tickers,
        expected_returns,
        covariance,
        previous,
    ) = prepared

    n = len(tickers)

    # --------------------------------------------------
    # Initial portfolio
    # --------------------------------------------------

    if previous_weights is None:

        # Equal-weight starting portfolio.
        x0 = np.ones(n) / n

    else:

        x0 = previous.copy()

        if x0.sum() <= 1e-12:

            x0 = np.ones(n) / n

        else:

            x0 = (
                x0
                /
                x0.sum()
            )

    # --------------------------------------------------
    # Objective
    # --------------------------------------------------

    def objective(weights):

        expected_return = np.dot(
            expected_returns,
            weights,
        )

        variance = float(
            weights.T
            @ covariance
            @ weights
        )

        trade = np.abs(
            weights - previous
        ).sum()

        return (
            -expected_return
            +
            risk_aversion
            * variance
            +
            turnover_penalty
            * trade
        )

    # --------------------------------------------------
    # Constraints
    # --------------------------------------------------

    constraints = [
        {
            "type": "eq",
            "fun": lambda w:
                np.sum(w) - 1.0,
        }
    ]

    # Do NOT impose a turnover constraint on the
    # initial investment from cash.
    if (
        max_turnover is not None
        and previous_weights is not None
    ):

        constraints.append({
            "type": "ineq",
            "fun": lambda w:
                max_turnover
                -
                np.abs(
                    w - previous
                ).sum(),
        })

    bounds = [
        (
            0.0,
            max_weight,
        )
        for _ in range(n)
    ]

    # --------------------------------------------------
    # Feasibility check for max weight
    # --------------------------------------------------

    if (
        n * max_weight
        < 1.0 - 1e-10
    ):

        raise ValueError(
            "Infeasible max_weight: "
            f"{n} assets × "
            f"{max_weight} < 1."
        )

    # --------------------------------------------------
    # Optimization
    # --------------------------------------------------

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 1000,
            "ftol": 1e-10,
            "disp": False,
        },
    )

    if not result.success:

        # IMPORTANT:
        # Never silently return an infeasible portfolio.
        return pd.Series(
            dtype=float
        )

    weights = pd.Series(
        np.asarray(
            result.x,
            dtype=float,
        ),
        index=tickers,
    )

    # --------------------------------------------------
    # Numerical cleanup
    # --------------------------------------------------

    weights[
        weights.abs() < 1e-10
    ] = 0.0

    total = weights.sum()

    if total <= 0:

        return pd.Series(
            dtype=float
        )

    weights = (
        weights
        /
        total
    )

    # --------------------------------------------------
    # Explicit validation
    # --------------------------------------------------

    if not np.isclose(
        weights.sum(),
        1.0,
        atol=1e-6,
    ):

        return pd.Series(
            dtype=float
        )

    if (
        weights < -1e-7
    ).any():

        return pd.Series(
            dtype=float
        )

    if (
        weights > max_weight + 1e-6
    ).any():

        return pd.Series(
            dtype=float
        )

    actual_turnover = turnover(
        previous_weights,
        weights,
    )

    if (
        max_turnover is not None
        and previous_weights is not None
        and actual_turnover
        > max_turnover + 1e-5
    ):

        return pd.Series(
            dtype=float
        )

    return weights


def risk_aversion_for_profile(
    risk_profile,
):
    """
    Investor risk-aversion mapping.
    """

    if risk_profile == "conservative":

        return 5.0

    if risk_profile == "aggressive":

        return 0.5

    return 2.0
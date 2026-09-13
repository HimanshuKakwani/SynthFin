import numpy as np
import pandas as pd
from scipy.optimize import minimize


def turnover(previous_weights, new_weights):
    """L1 turnover. Initial investment from cash is turnover=1."""
    if previous_weights is None:
        return 1.0
    universe = sorted(set(previous_weights.index) | set(new_weights.index))
    old = previous_weights.reindex(universe, fill_value=0.0).astype(float)
    new = new_weights.reindex(universe, fill_value=0.0).astype(float)
    return float(np.abs(new.values - old.values).sum())


def _build_inputs(
    predictions,
    historical_returns,
    previous_weights,
    k,
    use_prediction_signal=True,
):
    """Build the fixed optimization universe.

    If use_prediction_signal=False, the optimizer is genuinely signal-free:
    no prediction values are used anywhere in the expected-return vector.
    """
    universe = list(historical_returns.columns)
    if not universe:
        return None

    recent = historical_returns[universe].tail(60).copy()
    valid = [t for t in universe if recent[t].notna().sum() >= 20]
    if not valid:
        return None
    recent = recent[valid]

    expected_returns = pd.Series(0.0, index=valid, dtype=float)

    if use_prediction_signal:
        if predictions is None:
            raise ValueError("predictions are required when use_prediction_signal=True")
        pred = (
            predictions.dropna(subset=["Prediction"])
            .sort_values("Prediction", ascending=False)
            .drop_duplicates(subset=["Ticker"])
            .set_index("Ticker")
        )
        selected = pred.loc[pred.index.intersection(valid)].head(k)
        if not selected.empty:
            expected_returns.loc[selected.index] = selected["Prediction"].astype(float)

    covariance = recent.cov().fillna(0.0).values
    covariance = covariance + np.eye(len(valid)) * 1e-6

    if previous_weights is None:
        previous = pd.Series(0.0, index=valid, dtype=float)
    else:
        previous = previous_weights.reindex(valid, fill_value=0.0).astype(float)

    return valid, expected_returns.values, covariance, previous.values


def optimize_portfolio(
    predictions,
    historical_returns,
    previous_weights=None,
    k=10,
    risk_aversion=2.0,
    turnover_penalty=1.0,
    max_weight=0.20,
    max_turnover=None,
    use_prediction_signal=True,
):
    """Optimize a long-only portfolio with explicit signal control."""
    prepared = _build_inputs(
        predictions,
        historical_returns,
        previous_weights,
        k,
        use_prediction_signal=use_prediction_signal,
    )
    if prepared is None:
        return pd.Series(dtype=float)

    tickers, expected_returns, covariance, previous = prepared
    n = len(tickers)

    if previous_weights is None:
        x0 = np.ones(n) / n
    else:
        x0 = previous.copy()
        x0 = x0 / x0.sum() if x0.sum() > 1e-12 else np.ones(n) / n

    def objective(weights):
        expected_return = np.dot(expected_returns, weights)
        variance = float(weights.T @ covariance @ weights)
        trade = np.abs(weights - previous).sum()
        return -expected_return + risk_aversion * variance + turnover_penalty * trade

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if max_turnover is not None and previous_weights is not None:
        constraints.append({
            "type": "ineq",
            "fun": lambda w: max_turnover - np.abs(w - previous).sum(),
        })

    if n * max_weight < 1.0 - 1e-10:
        raise ValueError(f"Infeasible max_weight: {n} assets x {max_weight} < 1")

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=[(0.0, max_weight) for _ in range(n)],
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10, "disp": False},
    )

    if not result.success:
        return pd.Series(dtype=float)

    weights = pd.Series(np.asarray(result.x, dtype=float), index=tickers)
    weights[weights.abs() < 1e-10] = 0.0
    total = weights.sum()
    if total <= 0:
        return pd.Series(dtype=float)
    weights = weights / total

    if not np.isclose(weights.sum(), 1.0, atol=1e-6):
        return pd.Series(dtype=float)
    if (weights < -1e-7).any() or (weights > max_weight + 1e-6).any():
        return pd.Series(dtype=float)

    actual_turnover = turnover(previous_weights, weights)
    if (
        max_turnover is not None
        and previous_weights is not None
        and actual_turnover > max_turnover + 1e-5
    ):
        return pd.Series(dtype=float)

    return weights


def risk_aversion_for_profile(risk_profile):
    if risk_profile == "conservative":
        return 5.0
    if risk_profile == "aggressive":
        return 0.5
    return 2.0

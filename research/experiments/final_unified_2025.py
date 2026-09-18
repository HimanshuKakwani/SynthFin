"""Corrected unified 2025 experiment for the ICCSDI revision.

Key corrections:
1. BuyHold is prediction-independent and never rebalances.
2. EqualWeight is prediction-independent and uses the full evaluation universe.
3. RiskOnly uses historical volatility only; no ML prediction enters selection.
4. XGB EqualWeight uses XGB Top-10 ranking + equal weights.
5. XGB Risk uses XGB Top-10 ranking + risk-aware optimization.
6. XGB Turnover uses XGB + risk + turnover penalty.
7. Transaction cost is configurable for sensitivity analysis.
"""
from pathlib import Path
import os
import numpy as np
import pandas as pd

from research.portfolio.turnover_optimizer import (
    optimize_portfolio,
    turnover,
    risk_aversion_for_profile,
    risk_only_weights,
    ml_equal_weights,
)

PREDICTIONS = Path("results/tables/walk_forward_predictions.csv")
DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("results/tables")
HOLDING_DAYS = 5
INITIAL_CAPITAL = 1_000_000
COST_BPS = float(os.getenv("SYNTHFIN_COST_BPS", "10"))

SELECTED_GAMMA = {
    "conservative": 0.02,
    "moderate": 0.02,
    "aggressive": 0.02,
}


def load_prices(tickers):
    frames = []
    for ticker in tickers:
        path = DATA_DIR / (ticker.replace("/", "_") + ".csv")
        if not path.exists():
            continue
        df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
        if "Close" in df.columns:
            frames.append(df["Close"].rename(ticker))
    if not frames:
        raise FileNotFoundError("No processed price files found.")
    return pd.concat(frames, axis=1).sort_index()


def calculate_metrics(history):
    r = history["NetReturn"]
    equity = history["PortfolioValue"]
    periods = 252 / HOLDING_DAYS
    years = len(r) / periods
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    vol = r.std(ddof=1) * np.sqrt(periods)
    sharpe = r.mean() / r.std(ddof=1) * np.sqrt(periods) if r.std(ddof=1) > 1e-12 else np.nan
    downside = r[r < 0]
    sortino = np.nan
    if len(downside) > 1 and downside.std(ddof=1) > 0:
        sortino = r.mean() * periods / (downside.std(ddof=1) * np.sqrt(periods))
    dd = equity / equity.cummax() - 1
    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MaxDrawdown": dd.min(),
        "TotalReturn": equity.iloc[-1] / equity.iloc[0] - 1,
        "AverageTurnover": history["Turnover"].mean(),
        "TotalTransactionCosts": history["TransactionCost"].sum(),
        "Observations": len(history),
    }


def _equal_weights(tickers):
    tickers = list(tickers)
    if not tickers:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(tickers), index=tickers, dtype=float)


def _xgb_top10(current, k=10):
    selected = (
        current.dropna(subset=["Prediction"])
        .sort_values("Prediction", ascending=False)
        .drop_duplicates("Ticker")
        .head(k)["Ticker"]
        .tolist()
    )
    return selected


def make_portfolio(current, historical, previous_weights, strategy, profile=None, universe=None):
    all_tickers = [t for t in (universe or historical.columns) if t in historical.columns]

    if strategy == "BuyHold":
        # Pure benchmark: no prediction, no subsequent rebalancing.
        if previous_weights is not None:
            return previous_weights.copy()
        return _equal_weights(all_tickers)

    if strategy == "EqualWeight":
        # Pure benchmark: full universe, no prediction ranking.
        return _equal_weights(all_tickers)

    if strategy == "RiskOnly":
        # Pure risk benchmark: lowest-volatility Top-10 + inverse-vol weighting.
        return risk_only_weights(all_tickers, historical, k=10, max_weight=0.20)

    if strategy == "XGB_EqualWeight":
        return _equal_weights(_xgb_top10(current, 10))

    if strategy == "XGB_Risk":
        return optimize_portfolio(
            predictions=current,
            historical_returns=historical,
            previous_weights=previous_weights,
            k=10,
            risk_aversion=risk_aversion_for_profile(profile),
            turnover_penalty=0.0,
            max_weight=0.20,
            max_turnover=None,
            use_prediction_signal=True,
        )

    if strategy == "XGB_TurnoverAware":
        return optimize_portfolio(
            predictions=current,
            historical_returns=historical,
            previous_weights=previous_weights,
            k=10,
            risk_aversion=risk_aversion_for_profile(profile),
            turnover_penalty=SELECTED_GAMMA[profile],
            max_weight=0.20,
            max_turnover=None,
            use_prediction_signal=True,
        )

    raise ValueError(strategy)


def run_strategy(predictions, returns, strategy, profile=None):
    dates = np.array(sorted(pd.to_datetime(predictions["Date"]).unique()))
    rebalance_dates = dates[::HOLDING_DAYS]
    universe = list(returns.columns)
    previous_weights = None
    capital = INITIAL_CAPITAL
    rows = []

    for date in rebalance_dates:
        idx = np.where(dates == date)[0][0]
        future_dates = dates[idx + 1: idx + 1 + HOLDING_DAYS]
        if len(future_dates) < HOLDING_DAYS:
            continue

        current = predictions[predictions["Date"] == date].copy()
        historical = returns.loc[returns.index < date]
        if current.empty or historical.empty:
            continue

        weights = make_portfolio(current, historical, previous_weights, strategy, profile, universe)
        if weights is None or weights.empty:
            continue
        available = [t for t in weights.index if t in returns.columns]
        weights = weights[available]
        if weights.empty:
            continue
        weights = weights / weights.sum()

        period = returns.loc[future_dates, available]
        gross_return = (1 + (period * weights).sum(axis=1)).prod() - 1
        current_turnover = turnover(previous_weights, weights)
        transaction_cost = current_turnover * COST_BPS / 10000.0
        net_return = gross_return - transaction_cost
        capital *= 1 + net_return

        rows.append({
            "Date": date,
            "PortfolioValue": capital,
            "GrossReturn": gross_return,
            "TransactionCost": transaction_cost,
            "NetReturn": net_return,
            "Turnover": current_turnover,
            "NStocks": len(weights),
            "Strategy": strategy,
            "RiskProfile": profile or "",
            "CostBps": COST_BPS,
        })
        previous_weights = weights.copy()

    return pd.DataFrame(rows)


def main():
    predictions = pd.read_csv(PREDICTIONS, parse_dates=["Date"])
    predictions = predictions[predictions["Date"].dt.year == 2025].copy()
    tickers = sorted(predictions["Ticker"].unique())
    prices = load_prices(tickers)
    returns = prices.pct_change()

    configurations = [
        ("BuyHold", None),
        ("EqualWeight", None),
        ("RiskOnly", None),
        ("XGB_EqualWeight", None),
        ("XGB_Risk", "conservative"),
        ("XGB_Risk", "moderate"),
        ("XGB_Risk", "aggressive"),
        ("XGB_TurnoverAware", "conservative"),
        ("XGB_TurnoverAware", "moderate"),
        ("XGB_TurnoverAware", "aggressive"),
    ]

    all_histories, summaries = [], []
    for strategy, profile in configurations:
        label = strategy + (f" | {profile}" if profile else "")
        print(f"Running {label} | cost={COST_BPS:.1f} bps")
        history = run_strategy(predictions, returns, strategy, profile)
        if history.empty:
            print("  ERROR: no observations")
            continue
        all_histories.append(history)
        m = calculate_metrics(history)
        m.update({
            "Strategy": strategy,
            "RiskProfile": profile or "",
            "Gamma": SELECTED_GAMMA[profile] if strategy == "XGB_TurnoverAware" else "",
            "CostBps": COST_BPS,
        })
        summaries.append(m)
        print(f"  Observations: {len(history)}")

    summary = pd.DataFrame(summaries)
    history = pd.concat(all_histories, ignore_index=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_DIR / "REVISION_UNIFIED_2025_RESULTS.csv", index=False)
    history.to_csv(OUTPUT_DIR / "REVISION_UNIFIED_2025_HISTORY.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

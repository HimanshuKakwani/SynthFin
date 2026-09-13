from pathlib import Path
import numpy as np
import pandas as pd

from research.portfolio.turnover_optimizer import (
    optimize_portfolio,
    turnover,
    risk_aversion_for_profile,
)

PREDICTIONS = "results/tables/walk_forward_predictions.csv"
DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("results/tables")

HOLDING_DAYS = 5
COST_BPS = 10
INITIAL_CAPITAL = 1_000_000

# Frozen from the earlier validation.
SELECTED_GAMMA = {
    "conservative": 0.01,
    "moderate": 0.01,
    "aggressive": 0.005,
}


def load_prices(tickers):
    frames = []

    for ticker in tickers:
        path = DATA_DIR / (ticker.replace("/", "_") + ".csv")

        if not path.exists():
            continue

        df = pd.read_csv(
            path,
            index_col=0,
            parse_dates=True,
        ).sort_index()

        if "Close" not in df.columns:
            continue

        frames.append(df["Close"].rename(ticker))

    if not frames:
        raise FileNotFoundError("No processed price files found.")

    return pd.concat(frames, axis=1).sort_index()


def calculate_metrics(history):
    returns = history["NetReturn"]
    equity = history["PortfolioValue"]

    periods = 252 / HOLDING_DAYS
    years = len(returns) / periods

    cagr = (
        equity.iloc[-1] / equity.iloc[0]
    ) ** (1 / years) - 1

    volatility = (
        returns.std(ddof=1) * np.sqrt(periods)
    )

    sharpe = (
        returns.mean()
        / returns.std(ddof=1)
        * np.sqrt(periods)
        if returns.std(ddof=1) > 1e-12
        else np.nan
    )

    downside = returns[returns < 0]

    if len(downside) > 1 and downside.std(ddof=1) > 0:
        downside_std = (
            downside.std(ddof=1) * np.sqrt(periods)
        )
        sortino = (
            returns.mean() * periods / downside_std
        )
    else:
        sortino = np.nan

    drawdown = (
        equity / equity.cummax() - 1
    )

    return {
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MaxDrawdown": drawdown.min(),
        "TotalReturn": (
            equity.iloc[-1] / equity.iloc[0] - 1
        ),
        "AverageTurnover": history["Turnover"].mean(),
        "TotalTransactionCosts": history[
            "TransactionCost"
        ].sum(),
        "Observations": len(history),
    }


def make_portfolio(
    current,
    historical,
    previous_weights,
    strategy,
    profile=None,
):
    if strategy == "BuyHold":
        if previous_weights is not None:
            return previous_weights.copy()

        tickers = (
            current
            .sort_values("Prediction", ascending=False)
            ["Ticker"]
            .head(20)
            .tolist()
        )

        return pd.Series(
            1 / len(tickers),
            index=tickers,
        )

    if strategy == "EqualWeight":
        tickers = (
            current
            .sort_values("Prediction", ascending=False)
            ["Ticker"]
            .head(10)
            .tolist()
        )

        return pd.Series(
            1 / len(tickers),
            index=tickers,
        )

    if strategy == "RiskOnly":
        return optimize_portfolio(
            predictions=current,
            historical_returns=historical,
            previous_weights=previous_weights,
            k=10,
            risk_aversion=2.0,
            turnover_penalty=0.0,
            max_weight=0.20,
            max_turnover=None,
        )

    if strategy == "XGB_EqualWeight":
        tickers = (
            current
            .sort_values("Prediction", ascending=False)
            ["Ticker"]
            .head(10)
            .tolist()
        )

        return pd.Series(
            1 / len(tickers),
            index=tickers,
        )

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
        )

    raise ValueError(strategy)


def run_strategy(
    predictions,
    returns,
    strategy,
    profile=None,
):
    dates = np.array(
        sorted(predictions["Date"].unique())
    )

    # EXACT SAME schedule for every strategy.
    rebalance_dates = dates[::HOLDING_DAYS]

    previous_weights = None
    capital = INITIAL_CAPITAL
    rows = []

    for date in rebalance_dates:
        idx = np.where(dates == date)[0][0]

        future_dates = dates[
            idx + 1:
            idx + 1 + HOLDING_DAYS
        ]

        if len(future_dates) < HOLDING_DAYS:
            continue

        current = predictions[
            predictions["Date"] == date
        ].copy()

        historical = returns.loc[
            returns.index < date
        ]

        if current.empty or historical.empty:
            continue

        weights = make_portfolio(
            current=current,
            historical=historical,
            previous_weights=previous_weights,
            strategy=strategy,
            profile=profile,
        )

        if weights is None or weights.empty:
            continue

        available = [
            ticker
            for ticker in weights.index
            if ticker in returns.columns
        ]

        weights = weights[available]

        if weights.empty:
            continue

        # Normalize after restricting to available assets.
        weights = weights / weights.sum()

        period = returns.loc[
            future_dates,
            available,
        ]

        portfolio_daily = (
            period * weights
        ).sum(axis=1)

        gross_return = (
            1 + portfolio_daily
        ).prod() - 1

        current_turnover = turnover(
            previous_weights,
            weights,
        )

        transaction_cost = (
            current_turnover
            * COST_BPS
            / 10000
        )

        net_return = (
            gross_return
            - transaction_cost
        )

        capital *= (
            1 + net_return
        )

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
        })

        previous_weights = weights.copy()

    return pd.DataFrame(rows)


def main():
    predictions = pd.read_csv(
        PREDICTIONS,
        parse_dates=["Date"],
    )

    predictions = predictions[
        predictions["Date"].dt.year == 2025
    ].copy()

    tickers = sorted(
        predictions["Ticker"].unique()
    )

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

    all_histories = []
    summaries = []

    for strategy, profile in configurations:
        label = (
            f"{strategy}"
            + (
                f" | {profile}"
                if profile
                else ""
            )
        )

        print(f"Running {label}")

        history = run_strategy(
            predictions,
            returns,
            strategy,
            profile,
        )

        if history.empty:
            print("  ERROR: no observations")
            continue

        all_histories.append(history)

        m = calculate_metrics(history)

        m["Strategy"] = strategy
        m["RiskProfile"] = profile or ""

        if strategy == "XGB_TurnoverAware":
            m["Gamma"] = SELECTED_GAMMA[profile]
        else:
            m["Gamma"] = ""

        summaries.append(m)

        print(
            f"  Observations: {len(history)}"
        )

    summary = pd.DataFrame(summaries)

    summary = summary[
        [
            "Strategy",
            "RiskProfile",
            "Gamma",
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
    ]

    history = pd.concat(
        all_histories,
        ignore_index=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        OUTPUT_DIR
        / "FINAL_UNIFIED_2025_RESULTS.csv"
    )

    history_path = (
        OUTPUT_DIR
        / "FINAL_UNIFIED_2025_HISTORY.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    history.to_csv(
        history_path,
        index=False,
    )

    print()
    print("=" * 110)
    print("FINAL UNIFIED 2025 — SAME 5-DAY METHODOLOGY")
    print("=" * 110)
    print(summary.to_string(index=False))

    print()
    print("Saved:")
    print(summary_path)
    print(history_path)


if __name__ == "__main__":
    main()

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

COST_BPS = 10
HOLDING_DAYS = 5
INITIAL_CAPITAL = 1_000_000

PROFILES = {
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
        return pd.DataFrame()

    return pd.concat(frames, axis=1).sort_index()


def metrics(history):
    returns = history["NetReturn"]
    equity = history["PortfolioValue"]

    periods = 252 / HOLDING_DAYS

    years = len(returns) / periods

    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1

    vol = returns.std(ddof=1) * np.sqrt(periods)

    sharpe = (
        returns.mean() / returns.std(ddof=1) * np.sqrt(periods)
        if returns.std(ddof=1) > 1e-12
        else np.nan
    )

    downside = returns[returns < 0]

    if len(downside) > 1 and downside.std(ddof=1) > 0:
        downside_std = downside.std(ddof=1) * np.sqrt(periods)
        sortino = returns.mean() * periods / downside_std
    else:
        sortino = np.nan

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


def run_profile(predictions, returns, profile, gamma):
    dates = np.array(sorted(predictions["Date"].unique()))

    # IMPORTANT:
    # Same 5-day schedule as FINAL_PORTFOLIO_RESULTS.
    rebalance_dates = dates[::HOLDING_DAYS]

    previous_weights = None
    capital = INITIAL_CAPITAL
    rows = []

    risk_aversion = risk_aversion_for_profile(profile)

    for date in rebalance_dates:
        idx = np.where(dates == date)[0][0]

        future_dates = dates[
            idx + 1: idx + 1 + HOLDING_DAYS
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

        weights = optimize_portfolio(
            predictions=current,
            historical_returns=historical,
            previous_weights=previous_weights,
            k=10,
            risk_aversion=risk_aversion,
            turnover_penalty=gamma,
            max_weight=0.20,
            max_turnover=None,
        )

        if weights.empty:
            continue

        available = [
            t for t in weights.index
            if t in returns.columns
        ]

        weights = weights[available]

        period = returns.loc[
            future_dates,
            available,
        ]

        portfolio_daily = (
            period * weights
        ).sum(axis=1)

        gross = (
            1 + portfolio_daily
        ).prod() - 1

        t = turnover(
            previous_weights,
            weights,
        )

        cost = t * COST_BPS / 10000

        net = gross - cost

        capital *= 1 + net

        rows.append({
            "Date": date,
            "PortfolioValue": capital,
            "GrossReturn": gross,
            "TransactionCost": cost,
            "NetReturn": net,
            "Turnover": t,
            "NStocks": len(weights),
            "RiskProfile": profile,
            "Gamma": gamma,
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

    results = []

    for profile, gamma in PROFILES.items():
        print(
            f"Running {profile:<12} "
            f"gamma={gamma}"
        )

        history = run_profile(
            predictions,
            returns,
            profile,
            gamma,
        )

        if history.empty:
            print("  ERROR: no observations")
            continue

        m = metrics(history)

        m["Strategy"] = "XGB_TurnoverAware_5Day"
        m["RiskProfile"] = profile
        m["Gamma"] = gamma

        results.append(m)

        print(
            f"  Observations: {len(history)}"
        )

    result = pd.DataFrame(results)

    columns = [
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

    result = result[columns]

    output = Path(
        "results/tables/"
        "FINAL_2025_TURNOVER_5DAY_RESULTS.csv"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        output,
        index=False,
    )

    print()
    print("=" * 100)
    print("FINAL 2025 TURNOVER-AWARE — 5 DAY RESULTS")
    print("=" * 100)
    print(result.to_string(index=False))
    print()
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()

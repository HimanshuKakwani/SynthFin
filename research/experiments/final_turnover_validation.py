from pathlib import Path

import numpy as np
import pandas as pd

from research.portfolio.turnover_optimizer import (
    optimize_portfolio,
    turnover,
    risk_aversion_for_profile,
)


PREDICTIONS = Path(
    "results/tables/walk_forward_predictions.csv"
)

DATA_DIR = Path("data/processed")

COST_BPS = 10
HOLDING_DAYS = 5
INITIAL_CAPITAL = 1_000_000

GAMMAS = [
    0.0,
    0.001,
    0.003,
    0.005,
    0.01,
]

PROFILES = [
    "conservative",
    "moderate",
    "aggressive",
]


def load_prices(tickers):

    frames = []

    for ticker in tickers:

        path = (
            DATA_DIR
            / f"{ticker}.csv"
        )

        if not path.exists():
            continue

        df = pd.read_csv(
            path,
            index_col=0,
            parse_dates=True,
        ).sort_index()

        if "Close" not in df.columns:
            continue

        frames.append(
            df["Close"].rename(ticker)
        )

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        axis=1,
    ).sort_index()


def portfolio_history(
    predictions,
    returns,
    gamma,
    risk_profile,
    start_year,
    end_year,
    max_turnover=0.50,
    k=10,
):

    dates = sorted(
        predictions["Date"].unique()
    )

    dates = [
        d for d in dates
        if start_year <= d.year <= end_year
    ]

    previous_weights = None
    capital = INITIAL_CAPITAL
    rows = []

    risk_aversion = (
        risk_aversion_for_profile(
            risk_profile
        )
    )

    for date in dates:

        current = predictions[
            predictions["Date"] == date
        ].copy()

        if current.empty:
            continue

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
            t for t in weights.index
            if t in returns.columns
        ]

        weights = weights[
            available
        ]

        # Find the next HOLDING_DAYS
        # trading observations after
        # the prediction date.
        future_dates = [
            d for d in returns.index
            if d > date
        ][:HOLDING_DAYS]

        if len(future_dates) < HOLDING_DAYS:
            continue

        period = returns.loc[
            future_dates,
            available,
        ]

        daily_portfolio = (
            period * weights
        ).sum(axis=1)

        gross_return = (
            1.0 + daily_portfolio
        ).prod() - 1.0

        current_turnover = turnover(
            previous_weights,
            weights,
        )

        transaction_cost = (
            current_turnover
            * COST_BPS
            / 10000.0
        )

        net_return = (
            gross_return
            - transaction_cost
        )

        capital *= (
            1.0 + net_return
        )

        rows.append({
            "Date": date,
            "PortfolioValue": capital,
            "GrossReturn": gross_return,
            "TransactionCost": transaction_cost,
            "NetReturn": net_return,
            "Turnover": current_turnover,
            "Gamma": gamma,
            "RiskProfile": risk_profile,
        })

        previous_weights = weights.copy()

    return pd.DataFrame(rows)


def metrics(history):

    if history.empty:
        return {}

    r = history["NetReturn"]

    equity = history[
        "PortfolioValue"
    ]

    periods_per_year = (
        252 / HOLDING_DAYS
    )

    years = (
        len(r)
        / periods_per_year
    )

    if years <= 0:
        return {}

    cagr = (
        equity.iloc[-1]
        / equity.iloc[0]
    ) ** (
        1 / years
    ) - 1

    volatility = (
        r.std(ddof=1)
        * np.sqrt(periods_per_year)
    )

    sharpe = np.nan

    if r.std(ddof=1) > 1e-12:
        sharpe = (
            r.mean()
            / r.std(ddof=1)
        ) * np.sqrt(
            periods_per_year
        )

    downside = r[r < 0]

    sortino = np.nan

    if len(downside) > 1:

        downside_std = (
            downside.std(ddof=1)
            * np.sqrt(periods_per_year)
        )

        if downside_std > 1e-12:

            sortino = (
                r.mean()
                * periods_per_year
                / downside_std
            )

    running_max = equity.cummax()

    drawdown = (
        equity / running_max
        - 1
    )

    return {
        "CAGR": cagr,
        "Volatility": volatility,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MaxDrawdown": drawdown.min(),
        "TotalReturn": (
            equity.iloc[-1]
            / equity.iloc[0]
            - 1
        ),
        "AverageTurnover":
            history["Turnover"].mean(),
        "TotalTransactionCosts":
            history[
                "TransactionCost"
            ].sum(),
        "Observations": len(history),
    }


def main():

    predictions = pd.read_csv(
        PREDICTIONS,
        parse_dates=["Date"],
    )

    predictions = predictions.sort_values(
        ["Date", "Ticker"]
    )

    tickers = sorted(
        predictions["Ticker"].unique()
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

    # ==================================================
    # STEP 1
    # DEVELOPMENT / VALIDATION
    #
    # 2022-2024 only
    # ==================================================

    validation_rows = []

    print("=" * 90)
    print("STEP 1: GAMMA VALIDATION")
    print("=" * 90)

    for profile in PROFILES:

        for gamma in GAMMAS:

            print(
                f"Testing "
                f"{profile:<12} "
                f"gamma={gamma}"
            )

            history = portfolio_history(
                predictions,
                returns,
                gamma=gamma,
                risk_profile=profile,
                start_year=2022,
                end_year=2024,
                max_turnover=0.50,
                k=10,
            )

            if history.empty:
                continue

            m = metrics(history)

            validation_rows.append({
                "RiskProfile": profile,
                "Gamma": gamma,
                **m,
            })

    validation = pd.DataFrame(
        validation_rows
    )

    validation.to_csv(
        output
        / "gamma_validation_results.csv",
        index=False,
    )

    print("\n")
    print("=" * 90)
    print("GAMMA VALIDATION RESULTS")
    print("=" * 90)

    print(
        validation.to_string(
            index=False
        )
    )

    # ==================================================
    # STEP 2
    # SELECT GAMMA
    #
    # Primary criterion: Sharpe
    # Secondary: lower turnover
    # ==================================================

    selected = {}

    for profile in PROFILES:

        subset = validation[
            validation["RiskProfile"]
            == profile
        ].copy()

        if subset.empty:
            continue

        subset = subset.sort_values(
            [
                "Sharpe",
                "AverageTurnover",
            ],
            ascending=[
                False,
                True,
            ],
        )

        best = subset.iloc[0]

        selected[profile] = float(
            best["Gamma"]
        )

    selected_df = pd.DataFrame([
        {
            "RiskProfile": profile,
            "SelectedGamma": gamma,
        }
        for profile, gamma
        in selected.items()
    ])

    selected_df.to_csv(
        output
        / "selected_gamma.csv",
        index=False,
    )

    print("\n")
    print("=" * 90)
    print("SELECTED GAMMA")
    print("=" * 90)

    print(
        selected_df.to_string(
            index=False
        )
    )

    # ==================================================
    # STEP 3
    # FINAL 2025 TEST
    #
    # Gamma is now frozen.
    # 2025 is NOT used for selection.
    # ==================================================

    final_rows = []

    print("\n")
    print("=" * 90)
    print("STEP 2: FINAL 2025 OUT-OF-SAMPLE TEST")
    print("=" * 90)

    for profile in PROFILES:

        if profile not in selected:
            continue

        gamma = selected[
            profile
        ]

        print(
            f"Testing "
            f"{profile:<12} "
            f"gamma={gamma}"
        )

        history = portfolio_history(
            predictions,
            returns,
            gamma=gamma,
            risk_profile=profile,
            start_year=2025,
            end_year=2025,
            max_turnover=0.50,
            k=10,
        )

        if history.empty:
            continue

        m = metrics(history)

        final_rows.append({
            "Strategy":
                "XGB_TurnoverAware",
            "RiskProfile":
                profile,
            "Gamma":
                gamma,
            **m,
        })

        history.to_csv(
            output
            / (
                f"final_2025_"
                f"{profile}.csv"
            ),
            index=False,
        )

    final = pd.DataFrame(
        final_rows
    )

    final.to_csv(
        output
        / "FINAL_2025_TURNOVER_RESULTS.csv",
        index=False,
    )

    print("\n")
    print("=" * 90)
    print("FINAL 2025 RESULTS")
    print("=" * 90)

    print(
        final.to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(
        output
        / "gamma_validation_results.csv"
    )
    print(
        output
        / "selected_gamma.csv"
    )
    print(
        output
        / "FINAL_2025_TURNOVER_RESULTS.csv"
    )


if __name__ == "__main__":
    main()
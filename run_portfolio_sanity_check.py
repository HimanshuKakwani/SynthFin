from pathlib import Path

import numpy as np
import pandas as pd

from research.portfolio.turnover_optimizer import (
    optimize_portfolio,
    turnover,
)


PREDICTIONS = (
    "results/tables/"
    "walk_forward_predictions.csv"
)

DATA_DIR = Path(
    "data/processed"
)


def load_returns(tickers):

    frames = []

    for ticker in tickers:

        path = (
            DATA_DIR
            /
            f"{ticker.replace('/', '_')}.csv"
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

    prices = pd.concat(
        frames,
        axis=1,
    ).sort_index()

    return prices.pct_change()


def main():

    predictions = pd.read_csv(
        PREDICTIONS,
        parse_dates=["Date"],
    )

    tickers = sorted(
        predictions["Ticker"].unique()
    )

    returns = load_returns(
        tickers
    )

    dates = sorted(
        predictions["Date"].unique()
    )

    # Use several real rebalance dates.
    test_dates = dates[
        0:50:5
    ]

    previous = None

    print()
    print("=" * 80)
    print("PORTFOLIO SANITY CHECK")
    print("=" * 80)

    for date in test_dates:

        current = predictions[
            predictions["Date"] == date
        ].copy()

        historical = returns.loc[
            returns.index < date
        ]

        weights = optimize_portfolio(
            predictions=current,
            historical_returns=historical,
            previous_weights=previous,
            k=10,
            risk_aversion=2.0,
            turnover_penalty=0.01,
            max_weight=0.20,
            max_turnover=0.50,
        )

        if weights.empty:

            print(
                f"{date.date()} "
                "FAILED: empty portfolio"
            )

            continue

        actual_turnover = turnover(
            previous,
            weights,
        )

        weight_sum = weights.sum()
        max_position = weights.max()

        violations = []

        if not np.isclose(
            weight_sum,
            1.0,
            atol=1e-6,
        ):
            violations.append(
                f"weight_sum={weight_sum:.6f}"
            )

        if (
            weights < -1e-8
        ).any():
            violations.append(
                "negative weight"
            )

        if (
            weights > 0.20 + 1e-6
        ).any():
            violations.append(
                f"max_weight={max_position:.6f}"
            )

        # The first portfolio necessarily has
        # turnover 1.0 from cash.
        if (
            previous is not None
            and actual_turnover > 0.50 + 1e-6
        ):
            violations.append(
                f"turnover={actual_turnover:.6f}"
            )

        status = (
            "PASS"
            if not violations
            else "FAIL"
        )

        print(
            f"{date.date()} | "
            f"{status:<4} | "
            f"sum={weight_sum:.6f} | "
            f"max={max_position:.6f} | "
            f"turnover={actual_turnover:.6f}"
        )

        if violations:

            for violation in violations:

                print(
                    f"    -> {violation}"
                )

        previous = weights


if __name__ == "__main__":
    main()
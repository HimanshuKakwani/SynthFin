from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from research.data.universe import load_universe
from research.features.technical_features import (
    add_features,
    FEATURE_COLUMNS
)


DATA_DIR = Path(
    "data/processed"
)


TRAIN_START = "2018-01-01"

TEST_START = "2022-01-01"

TEST_END = "2025-12-31"

RETRAIN_YEARS = 1

FORECAST_HORIZON = 5


def load_stock(ticker):

    filename = (
        ticker.replace("/", "_")
        + ".csv"
    )

    path = DATA_DIR / filename

    if not path.exists():

        return None

    df = pd.read_csv(
        path,
        index_col=0,
        parse_dates=True
    )

    df = df.sort_index()

    df = add_features(
        df,
        horizon=FORECAST_HORIZON
    )

    return df


def train_xgboost(
    X_train,
    y_train
):

    model = XGBRegressor(

        n_estimators=300,

        max_depth=4,

        learning_rate=0.03,

        subsample=0.8,

        colsample_bytree=0.8,

        objective="reg:squarederror",

        random_state=42,

        n_jobs=4
    )

    model.fit(
        X_train,
        y_train
    )

    return model


def generate_predictions(
    train_end,
    prediction_date,
    tickers
):

    rows = []

    for ticker in tickers:

        df = load_stock(
            ticker
        )

        if df is None:
            continue

        train = df[
            df.index <= train_end
        ].copy()

        test = df[
            df.index == prediction_date
        ].copy()

        if len(train) < 250:
            continue

        if test.empty:
            continue

        train = train.dropna(
            subset=FEATURE_COLUMNS
            + ["target_return"]
        )

        test = test.dropna(
            subset=FEATURE_COLUMNS
            + ["target_return"]
        )

        if train.empty or test.empty:
            continue

        X_train = train[
            FEATURE_COLUMNS
        ]

        y_train = train[
            "target_return"
        ]

        X_test = test[
            FEATURE_COLUMNS
        ]

        scaler = StandardScaler()

        X_train_scaled = (
            scaler.fit_transform(
                X_train
            )
        )

        X_test_scaled = (
            scaler.transform(
                X_test
            )
        )

        model = train_xgboost(
            X_train_scaled,
            y_train
        )

        prediction = model.predict(
            X_test_scaled
        )[0]

        actual = test[
            "target_return"
        ].iloc[0]

        rows.append({

            "Date": prediction_date,

            "Ticker": ticker,

            "Prediction": prediction,

            "ActualReturn": actual

        })

    return pd.DataFrame(
        rows
    )


def get_rebalance_dates():

    # Use actual trading dates from the
    # processed stock data rather than
    # inventing calendar dates.

    sample = load_stock(
        "RELIANCE.NS"
    )

    if sample is None:

        raise FileNotFoundError(
            "RELIANCE.NS data not found."
        )

    dates = sample.index

    dates = dates[
        (dates >= TEST_START)
        &
        (dates <= TEST_END)
    ]

    return dates


def run_walk_forward():

    tickers = load_universe(
        max_tickers=20
    )

    rebalance_dates = (
        get_rebalance_dates()
    )

    # We don't want to train 1000+ times.
    # Retrain once per year and generate
    # predictions on the rebalance dates
    # within that year.

    results = []

    for year in sorted(
        set(
            rebalance_dates.year
        )
    ):

        train_end = (
            f"{year - 1}-12-31"
        )

        year_dates = [
            d for d in rebalance_dates
            if d.year == year
        ]

        print(
            "\n"
            + "=" * 60
        )

        print(
            f"Walk-forward year: {year}"
        )

        print(
            f"Training through: {train_end}"
        )

        print(
            "=" * 60
        )

        # For computational efficiency,
        # train once per stock per year,
        # then generate predictions for
        # all rebalance dates in that year.

        for ticker in tickers:

            df = load_stock(
                ticker
            )

            if df is None:
                continue

            train = df[
                df.index <= train_end
            ].copy()

            train = train.dropna(
                subset=FEATURE_COLUMNS
                + ["target_return"]
            )

            if len(train) < 250:
                continue

            X_train = train[
                FEATURE_COLUMNS
            ]

            y_train = train[
                "target_return"
            ]

            scaler = StandardScaler()

            X_train_scaled = (
                scaler.fit_transform(
                    X_train
                )
            )

            model = train_xgboost(
                X_train_scaled,
                y_train
            )

            for date in year_dates:

                test = df[
                    df.index == date
                ]

                if test.empty:
                    continue

                test = test.dropna(
                    subset=FEATURE_COLUMNS
                    + ["target_return"]
                )

                if test.empty:
                    continue

                X_test = test[
                    FEATURE_COLUMNS
                ]

                X_test_scaled = (
                    scaler.transform(
                        X_test
                    )
                )

                prediction = model.predict(
                    X_test_scaled
                )[0]

                actual = test[
                    "target_return"
                ].iloc[0]

                results.append({

                    "Date": date,

                    "Ticker": ticker,

                    "Prediction": prediction,

                    "ActualReturn": actual,

                    "TrainingEnd": train_end,

                    "Model": "XGBoost"

                })

            print(
                f"  {ticker}: "
                f"{len(year_dates)} dates"
            )

    result = pd.DataFrame(
        results
    )

    result = result.sort_values(
        [
            "Date",
            "Ticker"
        ]
    )

    Path(
        "results/tables"
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    result.to_csv(
        "results/tables/"
        "walk_forward_predictions.csv",
        index=False
    )

    print(
        "\nSaved:"
    )

    print(
        "results/tables/"
        "walk_forward_predictions.csv"
    )

    print(
        f"\nRows: {len(result):,}"
    )

    print(
        f"Dates: "
        f"{result['Date'].nunique()}"
    )

    print(
        f"Stocks: "
        f"{result['Ticker'].nunique()}"
    )

    return result


if __name__ == "__main__":

    run_walk_forward()
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from scipy.stats import spearmanr

from research.data.universe import load_universe
from research.features.technical_features import (
    add_features,
    FEATURE_COLUMNS,
)

from research.models.baseline import MeanReturnBaseline
from research.models.random_forest import build_model as rf_model
from research.models.xgboost_model import build_model as xgb_model
from research.models.lstm import (
    make_sequences,
    train as train_lstm,
)


DATA_DIR = Path("data/processed")

TRAIN_END = "2021-12-31"
VALIDATION_START = "2022-01-01"
VALIDATION_END = "2022-12-31"
TEST_START = "2023-01-01"

SEQUENCE_LENGTH = 20


def load_stock_data(ticker):

    filename = ticker.replace("/", "_") + ".csv"
    path = DATA_DIR / filename

    if not path.exists():
        return None

    df = pd.read_csv(
        path,
        index_col=0,
        parse_dates=True,
    )

    df = df.sort_index()

    return add_features(
        df,
        horizon=5,
    )


def standard_metrics(y_true, predictions):

    y_true = np.asarray(y_true, dtype=float)
    predictions = np.asarray(predictions, dtype=float)

    direction = (
        np.sign(predictions)
        == np.sign(y_true)
    ).mean()

    return {
        "MAE": mean_absolute_error(
            y_true,
            predictions,
        ),

        "RMSE": mean_squared_error(
            y_true,
            predictions,
        ) ** 0.5,

        "R2": r2_score(
            y_true,
            predictions,
        ),

        "DirectionalAccuracy": direction,
    }


def cross_sectional_ic(prediction_table):

    rows = []

    for date, group in prediction_table.groupby("Date"):

        group = group.dropna(
            subset=[
                "ActualReturn",
                "Prediction",
            ]
        )

        if len(group) < 3:
            continue

        if group["Prediction"].nunique() < 2:
            continue

        if group["ActualReturn"].nunique() < 2:
            continue

        ic = spearmanr(
            group["Prediction"],
            group["ActualReturn"],
        ).statistic

        if np.isfinite(ic):

            rows.append({
                "Date": date,
                "IC": ic,
                "NStocks": len(group),
            })

    return pd.DataFrame(rows)


def summarize_ic(ic_df):

    if ic_df.empty:

        return {
            "MeanIC": np.nan,
            "MedianIC": np.nan,
            "ICStd": np.nan,
            "ICIR": np.nan,
            "PositiveICRate": np.nan,
            "Observations": 0,
        }

    mean_ic = ic_df["IC"].mean()
    std_ic = ic_df["IC"].std(ddof=1)

    if std_ic > 0:

        icir = mean_ic / std_ic

    else:

        icir = np.nan

    return {
        "MeanIC": mean_ic,
        "MedianIC": ic_df["IC"].median(),
        "ICStd": std_ic,
        "ICIR": icir,
        "PositiveICRate": (
            ic_df["IC"] > 0
        ).mean(),
        "Observations": len(ic_df),
    }


def prepare_lstm_data(df):

    train = df[
        df.index <= TRAIN_END
    ].copy()

    validation = df[
        (df.index >= VALIDATION_START)
        &
        (df.index <= VALIDATION_END)
    ].copy()

    test = df[
        df.index >= TEST_START
    ].copy()

    if (
        len(train) < 200
        or
        len(validation) < SEQUENCE_LENGTH + 10
        or
        len(test) < SEQUENCE_LENGTH + 10
    ):

        return None

    X_train_raw = train[
        FEATURE_COLUMNS
    ].values

    y_train = train[
        "target_return"
    ].values

    X_validation_raw = validation[
        FEATURE_COLUMNS
    ].values

    y_validation = validation[
        "target_return"
    ].values

    X_test_raw = test[
        FEATURE_COLUMNS
    ].values

    y_test = test[
        "target_return"
    ].values

    # IMPORTANT:
    # Fit the scaler ONLY on the training period.
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train_raw
    )

    X_validation_scaled = scaler.transform(
        X_validation_raw
    )

    X_test_scaled = scaler.transform(
        X_test_raw
    )

    # Create sequences independently inside this stock.
    X_train_seq, y_train_seq = make_sequences(
        X_train_scaled,
        y_train,
        SEQUENCE_LENGTH,
    )

    # Validation needs the last training observations
    # to construct the first validation sequence.
    validation_context = np.vstack([
        X_train_scaled[
            -SEQUENCE_LENGTH:
        ],
        X_validation_scaled,
    ])

    validation_target = np.concatenate([
        y_train[
            -SEQUENCE_LENGTH:
        ],
        y_validation,
    ])

    X_validation_seq, y_validation_seq = make_sequences(
        validation_context,
        validation_target,
        SEQUENCE_LENGTH,
    )

    # Test sequences use the final validation observations
    # as historical context.
    test_context = np.vstack([
        X_validation_scaled[
            -SEQUENCE_LENGTH:
        ],
        X_test_scaled,
    ])

    test_target = np.concatenate([
        y_validation[
            -SEQUENCE_LENGTH:
        ],
        y_test,
    ])

    X_test_seq, y_test_seq = make_sequences(
        test_context,
        test_target,
        SEQUENCE_LENGTH,
    )

    return (
        X_train_seq,
        y_train_seq,
        X_validation_seq,
        y_validation_seq,
        X_test_seq,
        y_test_seq,
    )


def run_lstm_for_stock(
    ticker,
    df,
):

    prepared = prepare_lstm_data(
        df
    )

    if prepared is None:

        print(
            f"  [SKIP] LSTM insufficient data"
        )

        return None

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
    ) = prepared

    print(
        f"  Training LSTM "
        f"({len(X_train)} sequences)"
    )

    model = train_lstm(
        X_train,
        y_train,
        X_validation,
        y_validation,
        epochs=30,
        batch_size=32,
    )

    predictions = (
        model.predict(
            X_test,
            verbose=0,
        )
        .reshape(-1)
    )

    metrics = standard_metrics(
        y_test,
        predictions,
    )

    dates = df[
        df.index >= TEST_START
    ].index

    # Because the sequence construction preserves the test rows,
    # predictions correspond to the test dates.
    dates = dates[
        :len(predictions)
    ]

    observations = pd.DataFrame({
        "Date": dates,
        "Ticker": ticker,
        "Model": "LSTM",
        "ActualReturn": y_test[
            :len(dates)
        ],
        "Prediction": predictions[
            :len(dates)
        ],
    })

    metrics["Ticker"] = ticker
    metrics["Model"] = "LSTM"

    return metrics, observations


def evaluate():

    print(
        "\nLoading stock universe..."
    )

    tickers = load_universe(
        max_tickers=20
    )

    prediction_rows = []
    metric_rows = []

    # ---------------------------------------------
    # Traditional ML models
    # ---------------------------------------------

    for ticker in tickers:

        print(
            f"\nProcessing {ticker}"
        )

        df = load_stock_data(
            ticker
        )

        if df is None:

            print(
                "  [SKIP] No processed data"
            )

            continue

        train = df[
            df.index <= TRAIN_END
        ]

        test = df[
            df.index >= TEST_START
        ]

        if len(train) < 200:

            print(
                "  [SKIP] Insufficient training data"
            )

            continue

        if len(test) < 50:

            print(
                "  [SKIP] Insufficient test data"
            )

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

        y_test = test[
            "target_return"
        ]

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(
            X_train
        )

        X_test_scaled = scaler.transform(
            X_test
        )

        models = {
            "Naive": MeanReturnBaseline(),
            "RandomForest": rf_model(),
            "XGBoost": xgb_model(),
        }

        for model_name, model in models.items():

            print(
                f"  Running {model_name}"
            )

            model.fit(
                X_train_scaled,
                y_train,
            )

            predictions = model.predict(
                X_test_scaled
            )

            metrics = standard_metrics(
                y_test,
                predictions,
            )

            metrics["Ticker"] = ticker
            metrics["Model"] = model_name

            metric_rows.append(
                metrics
            )

            for date, actual, prediction in zip(
                test.index,
                y_test,
                predictions,
            ):

                prediction_rows.append({
                    "Date": date,
                    "Ticker": ticker,
                    "Model": model_name,
                    "ActualReturn": actual,
                    "Prediction": prediction,
                })

        # ---------------------------------------------
        # LSTM
        # ---------------------------------------------

        print(
            "  Running LSTM"
        )

        lstm_result = run_lstm_for_stock(
            ticker,
            df,
        )

        if lstm_result is not None:

            lstm_metrics, lstm_observations = (
                lstm_result
            )

            metric_rows.append(
                lstm_metrics
            )

            prediction_rows.extend(
                lstm_observations.to_dict(
                    "records"
                )
            )

    prediction_table = pd.DataFrame(
        prediction_rows
    )

    metric_table = pd.DataFrame(
        metric_rows
    )

    # ---------------------------------------------
    # Aggregate prediction metrics
    # ---------------------------------------------

    prediction_summary = (
        metric_table
        .groupby("Model")
        [
            [
                "MAE",
                "RMSE",
                "R2",
                "DirectionalAccuracy",
            ]
        ]
        .mean()
        .reset_index()
    )

    # ---------------------------------------------
    # Cross-sectional IC
    # ---------------------------------------------

    ic_summaries = []

    models = [
        "Naive",
        "RandomForest",
        "XGBoost",
        "LSTM",
    ]

    for model_name in models:

        model_predictions = prediction_table[
            prediction_table["Model"]
            ==
            model_name
        ]

        if model_predictions.empty:
            continue

        ic_df = cross_sectional_ic(
            model_predictions
        )

        summary = summarize_ic(
            ic_df
        )

        summary["Model"] = model_name

        ic_summaries.append(
            summary
        )

        if not ic_df.empty:

            ic_df = ic_df.copy()

            ic_df["Model"] = model_name

            Path(
                "results/tables"
            ).mkdir(
                parents=True,
                exist_ok=True,
            )

            ic_df.to_csv(
                f"results/tables/"
                f"ic_timeseries_{model_name}.csv",
                index=False,
            )

    ic_summary = pd.DataFrame(
        ic_summaries
    )

    # ---------------------------------------------
    # Final summary
    # ---------------------------------------------

    final_summary = prediction_summary.merge(
        ic_summary,
        on="Model",
        how="left",
    )

    output_dir = Path(
        "results/tables"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_table.to_csv(
        output_dir /
        "prediction_observations.csv",
        index=False,
    )

    metric_table.to_csv(
        output_dir /
        "prediction_by_stock.csv",
        index=False,
    )

    prediction_summary.to_csv(
        output_dir /
        "prediction_metrics.csv",
        index=False,
    )

    ic_summary.to_csv(
        output_dir /
        "information_coefficient.csv",
        index=False,
    )

    final_summary.to_csv(
        output_dir /
        "prediction_final_summary.csv",
        index=False,
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FINAL PREDICTION + RANKING RESULTS"
    )

    print(
        "=" * 80
    )

    print(
        final_summary.to_string(
            index=False
        )
    )

    print(
        "\nSaved results to:"
    )

    print(
        "results/tables/"
    )

    return final_summary


if __name__ == "__main__":
    evaluate()
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from research.data.universe import load_universe
from research.features.technical_features import add_features, FEATURE_COLUMNS
from research.models.baseline import MeanReturnBaseline
from research.models.random_forest import build_model as rf_model
from research.models.xgboost_model import build_model as xgb_model


OUT = Path("results/tables")


def load_processed(tickers, directory="data/processed", horizon=5):
    frames = []

    for ticker in tickers:
        path = Path(directory) / f"{ticker.replace('/', '_')}.csv"

        if path.exists():
            df = pd.read_csv(
                path,
                index_col=0,
                parse_dates=True,
            )

            features = add_features(df, horizon)
            features["ticker"] = ticker
            frames.append(features)

    if not frames:
        raise FileNotFoundError(
            "No processed data. Run download_data.py then clean_data.py."
        )

    return pd.concat(frames).sort_index()


def evaluate():
    """
    Evaluate Naive, Random Forest and XGBoost on the fixed
    train/test split and save both aggregate metrics and
    observation-level predictions.

    Training period:
        through 2021-12-31

    Test period:
        from 2023-01-01 onward

    Observation file columns:
        Date, Ticker, Model, ActualReturn, Prediction
    """

    tickers = load_universe(max_tickers=20)
    df = load_processed(tickers)

    train = df[df.index <= "2021-12-31"].copy()
    test = df[df.index >= "2023-01-01"].copy()

    Xtr = train[FEATURE_COLUMNS]
    ytr = train["target_return"]

    Xt = test[FEATURE_COLUMNS]
    yt = test["target_return"]

    scaler = StandardScaler()

    Xtr_scaled = scaler.fit_transform(Xtr)
    Xt_scaled = scaler.transform(Xt)

    models = {
        "Naive": MeanReturnBaseline(),
        "RandomForest": rf_model(),
        "XGBoost": xgb_model(),
    }

    metric_rows = []
    observation_rows = []

    # Preserve the original row index so that Date/Ticker
    # remain aligned with every prediction.
    test_meta = test[["ticker"]].copy()

    for name, model in models.items():

        model.fit(Xtr_scaled, ytr)
        predictions = model.predict(Xt_scaled)

        # Aggregate metrics
        metric_rows.append({
            "Model": name,
            "MAE": mean_absolute_error(yt, predictions),
            "RMSE": mean_squared_error(yt, predictions) ** 0.5,
            "R2": r2_score(yt, predictions),
            "DirectionalAccuracy": (
                (predictions.clip(-1, 1).round()
                 == yt.clip(-1, 1).round())
            ).mean(),
            "Observations": len(yt),
        })

        # Observation-level records required by revision_analysis.py
        for date, ticker, actual, prediction in zip(
            test.index,
            test_meta["ticker"],
            yt.to_numpy(),
            predictions,
        ):
            observation_rows.append({
                "Date": date,
                "Ticker": ticker,
                "Model": name,
                "ActualReturn": actual,
                "Prediction": prediction,
            })

    metrics = pd.DataFrame(metric_rows)
    observations = pd.DataFrame(observation_rows)

    OUT.mkdir(parents=True, exist_ok=True)

    metrics.to_csv(
        OUT / "prediction_metrics.csv",
        index=False,
    )

    observations.to_csv(
        OUT / "prediction_observations.csv",
        index=False,
    )

    print("\nPrediction metrics:")
    print(metrics.to_string(index=False))

    print("\nPrediction observations:")
    print(f"Rows: {len(observations)}")
    print(f"Dates: {observations['Date'].nunique()}")
    print(f"Tickers: {observations['Ticker'].nunique()}")
    print(f"Models: {observations['Model'].unique().tolist()}")

    print(
        f"\nSaved:\n"
        f"  {OUT / 'prediction_metrics.csv'}\n"
        f"  {OUT / 'prediction_observations.csv'}"
    )

    return metrics, observations


if __name__ == "__main__":
    evaluate()
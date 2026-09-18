"""All reviewer-requested analyses for the ICCSDI revision.

Produces:
- forecast significance tests (Naive vs XGB)
- IC confidence intervals/significance
- portfolio paired tests + bootstrap CIs
- XGB signal-to-portfolio degradation analysis
- transaction-cost sensitivity
- gamma validation on 2022-2024 only
- year-by-year robustness (2022-2025)

Run from repository root:
    python -m research.experiments.revision_analysis

The script assumes the corrected final_unified_2025.py is installed and that
results/tables/walk_forward_predictions.csv and prediction_observations.csv
already exist.
"""
from pathlib import Path
import os
import subprocess
import sys
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, ttest_1samp
import statsmodels.api as sm

import research.experiments.final_unified_2025 as unified

from research.experiments.final_unified_2025 import (
    load_prices,
    run_strategy,
    calculate_metrics,
    HOLDING_DAYS,
    INITIAL_CAPITAL,
    SELECTED_GAMMA,
)

OUT = Path("results/tables")
PRED = OUT / "walk_forward_predictions.csv"
OBS = OUT / "prediction_observations.csv"


def bootstrap_ci(values, n_bootstrap=10000, seed=42):
    x = np.asarray(pd.Series(values).dropna(), dtype=float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    samples = rng.choice(x, size=(n_bootstrap, len(x)), replace=True)
    means = samples.mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_bootstrap(a, b, n_bootstrap=10000, seed=42):
    a, b = pd.Series(a), pd.Series(b)
    joined = pd.concat([a, b], axis=1).dropna()
    d = joined.iloc[:, 0].to_numpy() - joined.iloc[:, 1].to_numpy()
    return bootstrap_ci(d, n_bootstrap=n_bootstrap, seed=seed)


def forecast_tests():
    df = pd.read_csv(OBS, parse_dates=["Date"])
    # Compute model losses at the stock level, then aggregate by date.
    # Inference is performed over dates rather than treating 20 stocks on
    # the same date as independent observations.
    df = df[df["Model"].isin(["Naive", "XGBoost"])].copy()
    pivot = df.pivot_table(index=["Date", "Ticker"], columns="Model", values=["ActualReturn", "Prediction"])
    common = pd.DataFrame({
        "y": pivot["ActualReturn"]["Naive"],
        "naive": pivot["Prediction"]["Naive"],
        "xgb": pivot["Prediction"]["XGBoost"],
    }).dropna()
    common["abs_naive"] = (common["y"] - common["naive"]).abs()
    common["abs_xgb"] = (common["y"] - common["xgb"]).abs()
    common["sq_naive"] = (common["y"] - common["naive"]) ** 2
    common["sq_xgb"] = (common["y"] - common["xgb"]) ** 2
    date_losses = common.groupby(level="Date")[["abs_naive", "abs_xgb", "sq_naive", "sq_xgb"]].mean()

    rows = []
    for metric, a, b in [
        ("AbsoluteError", date_losses["abs_naive"], date_losses["abs_xgb"]),
        ("SquaredError", date_losses["sq_naive"], date_losses["sq_xgb"]),
    ]:
        t, p = ttest_rel(a, b, nan_policy="omit")
        diff = (a - b).dropna()
        hac = sm.OLS(diff.values, np.ones((len(diff), 1))).fit(
            cov_type="HAC", cov_kwds={"maxlags": 5}
        )
        mean_diff, lo, hi = paired_bootstrap(a, b)
        rows.append({
            "Comparison": "Naive - XGBoost", "Metric": metric,
            "MeanDifference": mean_diff, "CI95Lower": lo, "CI95Upper": hi,
            "PairedT": float(t), "PValue": float(p),
            "HAC_TStat": float(hac.tvalues[0]), "HAC_PValue": float(hac.pvalues[0]),
            "Observations": len(a),
        })
    pd.DataFrame(rows).to_csv(OUT / "REVISION_FORECAST_SIGNIFICANCE.csv", index=False)

def ic_tests():
    rows = []
    for model in ["Naive", "RandomForest", "XGBoost", "LSTM"]:
        path = OUT / f"ic_timeseries_{model}.csv"
        if not path.exists():
            continue
        x = pd.read_csv(path)["IC"].dropna()
        mean, lo, hi = bootstrap_ci(x)
        t, p = ttest_1samp(x, 0.0)
        rows.append({
            "Model": model,
            "MeanIC": mean,
            "CI95Lower": lo,
            "CI95Upper": hi,
            "TStat": float(t),
            "PValue": float(p),
            "PositiveICRate": float((x > 0).mean()),
            "Observations": len(x),
        })
    pd.DataFrame(rows).to_csv(OUT / "REVISION_IC_SIGNIFICANCE.csv", index=False)


def portfolio_significance():
    history = pd.read_csv(OUT / "REVISION_UNIFIED_2025_HISTORY.csv", parse_dates=["Date"])
    history["RiskProfile"] = history["RiskProfile"].fillna("").replace("", "NONE")
    piv = history.pivot_table(index="Date", columns=["Strategy", "RiskProfile"], values="NetReturn")

    comparisons = [
        (("XGB_EqualWeight", "NONE"), ("EqualWeight", "NONE")),
        (("XGB_Risk", "moderate"), ("RiskOnly", "NONE")),
        (("XGB_TurnoverAware", "aggressive"), ("XGB_Risk", "aggressive")),
        (("XGB_Risk", "aggressive"), ("BuyHold", "NONE")),
    ]
    rows = []
    for a, b in comparisons:
        if a not in piv.columns or b not in piv.columns:
            continue
        x = piv[a].dropna()
        y = piv[b].reindex(x.index).dropna()
        common = pd.concat([x, y], axis=1).dropna()
        if common.empty:
            continue
        diff = common.iloc[:, 0] - common.iloc[:, 1]
        t, p = ttest_rel(common.iloc[:, 0], common.iloc[:, 1])
        mean, lo, hi = bootstrap_ci(diff)
        rows.append({
            "A": f"{a[0]}:{a[1]}",
            "B": f"{b[0]}:{b[1]}",
            "MeanReturnDifference": mean,
            "CI95Lower": lo,
            "CI95Upper": hi,
            "PairedT": float(t),
            "PValue": float(p),
            "Observations": len(diff),
        })
    pd.DataFrame(rows).to_csv(OUT / "REVISION_PORTFOLIO_SIGNIFICANCE.csv", index=False)


def signal_degradation():
    df = pd.read_csv(PRED, parse_dates=["Date"])
    df = df[df["Date"].dt.year == 2025].copy()
    rows = []
    for date, g in df.groupby("Date"):
        g = g.dropna(subset=["Prediction", "ActualReturn"]).sort_values("Prediction", ascending=False)
        if len(g) < 20:
            continue
        k = min(10, len(g) // 2)
        top = g.head(k)["ActualReturn"].mean()
        bottom = g.tail(k)["ActualReturn"].mean()
        universe = g["ActualReturn"].mean()
        rows.append({
            "Date": date,
            "Top10MeanReturn": top,
            "Bottom10MeanReturn": bottom,
            "TopBottomSpread": top - bottom,
            "UniverseMeanReturn": universe,
            "Top10ExcessUniverse": top - universe,
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "REVISION_XGB_SIGNAL_DEGRADATION_TIMESERIES.csv", index=False)
    if not out.empty:
        summary = pd.DataFrame([{
            "MeanTop10Return": out["Top10MeanReturn"].mean(),
            "MeanBottom10Return": out["Bottom10MeanReturn"].mean(),
            "MeanTopBottomSpread": out["TopBottomSpread"].mean(),
            "SpreadCI95Lower": bootstrap_ci(out["TopBottomSpread"])[1],
            "SpreadCI95Upper": bootstrap_ci(out["TopBottomSpread"])[2],
            "MeanTop10ExcessUniverse": out["Top10ExcessUniverse"].mean(),
            "Observations": len(out),
        }])
        summary.to_csv(OUT / "REVISION_XGB_SIGNAL_DEGRADATION.csv", index=False)


def run_cost_sensitivity():
    costs = [0, 5, 10, 20, 30, 50]
    rows = []
    predictions = pd.read_csv(PRED, parse_dates=["Date"])
    predictions = predictions[predictions["Date"].dt.year == 2025].copy()
    prices = load_prices(sorted(predictions["Ticker"].unique()))
    returns = prices.pct_change()
    for cost in costs:
        unified.COST_BPS = float(cost)
        for strategy, profile in [
            ("BuyHold", None), ("EqualWeight", None), ("RiskOnly", None),
            ("XGB_EqualWeight", None), ("XGB_Risk", "aggressive"),
            ("XGB_TurnoverAware", "aggressive"),
        ]:
            h = run_strategy(predictions, returns, strategy, profile)
            if h.empty:
                continue
            rows.append({"CostBps": cost, "Strategy": strategy, "RiskProfile": profile or "", **calculate_metrics(h)})
    unified.COST_BPS = 10.0
    pd.DataFrame(rows).to_csv(OUT / "REVISION_TRANSACTION_COST_SENSITIVITY.csv", index=False)

def gamma_validation():
    """Fast pre-2025 gamma selection using 2022-2024 validation dates only.

    To keep the revision reproducible and computationally practical, validation
    uses approximately monthly rebalance dates (every 20 prediction dates).
    No 2025 observation is used for gamma selection.
    """
    predictions = pd.read_csv(PRED, parse_dates=["Date"])
    predictions = predictions[predictions["Date"].dt.year.between(2022, 2024)].copy()
    prices = load_prices(sorted(predictions["Ticker"].unique()))
    returns = prices.pct_change()
    gammas = [0.0, 0.005, 0.01, 0.02]
    rows = []
    from research.portfolio.turnover_optimizer import optimize_portfolio, turnover, risk_aversion_for_profile

    for profile in ["conservative", "moderate", "aggressive"]:
        for gamma in gammas:
            dates = np.array(sorted(predictions["Date"].unique()))[::20]
            previous = None
            capital = INITIAL_CAPITAL
            hist = []
            for date in dates:
                current = predictions[predictions["Date"] == date]
                historical = returns.loc[returns.index < date]
                future = [d for d in returns.index if d > date][:HOLDING_DAYS]
                if current.empty or historical.empty or len(future) < HOLDING_DAYS:
                    continue
                weights = optimize_portfolio(
                    predictions=current, historical_returns=historical, previous_weights=previous,
                    k=10, risk_aversion=risk_aversion_for_profile(profile),
                    turnover_penalty=gamma, max_weight=0.20, max_turnover=None,
                    use_prediction_signal=True,
                )
                if weights.empty:
                    continue
                available = [t for t in weights.index if t in returns.columns]
                weights = weights[available]
                gross = (1 + (returns.loc[future, available] * weights).sum(axis=1)).prod() - 1
                t = turnover(previous, weights)
                net = gross - t * 10 / 10000
                capital *= 1 + net
                hist.append({"Date": date, "NetReturn": net, "PortfolioValue": capital, "Turnover": t, "TransactionCost": t * 10 / 10000})
                previous = weights.copy()
            h = pd.DataFrame(hist)
            if h.empty:
                continue
            rows.append({"RiskProfile": profile, "Gamma": gamma, **calculate_metrics(h)})

    result = pd.DataFrame(rows)
    result.to_csv(OUT / "REVISION_GAMMA_VALIDATION_2022_2024.csv", index=False)
    selected = (
        result.sort_values(["RiskProfile", "Sharpe", "AverageTurnover"], ascending=[True, False, True])
        .groupby("RiskProfile", as_index=False).head(1)
    )
    selected.to_csv(OUT / "REVISION_SELECTED_GAMMA.csv", index=False)
    print(selected[["RiskProfile", "Gamma", "Sharpe", "AverageTurnover", "Observations"]].to_string(index=False))

def yearly_robustness():
    predictions = pd.read_csv(PRED, parse_dates=["Date"])
    prices = load_prices(sorted(predictions["Ticker"].unique()))
    returns = prices.pct_change()
    rows = []
    for year in [2022, 2023, 2024, 2025]:
        p = predictions[predictions["Date"].dt.year == year].copy()
        if p.empty:
            continue
        for strategy, profile in [
            ("BuyHold", None),
            ("EqualWeight", None),
            ("RiskOnly", None),
            ("XGB_EqualWeight", None),
            ("XGB_Risk", "moderate"),
            ("XGB_TurnoverAware", "aggressive"),
        ]:
            h = run_strategy(p, returns, strategy, profile)
            if h.empty:
                continue
            rows.append({"Year": year, "Strategy": strategy, "RiskProfile": profile or "", **calculate_metrics(h)})
    pd.DataFrame(rows).to_csv(OUT / "REVISION_YEARLY_ROBUSTNESS.csv", index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Generate 10-bps corrected baseline first.
    env = os.environ.copy()
    env["SYNTHFIN_COST_BPS"] = "10"
    subprocess.run([sys.executable, "-m", "research.experiments.final_unified_2025"], check=True, env=env)
    forecast_tests()
    ic_tests()
    portfolio_significance()
    signal_degradation()
    gamma_validation()
    yearly_robustness()
    run_cost_sensitivity()
    print("\nAll revision analyses completed. See results/tables/REVISION_*.csv")


if __name__ == "__main__":
    main()

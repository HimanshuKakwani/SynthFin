# SynthFin ICCSDI 2026 Revision Runbook

This revision code directly targets every reviewer request.

## 1. Correct benchmark definitions

`research/experiments/final_unified_2025.py` now defines:

- `BuyHold`: equal-weight all available universe stocks once; no prediction and no later rebalance.
- `EqualWeight`: equal-weight all available universe stocks; no prediction.
- `RiskOnly`: lowest-volatility Top-10 selected from historical returns only, inverse-volatility weighted.
- `XGB_EqualWeight`: XGB Top-10 + equal weight.
- `XGB_Risk`: XGB Top-10 + risk-aware optimization.
- `XGB_TurnoverAware`: XGB + risk-aware optimization + turnover penalty.

This removes the old `RiskOnly` dependence on XGB predictions and removes the old `EqualWeight == XGB_EqualWeight` construction.

## 2. Run corrected 2025 experiment

From repository root:

```bash
python -m research.experiments.final_unified_2025
```

Outputs:

- `results/tables/REVISION_UNIFIED_2025_RESULTS.csv`
- `results/tables/REVISION_UNIFIED_2025_HISTORY.csv`

The revised 2025 experiment still has 48 five-day portfolio observations. Do not treat 4,840 stock observations as 4,840 independent portfolio observations.

## 3. Statistical tests

`research/experiments/revision_analysis.py` performs:

- date-level paired tests for Naive vs XGBoost forecast errors;
- Newey-West/HAC inference for temporal dependence;
- bootstrap 95% confidence intervals;
- one-sample IC significance tests;
- paired portfolio-return tests and bootstrap confidence intervals;
- XGB Top-10 vs bottom-10 signal-spread analysis.

Run:

```bash
python - <<'PY'
from research.experiments.revision_analysis import forecast_tests, ic_tests, portfolio_significance, signal_degradation
forecast_tests()
ic_tests()
portfolio_significance()
signal_degradation()
PY
```

Outputs begin with `results/tables/REVISION_`.

## 4. Gamma selection

Gamma is selected without using 2025. The validation code uses 2022–2024 only, with approximately monthly validation dates to keep the SLSQP optimization computationally practical.

Run:

```bash
python - <<'PY'
from research.experiments.revision_analysis import gamma_validation
gamma_validation()
PY
```

Output:

- `REVISION_GAMMA_VALIDATION_2022_2024.csv`
- `REVISION_SELECTED_GAMMA.csv`

For the supplied 20-stock data, this validation selected gamma = 0.02 for all three risk profiles. The revised 2025 experiment freezes these values before 2025.

## 5. Transaction-cost sensitivity

The corrected experiment accepts:

```bash
SYNTHFIN_COST_BPS=0
```

The sensitivity script evaluates:

- 0 bps
- 5 bps
- 10 bps
- 20 bps
- 30 bps
- 50 bps

Run:

```bash
python - <<'PY'
from research.experiments.revision_analysis import run_cost_sensitivity
run_cost_sensitivity()
PY
```

Output:

`results/tables/REVISION_TRANSACTION_COST_SENSITIVITY.csv`

## 6. Year-by-year robustness

Run:

```bash
python - <<'PY'
from research.experiments.revision_analysis import yearly_robustness
yearly_robustness()
PY
```

Output:

`results/tables/REVISION_YEARLY_ROBUSTNESS.csv`

This evaluates 2022, 2023, 2024 and 2025 separately.

## 7. Larger universe robustness

An expanded fixed 50-stock universe is provided at:

`data/universe_50.csv`

This is intentionally a fixed robustness universe, not a claim of survivorship-bias-free historical index membership.

Download prices:

```bash
python -m research.data.download_data \
  --universe data/universe_50.csv \
  --start 2018-01-01 \
  --end 2025-12-31
```

The downloader stores raw files. Clean them:

```bash
python - <<'PY'
from research.data.clean_data import clean_all
clean_all('data/raw', 'data/processed')
PY
```

Generate XGBoost walk-forward predictions for the expanded universe:

```bash
python -m research.backtest.walk_forward \
  --universe data/universe_50.csv \
  --output results/tables/walk_forward_predictions_50.csv
```

Then run:

```bash
python -m research.experiments.expanded_universe_robustness
```

Output:

`results/tables/REVISION_50STOCK_2025.csv`

The main paper's 20-stock experiment remains the full four-model prediction comparison. The 50-stock experiment is a robustness check of the portfolio pipeline using XGBoost.

## 8. Reviewer mapping

| Reviewer | Request | Code/result |
|---|---|---|
| R1-1 | Naive better point prediction | forecast significance + XGB Top-10 portfolio comparison |
| R1-2 | Only 20 equities | `universe_50.csv` + expanded-universe experiment |
| R1-3 | Identical strategy results | corrected benchmark definitions |
| R1-4 | No significance tests | HAC tests + bootstrap CIs + IC tests |
| R2-1 | Larger/different universe | 50-stock robustness |
| R2-2 | 48 observations | portfolio-level tests explicitly use 48 observations |
| R2-3 | Gamma not untouched | gamma frozen using only 2022–2024 |
| R2-4 | Fixed 10 bps | 0/5/10/20/30/50 bps sensitivity |
| R2-5 | Explain IC → portfolio deterioration | Top-10/bottom-10 spread + portfolio-layer comparison |
| R2-6 | More regimes/OOS periods | year-by-year 2022–2025 robustness |

## Important interpretation rule

Do not claim that XGBoost is statistically superior merely because it has the highest mean IC. In the supplied results, XGBoost mean IC is positive but its 95% bootstrap interval crosses zero. The stronger contribution is the end-to-end analysis of how a predictive signal changes after ranking, risk allocation, turnover control and transaction costs.

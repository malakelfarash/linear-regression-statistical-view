# Linear Regression: Statistical View (Python)

A from-scratch implementation of **simple linear regression** with full statistical inference, written in pure NumPy/SciPy. This project computes regression coefficients, ANOVA-based F-statistics, t-based confidence intervals, and R² without relying on `scikit-learn` or `statsmodels` and then **proves its correctness** by cross-validating every metric against `statsmodels.OLS`.

## Overview

Most tutorials stop at "here's the slope." This project goes the full distance and implements the statistical machinery behind linear regression:

- Ordinary Least Squares (OLS) coefficient estimation (β₀ and β₁)
- Sum of Squares decomposition (SSE, SSR, SST)
- F-statistic for the overall model significance test
- Confidence intervals for β₀ and β₁ using the Student's *t*-distribution
- Coefficient of determination (R²) and Pearson correlation (*r*)

The notebook then fits two competing models on real data and uses the F-statistic to decide which predictor is superior.

The hypothesis tested throughout is:

- **H₀:** β₁ = 0 (no linear relationship between predictor and response)
- **H₁:** β₁ ≠ 0 (a statistically significant linear relationship exists)

## Dataset

- **Source:** [Student Performance Factors - Kaggle](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors)
- **Size:** 6,607 rows × 20 columns
- **Response (Y):** `Exam_Score`
- **Predictors compared:** `Hours_Studied` vs. `Attendance`

The two numeric predictors with the highest correlation to `Exam_Score` were selected from the correlation matrix:

| Predictor | Pearson r with `Exam_Score` |
|-----------|------------------------------|
| Attendance | **0.5811** |
| Hours_Studied | 0.4455 |

## Key Results

| Metric | Model A (`Hours_Studied`) | Model B (`Attendance`) |
|--------|---------------------------|------------------------|
| β₀ (intercept) | 61.4570 | 51.5786 |
| β₁ (slope) | 0.2893 | 0.1958 |
| R² | 0.1984 | **0.3376** |
| Pearson *r* | 0.4455 | **0.5811** |
| F-statistic | 1635.08 | **3366.99** |
| F-critical (α = 0.05) | 3.8429 | 3.8429 |
| Decision (H₀: β₁ = 0) | **Reject H₀** | **Reject H₀** |

**Conclusion:** Both predictors are statistically significant, but `Attendance` is the superior predictor: it explains ~34% of the variance in exam scores (vs. ~20% for hours studied) and produces more than double the F-statistic.

## Verification Against statsmodels

Every computed statistic is verified line-by-line against `statsmodels.OLS`. All 22 metrics (coefficients, SSE, SSR, MSE, F-stat, R², and CI bounds) match to **1e-6 precision**, every check **Passed**.

Example output:

```
'Model B: Attendance' Verification vs statsmodels OLS
Metric          Custom        Statsmodels   Match
Bo              51.578577     51.578577     Passed
B1              0.195769      0.195769      Passed
SSE             66226.328120  66226.328120  Passed
SSR             33759.750584  33759.750584  Passed
MSE             10.026696     10.026696     Passed
F-statistic     3366.986498   3366.986498   Passed
R-squared       0.337645      0.337645      Passed
...
```

> `statsmodels` is used **only** to verify the from-scratch implementation, the core module has zero dependency on it.

## Public API

### `LinearRegressionStatisticalView(alpha=0.05)`

A self-contained simple linear regression model with full statistical inference. All values are returned as Python-native `float` / `dict`, ready for testing or downstream use.

| Method | Returns | Description |
|--------|---------|-------------|
| `fit(X, Y)` | `None` | Fits the model. Must be called before any other method. |
| `f_statistic()` | `float` | Model F-statistic (MSR / MSE). |
| `f_critical()` | `float` | Critical F-value at the configured α (`scipy.stats.f.isf`). |
| `beta_confidence_intervals()` | `dict` | `{"beta_0": (low, high), "beta_1": (low, high)}` at level 1−α. |
| `degrees_of_freedom()` | `dict` | `{"df_model": 1, "df_resid": n−2, "df_total": n−1}`. |
| `r_squared()` | `float` | Coefficient of determination (R²). |
| `r()` | `float` | Pearson correlation with sign of β₁. |
| `summary()` | `dict` | All of the above consolidated into one dictionary. |

### `compare(df, x_col_1, x_col_2, y_col, alpha=0.05)`

Module-level function that fits two independent simple linear regressions against the same response column and returns whichever predictor produced the larger F-statistic.

```python
{
    "predictor_1": x_col_1,
    "predictor_2": x_col_2,
    "f_stat_1": ...,
    "f_stat_2": ...,
    "best_predictor": ...,   # the column name with the higher F-statistic
}
```

## Tech Stack

- **Language:** Python 3.10+ (developed on 3.12)
- **Libraries:** numpy, scipy, pandas
- **Validation only:** statsmodels (for verification), jupyter (for the notebook)

## Getting Started

### Prerequisites

- Python 3.10 or newer
- pip

### Installation

No requirements file needed. Install the four dependencies directly:

```bash
pip install numpy scipy pandas statsmodels jupyter
```

Then clone the repo:

```bash
git clone https://github.com/YourUsername/linear-regression-statistical-view.git
```

### Run the notebook

```bash
jupyter notebook regression_validation.ipynb
```

Then **Kernel → Restart & Run All** to reproduce every number shown in the notebook.

### Use the module directly

```python
import pandas as pd
from linear_regression_statistical_view import LinearRegressionStatisticalView, compare

df = pd.read_csv('StudentPerformanceFactors.csv')

model = LinearRegressionStatisticalView(alpha=0.05)
model.fit(df['Attendance'].values, df['Exam_Score'].values)

print(model.f_statistic())                  # 3366.9865...
print(model.r_squared())                    # 0.3376...
print(model.beta_confidence_intervals())    # {'beta_0': (...), 'beta_1': (...)}
print(model.summary())                      # all stats in one dict

# Compare two predictors head-to-head
result = compare(df, 'Hours_Studied', 'Attendance', 'Exam_Score')
print(result['best_predictor'])             # 'Attendance'
```

# 📈 Multiple Linear Regression

Predict startup profitability using Multiple Linear Regression with Backward Elimination — implemented in both Python and R.

## What It Does

Models how **R&D Spend**, **Administration**, **Marketing Spend**, and **State** influence a startup's **Profit** using the 50 Startups dataset.

1. Loads and preprocesses data (one-hot encodes the categorical `State` column)
2. Splits into 80/20 train/test sets
3. Fits a Multiple Linear Regression model and reports R² / RMSE
4. Performs **Backward Elimination** (statsmodels OLS) to find the optimal predictor subset

## Dataset

`50_Startups.csv` — 50 records with columns:

| Column | Description |
|---|---|
| R&D Spend | Research & development expenditure |
| Administration | Administrative costs |
| Marketing Spend | Marketing expenditure |
| State | New York, California, or Florida |
| **Profit** | Target variable |

## 🛠 Tech Stack

| | Language | Libraries |
|---|---|---|
| 🐍 | Python 3.10+ | `numpy` · `pandas` · `matplotlib` · `scikit-learn` · `statsmodels` |
| 📊 | R | `caTools` |

## Getting Started

### Python

```bash
pip install numpy pandas matplotlib scikit-learn statsmodels
python multiple_linear_regression.py
```

### R

```r
install.packages("caTools")   # first time only
source("multiple_linear_regression.R")
```

> Both scripts expect `50_Startups.csv` in the same directory.

## ⚠️ Known Issues

- No cross-validation or hyperparameter tuning (simple demonstration project).
- The R script encodes `State` as numeric factor levels (`1, 2, 3`), which may imply ordinality — acceptable for `lm()` with factor types but worth noting.

## License

[MIT](LICENSE) © 2018 Kaustabh Ganguly

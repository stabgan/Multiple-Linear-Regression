# 📈 Multiple Linear Regression

Predict startup profitability using Multiple Linear Regression with Backward Elimination — in Python and R.

## What It Does

Models how R&D Spend, Administration, Marketing Spend, and State influence a startup's Profit using the **50 Startups** dataset.

1. Loads and preprocesses data (one-hot encodes categorical variables, avoids dummy variable trap)
2. Splits into 80/20 train/test sets
3. Fits a Multiple Linear Regression model
4. Performs **Backward Elimination** to identify the optimal predictor subset
5. Evaluates predictions on the test set

## Dataset

`50_Startups.csv` — 50 records with columns: `R&D Spend`, `Administration`, `Marketing Spend`, `State`, and `Profit` (target).

## How to Run

### 🐍 Python

```bash
pip install numpy pandas matplotlib scikit-learn statsmodels
python multiple_linear_regression.py
```

### 📊 R

```r
install.packages("caTools")  # first time only
source("multiple_linear_regression.R")
```

> Both scripts expect `50_Startups.csv` in the working directory.

## Tech Stack

| | Language | Libraries |
|---|---|---|
| 🐍 | Python 3 | `numpy` · `pandas` · `matplotlib` · `scikit-learn` · `statsmodels` |
| 📊 | R | `caTools` |

## Known Issues

- The Python script runs as a flat script (not modular/callable). OLS `.summary()` output is not printed — wrap in `print()` if you want to see it in a terminal.
- No evaluation metrics (R², RMSE, etc.) are computed beyond raw predictions.
- The R script encodes State as `1, 2, 3` factor levels, which may imply ordinality where none exists. This is acceptable for `lm()` with factor types but worth noting.

## License

[MIT](LICENSE) © 2018 Kaustabh Ganguly

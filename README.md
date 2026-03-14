# 📈 Multiple Linear Regression

Predict startup profitability using Multiple Linear Regression with **Backward Elimination** — implemented in both Python and R.

Uses the classic **50 Startups** dataset to model how R&D Spend, Administration costs, Marketing Spend, and State influence a company's Profit.

## What It Does

1. Loads and preprocesses the 50 Startups dataset (encodes categorical variables, avoids the dummy variable trap)
2. Splits data into training (80%) and test (20%) sets
3. Fits a Multiple Linear Regression model
4. Performs **Backward Elimination** to find the optimal subset of predictors
5. Evaluates predictions on the test set

## Tech Stack

| | Language | Libraries |
|---|---|---|
| 🐍 | Python | `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `statsmodels` |
| 📊 | R | `caTools` |

## Dataset

**`50_Startups.csv`** — 50 records of startup financial data.

| Column | Description |
|---|---|
| R&D Spend | Research & development expenditure |
| Administration | Administrative costs |
| Marketing Spend | Marketing expenditure |
| State | Location (New York, California, Florida) |
| Profit | **Target variable** |

## How to Run

### Python

```bash
pip install numpy pandas matplotlib scikit-learn statsmodels
python multiple_linear_regression.py
```

### R

```r
# Install dependency (first time only)
install.packages("caTools")

# Run
source("multiple_linear_regression.R")
```

> Make sure `50_Startups.csv` is in the same directory as the script.

## ⚠️ Known Issues (Deprecated APIs)

The Python script uses some APIs that have been deprecated/removed in newer versions of scikit-learn:

- **`sklearn.cross_validation`** → replaced by `sklearn.model_selection` (since sklearn 0.20)
- **`OneHotEncoder(categorical_features=...)`** → use `ColumnTransformer` + `OneHotEncoder` instead
- **`statsmodels.formula.api.OLS`** → should use `statsmodels.api.OLS` for the non-formula interface

To run as-is, pin `scikit-learn<0.20`. Otherwise, update the imports to modern equivalents.

## License

[MIT](LICENSE) © 2018 Kaustabh Ganguly

## Author

**[Kaustabh Ganguly](https://github.com/stabgan)** ([@stabgan](https://github.com/stabgan))

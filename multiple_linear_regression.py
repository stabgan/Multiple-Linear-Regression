# Multiple Linear Regression
# Predict startup profitability using backward elimination

# --- Imports (PEP 8: all imports at top) ---
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, root_mean_squared_error
import statsmodels.api as sm


def main():
    # --- Load dataset (relative to script location) ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "50_Startups.csv")
    dataset = pd.read_csv(csv_path)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    # --- Encode categorical data ---
    # ColumnTransformer + OneHotEncoder (modern sklearn API)
    # drop='first' avoids the dummy variable trap automatically
    ct = ColumnTransformer(
        transformers=[("encoder", OneHotEncoder(drop="first"), [3])],
        remainder="passthrough",
    )
    X = np.array(ct.fit_transform(X))

    # --- Split into training / test sets (80/20) ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # --- Fit Multiple Linear Regression ---
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # --- Predict & evaluate ---
    y_pred = regressor.predict(X_test)
    print("=== Model Evaluation ===")
    print(f"R² Score : {r2_score(y_test, y_pred):.4f}")
    print(f"RMSE     : {root_mean_squared_error(y_test, y_pred):.2f}")
    print()

    # --- Backward Elimination (statsmodels OLS) ---
    # sm.add_constant() is the recommended way to prepend an intercept column
    X_with_const = sm.add_constant(X)

    print("=== Backward Elimination ===\n")

    # Step 1: all predictors
    cols = list(range(X_with_const.shape[1]))
    X_opt = X_with_const[:, cols]
    results = sm.OLS(endog=y, exog=X_opt).fit()
    print(results.summary(), "\n")

    # Step 2: remove predictor with highest p-value > 0.05
    cols = [0, 1, 3, 4, 5]
    X_opt = X_with_const[:, cols]
    results = sm.OLS(endog=y, exog=X_opt).fit()
    print(results.summary(), "\n")

    # Step 3
    cols = [0, 3, 4, 5]
    X_opt = X_with_const[:, cols]
    results = sm.OLS(endog=y, exog=X_opt).fit()
    print(results.summary(), "\n")

    # Step 4
    cols = [0, 3, 5]
    X_opt = X_with_const[:, cols]
    results = sm.OLS(endog=y, exog=X_opt).fit()
    print(results.summary(), "\n")

    # Step 5: optimal model — R&D Spend only
    cols = [0, 3]
    X_opt = X_with_const[:, cols]
    results = sm.OLS(endog=y, exog=X_opt).fit()
    print(results.summary(), "\n")


if __name__ == "__main__":
    main()

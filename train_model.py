import json
import pickle
import numpy as np
import pandas as pd


def train(input_path="processed_data.csv", model_path="model_.pkl",
          metrics_path="metrics.json", test_size=0.2, seed=42):
    """
    Train a linear regression model using the Normal Equation and save
    the coefficients plus evaluation metrics to disk.

    Parameters
    ----------
    input_path   : str   Path to the preprocessed CSV.
    model_path   : str   Where to write the trained model (pickle).
    metrics_path : str   Where to write train/test R² scores (JSON).
    test_size    : float Fraction of data reserved for testing (0–1).
    seed         : int   Random seed for reproducibility.
    """
    df = pd.read_csv(input_path)

    # Features and target
    X = df[["Humidity", "Wind Speed", "condition_encoded"]].values
    y = df["Temperature"].values

    # --- TRAIN / TEST SPLIT ---
    np.random.seed(seed)
    indices      = np.random.permutation(len(X))
    n_test       = int(test_size * len(X))
    test_idx     = indices[:n_test]
    train_idx    = indices[n_test:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # --- NORMAL EQUATION ---
    # Prepend a column of 1s for the intercept term
    X_train_b = np.c_[np.ones(len(X_train)), X_train]
    X_test_b  = np.c_[np.ones(len(X_test)),  X_test]

    # θ = (XᵀX)⁻¹ Xᵀy  — closed-form optimal solution
    theta = np.linalg.inv(X_train_b.T @ X_train_b) @ X_train_b.T @ y_train

    intercept    = theta[0]
    coefficients = theta[1:]

    # --- EVALUATE : R² SCORE ---
    # R² = 1 − (SS_residual / SS_total)
    # R² = 1.0 means perfect prediction; 0.0 means the model is as good as the mean
    y_train_pred = X_train_b @ theta
    y_test_pred  = X_test_b  @ theta

    r2_train = 1 - np.mean((y_train - y_train_pred) ** 2) / np.var(y_train)
    r2_test  = 1 - np.mean((y_test  - y_test_pred)  ** 2) / np.var(y_test)

    # --- SAVE MODEL ---
    model = {"coefficients": coefficients, "intercept": intercept}
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # --- SAVE METRICS ---
    metrics = {"train_r2": round(float(r2_train), 4),
               "test_r2":  round(float(r2_test),  4)}
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Train R²: {r2_train:.4f} | Test R²: {r2_test:.4f}")
    print(f"Model saved to {model_path}")
    return intercept, coefficients, metrics


if __name__ == "__main__":
    train()
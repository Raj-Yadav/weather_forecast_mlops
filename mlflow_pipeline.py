import os
import subprocess
import pickle
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from train_model import train


class LinearRegressionModel:
    def __init__(self, coefficients, intercept):
        self.coefficients = coefficients
        self.intercept    = intercept

    def predict(self, X):
        return X @ self.coefficients + self.intercept


def run_command(cmd):
    """Helper to run shell commands (used for DVC/git integration)."""
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"[CMD ERROR] {result.stderr.strip()}")
    else:
        print(result.stdout.strip())


def run_pipeline(raw_path="raw_data.csv",
                 processed_path="processed_data.csv",
                 model_path="model_.pkl"):
    """
    Full pipeline: preprocess → train → log to MLFlow → version with DVC.
    """
    from preprocessing import preprocess
    preprocess(raw_path, processed_path)

    # Train the model and retrieve coefficients
    intercept, coefficients, metrics = train(
        input_path=processed_path,
        model_path=model_path,
    )

    # Load the feature matrix for signature inference
    df      = pd.read_csv(processed_path)
    X_train = df[["Humidity", "Wind Speed", "condition_encoded"]].values
    y_pred  = X_train @ coefficients + intercept

    # --- MLFLOW LOGGING ---
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:4040")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Weather Data Pipeline")

    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params({
            "test_size":   0.2,
            "random_seed": 42,
        })

        # Log evaluation metrics
        mlflow.log_metric("train_r2", metrics["train_r2"])
        mlflow.log_metric("test_r2",  metrics["test_r2"])

        # Tag the run for easy filtering in the UI
        mlflow.set_tag("Model Type", "Linear Regression")
        mlflow.set_tag("Features", "Humidity, Wind Speed, condition_encoded")

        # Infer signature: records the exact input shape and output dtype
        signature = infer_signature(X_train, y_pred)

        # Log and register the model in MLFlow's Model Registry
        mlflow.sklearn.log_model(
            sk_model=LinearRegressionModel(coefficients, intercept),
            artifact_path="weather_model",
            registered_model_name="weather_forecast_model",
            signature=signature,
        )
        print("Model logged and registered in MLFlow.")

    # --- VERSION MODEL WITH DVC ---
    if not os.path.exists(f"{model_path}.dvc"):
        run_command(f"dvc add {model_path}")
    run_command("git add .")
    run_command("git commit -m 'chore: update trained model artifact'")
    run_command("dvc push")


if __name__ == "__main__":
    run_pipeline()
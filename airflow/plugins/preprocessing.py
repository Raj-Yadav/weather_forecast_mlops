import pandas as pd
import numpy as np


def preprocess(input_path="raw_data.csv", output_path="processed_data.csv"):
    """
    Load raw weather data, normalize numerical columns, encode the
    categorical weather condition column, and save the result.

    Parameters
    ----------
    input_path  : str  Path to the raw CSV file.
    output_path : str  Path where the processed CSV will be written.
    """
    df = pd.read_csv(input_path)

    # --- Z-SCORE NORMALIZATION ---
    # Formula: z = (x - μ) / σ
    # After this transformation each column has mean=0 and std=1.
    numerical_columns = ["Temperature", "Humidity", "Wind Speed"]
    means = df[numerical_columns].mean()
    stds  = df[numerical_columns].std()
    df[numerical_columns] = (df[numerical_columns] - means) / stds

    # --- LABEL ENCODING ---
    # Map each unique weather condition string to a unique integer.
    # Example: {"light rain": 0, "overcast clouds": 1, "clear sky": 2}
    conditions        = df["Weather Condition"].unique()
    condition_to_int  = {cond: idx for idx, cond in enumerate(conditions)}
    df["condition_encoded"] = df["Weather Condition"].map(condition_to_int)

    df.to_csv(output_path, index=False)
    print(f"Preprocessed {len(df)} rows → {output_path}")


if __name__ == "__main__":
    preprocess()
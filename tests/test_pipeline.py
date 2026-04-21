import unittest
import os
import json
import pickle
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock


class TestDataCollection(unittest.TestCase):
    """Test the data_collection module without calling the real API."""

    @patch("data_collection.requests.get")
    def test_fetch_returns_list(self, mock_get):
        """fetch_weather_data should return a list of forecast entries."""
        # Arrange: mock the API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "list": [
                {
                    "dt": 1698844800,
                    "main": {"temp": 14.2, "humidity": 72},
                    "wind": {"speed": 3.5},
                    "weather": [{"description": "light rain"}],
                }
            ]
        }
        mock_get.return_value = mock_response

        from data_collection import fetch_weather_data
        result = fetch_weather_data()

        # Assert: result is a list with one entry
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIn("main", result[0])


class TestPreprocessing(unittest.TestCase):
    """Test that preprocessing produces correctly normalized output."""

    def test_normalization_produces_zero_mean(self):
        """After Z-score normalization, each numerical column should have mean ≈ 0."""
        # Arrange: create a small synthetic DataFrame
        df = pd.DataFrame({
            "Date":              ["2024-01-01"] * 10,
            "Time":              ["12:00:00"] * 10,
            "Temperature":       np.random.uniform(10, 20, 10),
            "Humidity":          np.random.uniform(60, 80, 10),
            "Wind Speed":        np.random.uniform(2, 8, 10),
            "Weather Condition": ["light rain"] * 5 + ["clear sky"] * 5,
        })
        df.to_csv("test_raw.csv", index=False)

        from preprocessing import preprocess
        preprocess("test_raw.csv", "test_processed.csv")

        result = pd.read_csv("test_processed.csv")

        # Assert: normalized columns have mean close to 0
        for col in ["Temperature", "Humidity", "Wind Speed"]:
            self.assertAlmostEqual(result[col].mean(), 0.0, places=5)

        # Cleanup
        os.remove("test_raw.csv")
        os.remove("test_processed.csv")


class TestModel(unittest.TestCase):
    """Test the LinearRegressionModel wrapper."""

    def test_predict_output_shape(self):
        """predict() should return an array with the same length as the input."""
        from mlflow_pipeline import LinearRegressionModel

        coefficients = np.array([0.5, -0.3, 0.2])
        intercept    = 1.0
        model        = LinearRegressionModel(coefficients, intercept)

        X      = np.random.rand(10, 3)
        output = model.predict(X)

        self.assertEqual(output.shape, (10,))

    def test_predict_correct_values(self):
        """predict() should compute X·θ + b correctly."""
        from mlflow_pipeline import LinearRegressionModel

        coefficients = np.array([1.0, 0.0, 0.0])  # only first feature matters
        intercept    = 5.0
        model        = LinearRegressionModel(coefficients, intercept)

        X        = np.array([[2.0, 0.0, 0.0]])
        expected = 2.0 * 1.0 + 5.0      # = 7.0
        result   = model.predict(X)[0]

        self.assertAlmostEqual(result, expected, places=5)


if __name__ == "__main__":
    unittest.main()
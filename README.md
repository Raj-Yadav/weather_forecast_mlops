# Weather Forecast MLOps ⛅

A complete end-to-end Machine Learning Operations (MLOps) project that predicts weather temperatures based on humidity, wind speed, and weather condition data. 

### 🚀 Key Technologies Used:
* **Machine Learning Pipeline:** Data collection, preprocessing, and training a Custom Linear Regression model.
* **Experiment Tracking:** MLflow for tracking metrics and model artifacts.
* **Data Versioning:** DVC (Data Version Control) to track raw and processed datasets.
* **Orchestration:** Apache Airflow to automate the data pipeline.
* **Backend API:** FastAPI application serving the predictions and user authentication (SQLite).
* **Containerization & Deployment:** Dockerized application deployed on a local Kubernetes cluster (Minikube).
* **Testing:** Unit tests for pipeline validation using `unittest` and `mock`.

ecco
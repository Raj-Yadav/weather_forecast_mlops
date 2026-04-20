from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator

# --- DEFAULT ARGUMENTS ---
# These apply to every task in the DAG unless overridden at the task level.
default_args = {
    "owner":             "airflow",
    "depends_on_past":   False,    # do not wait for previous run to succeed
    "email_on_failure":  False,
    "email_on_retry":    False,
    "retries":           1,        # retry once on task failure
    "retry_delay":       timedelta(minutes=5),
}

# --- DAG DEFINITION ---
dag = DAG(
    dag_id="weather_data_pipeline",          # unique identifier shown in the UI
    default_args=default_args,
    description="Daily weather data collection and preprocessing",
    schedule = timedelta(days=1),     # run every 24 hours
    start_date=datetime(2024, 1, 1),
    catchup=False,                           # do not backfill missed runs
    tags=["weather", "mlops"],
)


# --- TASK FUNCTIONS ---
# Each function wraps the module-level function we already wrote.

def task_collect():
    """Fetch fresh weather data from the API and write raw_data.csv."""
    import sys
    import os
    # Add project root to path so imports work inside Airflow's process
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from data_collection import fetch_weather_data, save_weather_data
    data = fetch_weather_data()
    save_weather_data(data)


def task_preprocess():
    """Normalize and encode raw_data.csv → processed_data.csv."""
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from preprocessing import preprocess
    preprocess()


# --- TASK OPERATORS ---
collect_task = PythonOperator(
    task_id="collect_weather_data",
    python_callable=task_collect,
    dag=dag,
)

preprocess_task = PythonOperator(
    task_id="preprocess_weather_data",
    python_callable=task_preprocess,
    dag=dag,
)

# --- DEPENDENCY: collect must finish before preprocess starts ---
collect_task >> preprocess_task
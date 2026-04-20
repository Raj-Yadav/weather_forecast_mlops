import csv
import os
import requests
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
API_KEY = os.getenv("WEATHER_API_KEY")
CITY    = os.getenv("WEATHER_CITY", "London")

# Default to the standard 5-day forecast endpoint if not provided
BASE_URL = os.getenv("WEATHER_API_URL", "https://api.openweathermap.org/data/2.5/forecast")

def fetch_weather_data():
    """
    Call the OpenWeatherMap 5-day forecast endpoint.
    """
    # Define query parameters as a dictionary
    params = {
        "q": CITY,
        "appid": API_KEY,
        "units": "metric"
    }

    try:
        # Pass the base URL and the parameters dictionary
        response = requests.get(BASE_URL, params=params)
        # Raise an exception for 4xx or 5xx status codes
        response.raise_for_status()
        
        return response.json().get("list", [])
        
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"An unexpected error occurred: {err}")
    
    return None



def save_weather_data(data, filename="raw_data.csv"):
    """
    Persist the raw API response to a CSV file.

    Parameters
    ----------
    data : list
        The list returned by fetch_weather_data().
    filename : str
        Destination file path (default: raw_data.csv).
    """
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        
        # Header row
        writer.writerow([
            "Date", "Time", "Temperature",
            "Humidity", "Wind Speed", "Weather Condition"
        ])
        for entry in data:
            # Convert UNIX timestamp → human-readable date and time
            dt_str = datetime.fromtimestamp(entry["dt"]).strftime("%Y-%m-%d %H:%M:%S")
            date, time = dt_str.split(" ")
            writer.writerow([
                date,
                time,
                entry["main"]["temp"],              # °C  (units=metric)
                entry["main"]["humidity"],           # %
                entry["wind"]["speed"],              # m/s
                entry["weather"][0]["description"],  # e.g. "light rain"
            ])
    print(f"Saved {len(data)} rows to {filename}")


if __name__ == "__main__":
    weather_data = fetch_weather_data()
    save_weather_data(weather_data)
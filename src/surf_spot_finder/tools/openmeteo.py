import json
from datetime import datetime, timedelta
import requests


def _extract_hourly_data(data: dict) -> list[dict]:
    hourly_data = data["hourly"]
    result = [
        {k: v for k, v in zip(hourly_data.keys(), values)}
        for values in zip(*hourly_data.values())
    ]
    return result


def _filter_by_date(
    date: datetime, hourly_data: list[dict], timedelta: timedelta = timedelta(hours=1)
):
    start_date = date - timedelta
    end_date = date + timedelta
    return [
        item
        for item in hourly_data
        if start_date <= datetime.fromisoformat(item["time"]) <= end_date
    ]


def get_wave_forecast(lat: float, lon: float, date: str | None = None) -> list[dict]:
    """Get wave forecast for given location.

    Forecast will include:

    - wave_direction (degrees)
    - wave_height (meters)
    - wave_period (seconds)
    - sea_level_height_msl (meters)

    Args:
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.
        date (str | None): Date to filter by in any valid ISO 8601 format.
            If not provided, all data (default to 6 days forecast) will be returned.

    Returns:
        list[dict]: Hourly data for wave forecast.
            Example output:

            ```json
            [
                {'time': '2025-03-19T09:00', 'winddirection_10m': 140, 'windspeed_10m': 24.5}, {'time': '2025-03-19T10:00', 'winddirection_10m': 140, 'windspeed_10m': 27.1},
                {'time': '2025-03-19T10:00', 'winddirection_10m': 140, 'windspeed_10m': 27.1}, {'time': '2025-03-19T11:00', 'winddirection_10m': 141, 'windspeed_10m': 29.2}
            ]
            ```
    """
    url = "https://marine-api.open-meteo.com/v1/marine"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "wave_direction",
            "wave_height",
            "wave_period",
            "sea_level_height_msl",
        ],
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = json.loads(response.content.decode())
    hourly_data = _extract_hourly_data(data)
    if date is not None:
        date = datetime.fromisoformat(date)
        hourly_data = _filter_by_date(date, hourly_data)
    return hourly_data


def get_wind_forecast(lat: float, lon: float, date: str | None = None) -> list[dict]:
    """Get wind forecast for given location.

    Forecast will include:

    - wind_direction (degrees)
    - wind_speed (meters per second)

    Args:
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.
        date (str | None): Date to filter by in any valid ISO 8601 format.
            If not provided, all data (default to 6 days forecast) will be returned.

    Returns:
        list[dict]: Hourly data for wind forecast.
            Example output:

            ```json
            [
                {"time": "2025-03-18T22:00", "wave_direction": 264, "wave_height": 2.24, "wave_period": 10.45, "sea_level_height_msl": -1.27},
                {"time": "2025-03-18T23:00", "wave_direction": 264, "wave_height": 2.24, "wave_period": 10.35, "sea_level_height_msl": -1.35},
            ]
            ```
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["winddirection_10m", "windspeed_10m"],
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = json.loads(response.content.decode())
    hourly_data = _extract_hourly_data(data)
    if date is not None:
        date = datetime.fromisoformat(date)
        hourly_data = _filter_by_date(date, hourly_data)
    return hourly_data

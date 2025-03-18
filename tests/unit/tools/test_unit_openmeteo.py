import json
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from surf_spot_finder.tools import openmeteo


def test_extract_hourly_data():
    data = {
        "hourly": {
            "time": ["2023-01-01T00:00", "2023-01-01T01:00"],
            "wave_height": [1.5, 1.6],
            "wave_period": [10, 11],
        }
    }
    expected = [
        {"time": "2023-01-01T00:00", "wave_height": 1.5, "wave_period": 10},
        {"time": "2023-01-01T01:00", "wave_height": 1.6, "wave_period": 11},
    ]
    assert openmeteo._extract_hourly_data(data) == expected


def test_filter_by_date():
    hourly_data = [
        {"time": "2023-01-01T00:00", "wave_height": 1.5},
        {"time": "2023-01-01T01:00", "wave_height": 1.6},
        {"time": "2023-01-01T02:00", "wave_height": 1.7},
        {"time": "2023-01-01T03:00", "wave_height": 1.8},
    ]
    date = datetime.fromisoformat("2023-01-01T01:00")
    expected = [
        {"time": "2023-01-01T00:00", "wave_height": 1.5},
        {"time": "2023-01-01T01:00", "wave_height": 1.6},
        {"time": "2023-01-01T02:00", "wave_height": 1.7},
    ]
    assert openmeteo._filter_by_date(date, hourly_data) == expected

    expected = [
        {"time": "2023-01-01T01:00", "wave_height": 1.6},
    ]
    assert openmeteo._filter_by_date(date, hourly_data, timedelta(hours=0)) == expected


def test_get_wave_forecast():
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content.decode.return_value = json.dumps(
            {
                "hourly": {
                    "time": ["2023-02-02T00:00", "2023-02-02T01:00"],
                    "wave_direction": [270, 280],
                    "wave_height": [1.5, 1.6],
                    "wave_period": [10, 11],
                    "sea_level_height_msl": [0.5, 0.6],
                }
            }
        )
        mock_get.return_value = mock_response
        result = openmeteo.get_wave_forecast(lat=40.0, lon=-3.0)

        assert len(result) == 2
        assert result[1]["time"] == "2023-02-02T01:00"
        assert result[1]["wave_direction"] == 280
        assert result[1]["wave_height"] == 1.6
        assert result[1]["wave_period"] == 11
        assert result[1]["sea_level_height_msl"] == 0.6

        result_filtered = openmeteo.get_wave_forecast(
            lat=40.0, lon=-3.0, date="2023-02-02T02:00"
        )
        assert len(result_filtered) == 1
        assert result_filtered[0]["time"] == "2023-02-02T01:00"


def test_get_wind_forecast():
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content.decode.return_value = json.dumps(
            {
                "hourly": {
                    "time": ["2023-02-02T00:00", "2023-02-02T01:00"],
                    "winddirection_10m": [270, 280],
                    "windspeed_10m": [10, 11],
                }
            }
        )
        mock_get.return_value = mock_response

        result = openmeteo.get_wind_forecast(lat=40.0, lon=-3.0)
        assert len(result) == 2
        assert result[1]["time"] == "2023-02-02T01:00"
        assert result[1]["winddirection_10m"] == 280
        assert result[1]["windspeed_10m"] == 11

        result_filtered = openmeteo.get_wind_forecast(
            lat=40.0, lon=-3.0, date="2023-02-02T02:00"
        )
        assert len(result_filtered) == 1
        assert result_filtered[0]["time"] == "2023-02-02T01:00"
        assert result[0]["windspeed_10m"] == 10

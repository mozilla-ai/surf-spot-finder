import json
from unittest.mock import MagicMock, patch

from surf_spot_finder.tools import openstreetmap


def test_get_area_lat_lon():
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content.decode.return_value = json.dumps(
            [{"lat": "40.0", "lon": "-3.0"}]
        )
        mock_get.return_value = mock_response

        lat, lon = openstreetmap.get_area_lat_lon("Madrid")
        assert lat == "40.0"
        assert lon == "-3.0"


def test_driving_hours_to_meters():
    assert openstreetmap.driving_hours_to_meters(1) == 70000


def test_get_lat_lon_center():
    bounds = {"minlat": 40.0, "minlon": -3.0, "maxlat": 41.0, "maxlon": -2.0}
    lat, lon = openstreetmap.get_lat_lon_center(bounds)
    assert lat == 40.5
    assert lon == -2.5


def test_get_surfing_spots():
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "elements": [
                {
                    "tags": {"name": "Surf Spot 1", "sport": "surfing"},
                    "bounds": {
                        "minlat": 40.0,
                        "minlon": -3.0,
                        "maxlat": 40.1,
                        "maxlon": -2.9,
                    },
                },
                {
                    "tags": {"name": "Beach 2", "sport": "swimming"},
                    "bounds": {
                        "minlat": 41.0,
                        "minlon": -4.0,
                        "maxlat": 41.1,
                        "maxlon": -3.9,
                    },
                },
                {
                    "tags": {"name": "Surf Spot 3", "sport": "surfing"},
                    "bounds": {
                        "minlat": 42.0,
                        "minlon": -5.0,
                        "maxlat": 42.1,
                        "maxlon": -4.9,
                    },
                },
            ]
        }
        mock_get.return_value = mock_response

        results = openstreetmap.get_surfing_spots(lat=40.5, lon=-3.5, radius=10000)
        assert len(results) == 2
        assert results[0][0] == "Surf Spot 1"
        assert results[0][1] == (40.05, -2.95)
        assert results[1][0] == "Surf Spot 3"
        assert results[1][1] == (42.05, -4.95)

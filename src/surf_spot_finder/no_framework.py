import re
from datetime import datetime

from fire import Fire
from litellm import completion
from loguru import logger
from pydantic import BaseModel

from any_agent.tools.web_browsing import search_web, visit_webpage
from surf_spot_finder.tools.openmeteo import get_wave_forecast, get_wind_forecast
from surf_spot_finder.tools.openstreetmap import (
    driving_hours_to_meters,
    get_area_lat_lon,
    get_surfing_spots,
)


spot_info_pattern = r"\[.*?\]\((https:\/\/www\.surf-forecast\.com\/breaks\/[^)/]+)\)"


class SpotScore(BaseModel):
    score: int
    reason: str


@logger.catch(reraise=True)
def find_surf_spot_no_framework(
    location: str, max_driving_hours: int, date: datetime, model_id: str
) -> list[SpotScore]:
    """Find the best surf spot based on the given `location` and `date`.

    Uses the following tools:

    - any_agent.tools.web_browsing
    - [surf_spot_finder.tools.openmeteo][]
    - [surf_spot_finder.tools.openstreetmap][]

    To find nearby spots along with the forecast and
    recommended conditions for the spot.

    Then, uses `litellm` with the provided `model_id` to score
    each spot based on the available information.

    Args:
        location: The place of interest.
        max_driving_hours: Used to limit the surf spots based on
            the distance to `location`.
        date: Used to filter the forecast results.
        model_id: Can be any of the [litellm providers](https://docs.litellm.ai/docs/providers).

    Returns:
        A list of spot scores and reasons for the value.
    """
    max_driving_meters = driving_hours_to_meters(max_driving_hours)
    lat, lon = get_area_lat_lon(location)

    logger.info(f"Getting surfing spots around {location}")
    surf_spots = get_surfing_spots(lat, lon, max_driving_meters)

    if not surf_spots:
        logger.warning("No surfing spots found around {location}")
        return None

    spots_scores = []
    for spot_name, (spot_lat, spot_lon) in surf_spots:
        logger.info(f"Processing {spot_name}")
        logger.debug("Getting wave forecast...")
        wave_forecast = get_wave_forecast(spot_lat, spot_lon, date)
        logger.debug("Getting wind forecast...")
        wind_forecast = get_wind_forecast(spot_lat, spot_lon, date)

        logger.debug("Searching web for spot information")
        search_result = search_web(f"surf-forecast.com spot info {spot_name}")
        match = re.search(spot_info_pattern, search_result)
        if match:
            extracted_url = match.group(1)
            logger.debug(f"Visiting {extracted_url}")
            spot_info = visit_webpage(extracted_url)
        else:
            logger.debug(f"Couldn't find spot info for {spot_name}")
            continue

        logger.debug("Scoring conditions with LLM")
        response = completion(
            model="openai/gpt-4o-mini",
            messages=[
                {
                    "content": "Given the wind and wave forecast along with the spot information, "
                    "rate from 1 to 5 the expected surfing conditions."
                    f"Wind forecast:\n{wind_forecast}\n"
                    f"Wave forecast:\n{wave_forecast}\n"
                    f"Spot Information:\n{spot_info}",
                    "role": "user",
                }
            ],
            response_format=SpotScore,
        )
        spot_score = SpotScore.model_validate_json(response.choices[0].message.content)
        logger.debug(spot_score)
        spots_scores.append(spot_score)

    return spots_scores


def main():
    Fire(find_surf_spot_no_framework)


if __name__ == "__main__":
    main()

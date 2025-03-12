from typing import Optional

from fire import Fire
from loguru import logger

from surf_spot_finder.config import (
    Config,
    DEFAULT_PROMPT,
)
from surf_spot_finder.agents.smolagents import run_smolagent
from surf_spot_finder.tracing import setup_tracing


@logger.catch(reraise=True)
def find_surf_spot(
    location: str,
    date: str,
    max_driving_hours: int,
    model_id: str,
    api_key_var: Optional[str] = None,
    prompt: str = DEFAULT_PROMPT,
    json_tracer: bool = True,
    api_base: Optional[str] = None,
):
    logger.info("Loading config")
    config = Config(
        location=location,
        date=date,
        max_driving_hours=max_driving_hours,
        model_id=model_id,
        api_key_var=api_key_var,
        prompt=prompt,
        json_tracer=json_tracer,
        api_base=api_base,
    )

    logger.info("Setting up tracing")
    setup_tracing(project_name="surf-spot-finder", json_tracer=config.json_tracer)

    logger.info("Running agent")
    run_smolagent(
        model_id=config.model_id,
        api_key_var=config.api_key_var,
        api_base=config.api_base,
        prompt=config.prompt.format(
            LOCATION=config.location,
            MAX_DRIVING_HOURS=config.max_driving_hours,
            DATE=config.date,
        ),
    )


def main():
    Fire(find_surf_spot)

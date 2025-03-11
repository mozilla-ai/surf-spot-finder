from typing import Optional

from fire import Fire
from loguru import logger

from surf_spot_finder.config import (
    Config,
    DEFAULT_PROMPT,
)
from surf_spot_finder.agents.smolagents import load_smolagent
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
    )

    logger.info("Loading agent")
    agent = load_smolagent(config.model_id, config.api_key_var)

    logger.info("Setting up tracing")
    setup_tracing(project_name="find-surf-spot", json_tracer=config.json_tracer)

    logger.info("Running agent")
    agent.run(
        config.prompt.format(
            LOCATION=config.location,
            MAX_DRIVING_HOURS=config.max_driving_hours,
            DATE=config.date,
        )
    )


def main():
    Fire(find_surf_spot)

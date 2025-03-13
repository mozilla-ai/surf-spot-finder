from typing import Optional

from fire import Fire
from loguru import logger

from surf_spot_finder.config import (
    Config,
    DEFAULT_PROMPT,
)
from surf_spot_finder.agents import RUNNERS
from surf_spot_finder.tracing import get_tracer_provider, setup_tracing


@logger.catch(reraise=True)
def find_surf_spot(
    location: str,
    date: str,
    max_driving_hours: int,
    model_id: str,
    agent_type: str = "smolagents",
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
        agent_type=agent_type,
        api_key_var=api_key_var,
        prompt=prompt,
        json_tracer=json_tracer,
        api_base=api_base,
    )

    logger.info("Setting up tracing")
    tracer_provider = get_tracer_provider(
        project_name="surf-spot-finder", json_tracer=config.json_tracer
    )
    setup_tracing(tracer_provider, config.agent_type)

    logger.info(f"Running {config.agent_type} agent")
    RUNNERS[config.agent_type](
        model_id=config.model_id,
        prompt=config.prompt.format(
            LOCATION=config.location,
            MAX_DRIVING_HOURS=config.max_driving_hours,
            DATE=config.date,
        ),
    )


def main():
    Fire(find_surf_spot)

from pathlib import Path
from typing import Optional

import yaml
from fire import Fire
from loguru import logger

from surf_spot_finder.config import (
    Config,
)
from surf_spot_finder.agents import RUNNERS
from surf_spot_finder.prompts.shared import INPUT_PROMPT
from surf_spot_finder.tracing import get_tracer_provider, setup_tracing


@logger.catch(reraise=True)
def find_surf_spot(
    location: Optional[str] = None,
    date: Optional[str] = None,
    max_driving_hours: Optional[int] = None,
    model_id: Optional[str] = None,
    agent_type: str = "smolagents",
    api_key_var: Optional[str] = None,
    input_prompt_template: str = INPUT_PROMPT,
    json_tracer: bool = True,
    api_base: Optional[str] = None,
    tools: Optional[list[dict]] = None,
    from_config: Optional[str] = None,
) -> str:
    """Find the best surf spot based on the given criteria.

    Args:
        location (str): The location to search around.
            Required if `from_config` is not provided.
        date (str): The date to search for.
            Required if `from_config` is not provided.
        max_driving_hours (int): The maximum driving hours from the location.
            Required if `from_config` is not provided.
        model_id (str): The ID of the model to use.
            Required if `from_config` is not provided.

            If using `agent_type=smolagents`, use LiteLLM syntax (e.g., 'openai/o1', 'anthropic/claude-3-sonnet').
            If using `agent_type={openai,openai_multi_agent}`, use OpenAI syntax (e.g., 'o1').
        agent_type (str, optional): The type of agent to use.
            Must be one of the supported types in [RUNNERS][surf_spot_finder.agents.RUNNERS].
        api_key_var (Optional[str], optional): The name of the environment variable containing the API key.
        input_prompt_template (str, optional): The template for the imput_prompt.

            Must contain the following placeholders: `{LOCATION}`, `{MAX_DRIVING_HOURS}`, and `{DATE}`.
        json_tracer (bool, optional): Whether to use the custom JSON file exporter.
        api_base (Optional[str], optional): The base URL for the API.
        from_config (Optional[str], optional): Path to a YAML config file.

            If provided, all other arguments will be ignored.
    """
    if from_config:
        logger.info(f"Loading {from_config}")
        config = Config.model_validate(yaml.safe_load(Path(from_config).read_text()))
    else:
        config = Config(
            location=location,
            date=date,
            max_driving_hours=max_driving_hours,
            model_id=model_id,
            agent_type=agent_type,
            api_key_var=api_key_var,
            prompt=input_prompt_template,
            json_tracer=json_tracer,
            api_base=api_base,
            tools=tools,
        )

    logger.info("Setting up tracing")
    tracer_provider, tracing_path = get_tracer_provider(
        project_name="surf-spot-finder",
        json_tracer=config.json_tracer,
        agent_type=config.agent_type,
    )
    setup_tracing(tracer_provider, config.agent_type)

    logger.info(f"Running {config.agent_type} agent")
    RUNNERS[config.agent_type](
        model_id=config.model_id,
        prompt=config.input_prompt_template.format(
            LOCATION=config.location,
            MAX_DRIVING_HOURS=config.max_driving_hours,
            DATE=config.date,
        ),
        api_base=config.api_base,
        api_key_var=config.api_key_var,
        tools=config.tools,
    )
    return tracing_path


def main():
    Fire(find_surf_spot)


if __name__ == "__main__":
    main()

from any_agent import AgentFramework, AnyAgent
import yaml
from pathlib import Path

from fire import Fire
from loguru import logger

from surf_spot_finder.config import (
    Config,
)
from any_agent.tracing import setup_tracing

from surf_spot_finder.instructions.openai import SINGLE_AGENT_SYSTEM_PROMPT
from surf_spot_finder.instructions.smolagents import SYSTEM_PROMPT


@logger.catch(reraise=True)
def find_surf_spot(
    config_file: str,
) -> str:
    """Find the best surf spot based on the given criteria.

    Args:
        config_file: Path to a YAML config file.
            See [Config][surf_spot_finder.config.Config]

    """
    logger.info(f"Loading {config_file}")
    config = Config.model_validate(yaml.safe_load(Path(config_file).read_text()))

    if not config.main_agent.instructions:
        if config.framework == AgentFramework.SMOLAGENTS:
            config.main_agent.instructions = SYSTEM_PROMPT
        elif config.framework == AgentFramework.OPENAI:
            config.main_agent.instructions = SINGLE_AGENT_SYSTEM_PROMPT

    logger.info("Setting up tracing")
    tracing_path = setup_tracing(config.framework, "output")

    logger.info(f"Loading {config.framework} agent")
    logger.info(f"{config.managed_agents}")
    agent = AnyAgent.create(
        agent_framework=config.framework,
        agent_config=config.main_agent,
        managed_agents=config.managed_agents,
    )

    query = config.input_prompt_template.format(
        LOCATION=config.location,
        MAX_DRIVING_HOURS=config.max_driving_hours,
        DATE=config.date,
    )
    logger.info(f"Running agent with query:\n{query}")
    agent.run(query)

    logger.success("Done!")

    return tracing_path


def main():
    Fire(find_surf_spot)


if __name__ == "__main__":
    main()

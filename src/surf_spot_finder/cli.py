from any_agent import AnyAgent
import yaml
from pathlib import Path

from fire import Fire
from loguru import logger

from surf_spot_finder.config import (
    Config,
)
from any_agent.tracing import get_tracer_provider, setup_tracing


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

    logger.info("Setting up tracing")
    tracer_provider, tracing_path = get_tracer_provider(
        project_name="surf-spot-finder", agent_framework=config.framework
    )
    setup_tracing(tracer_provider, config.framework)

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

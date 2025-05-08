import datetime
import os
from pathlib import Path

from any_agent import AgentFramework, AnyAgent, TracingConfig
from any_agent.evaluation.schemas import TraceEvaluationResult
from fire import Fire
from any_agent.logging import logger
from any_agent.evaluation import evaluate

from surf_spot_finder.config import (
    Config,
)

from surf_spot_finder.instructions.openai import SINGLE_AGENT_SYSTEM_PROMPT
from surf_spot_finder.instructions.smolagents import SYSTEM_PROMPT


async def find_surf_spot(
    config_file: str | None = None,
) -> str:
    """Find the best surf spot based on the given criteria.

    Args:
        config_file: Path to a YAML config file.
            See [Config][surf_spot_finder.config.Config]

    """
    if config_file is None:
        config = Config.from_dict({})
    else:
        logger.info("Loading %s", config_file)
        config = Config.from_yaml(config_file)

    if not config.main_agent.instructions:
        if config.framework == AgentFramework.SMOLAGENTS:
            config.main_agent.instructions = SYSTEM_PROMPT
        elif config.framework == AgentFramework.OPENAI:
            config.main_agent.instructions = SINGLE_AGENT_SYSTEM_PROMPT

    logger.info("Loading %s agent", config.framework)
    logger.info("Managed agents: %s", config.managed_agents)
    agent = await AnyAgent.create_async(
        agent_framework=config.framework,
        agent_config=config.main_agent,
        managed_agents=config.managed_agents,
        tracing=TracingConfig(console=True, cost_info=True),
    )

    query = config.input_prompt_template.format(
        LOCATION=config.location,
        MAX_DRIVING_HOURS=config.max_driving_hours,
        DATE=config.date,
    )
    logger.info("Running agent with query:\n%s", query)
    agent_trace = await agent.run_async(query)

    logger.info("Final output from agent:\n%s", agent_trace.final_output)

    # dump the trace in the "output" directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = Path(output_dir) / f"{timestamp}_trace.json"
    with open(file_path, "w") as f:
        f.write(agent_trace.model_dump_json(indent=2))

    if config.evaluation_cases is not None:
        results = []
        logger.info("Found evaluation cases, running trace evaluation")
        for i, case in enumerate(config.evaluation_cases):
            logger.info("Evaluating case: %s", case)
            result: TraceEvaluationResult = evaluate(
                evaluation_case=case,
                trace=agent_trace,
                agent_framework=config.framework,
            )
            for list_of_checkpoints in [
                result.checkpoint_results,
                result.direct_results,
                result.hypothesis_answer_results,
            ]:
                for checkpoint in list_of_checkpoints:
                    msg = (
                        f"Checkpoint: {checkpoint.criteria}\n"
                        f"\tPassed: {checkpoint.passed}\n"
                        f"\tReason: {checkpoint.reason}\n"
                        f"\tScore: {'%d/%d' % (checkpoint.points, checkpoint.points) if checkpoint.passed else '0/%d' % checkpoint.points}"
                    )
                    logger.info(msg)
            logger.info("==========================")
            logger.info("Overall Score: %d%%", 100 * result.score)
            logger.info("==========================")
            results.append(result)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = Path(output_dir) / f"{timestamp}_eval_case_{i}.json"
        with open(file_path, "w") as f:
            f.write(result.model_dump_json(indent=2))


def main():
    Fire(find_surf_spot)


if __name__ == "__main__":
    main()

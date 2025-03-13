import json
import sys
from textwrap import dedent
from typing import Any, Dict, List, Optional
from loguru import logger
from fire import Fire
from surf_spot_finder.agents.smolagents import run_smolagent
from surf_spot_finder.config import (
    DEFAULT_PROMPT,
    Config,
)
from surf_spot_finder.tracing import get_tracer_provider, setup_tracing
from surf_spot_finder.evaluation.utils import (
    extract_hypothesis_answer,
    verify_checkpoints,
    verify_hypothesis_answer,
)
from surf_spot_finder.evaluation.test_case import TestCase

logger.remove()
logger = logger.opt(ansi=True)
logger.add(sys.stdout, colorize=True, format="{message}")


def run_agent(test_case: TestCase) -> str:
    input_data = test_case.input
    logger.info("Loading config")
    config = Config(
        location=input_data.location,
        date=input_data.date,
        max_driving_hours=input_data.max_driving_hours,
        model_id=input_data.model_id,
        api_key_var=input_data.api_key_var,
        prompt=DEFAULT_PROMPT,
        json_tracer=input_data.json_tracer,
        api_base=input_data.api_base,
        agent_type=input_data.agent_type,
    )
    # project_name is a name + uuid
    project_name = "surf-spot-finder"

    logger.info("Setting up tracing")
    tracer_provider, telemetry_path = get_tracer_provider(
        project_name=project_name, json_tracer=config.json_tracer
    )
    setup_tracing(tracer_provider, agent_type=config.agent_type)
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
    return telemetry_path


def evaluate_telemetry(test_case: TestCase, telemetry_path: str) -> bool:
    # load the json file
    with open(telemetry_path, "r") as f:
        telemetry: List[Dict[str, Any]] = json.loads(f.read())
    logger.info(f"Telemetry loaded from {telemetry_path}")

    # Extract the final answer from the telemetry
    hypothesis_answer = extract_hypothesis_answer(telemetry)
    logger.info(
        f"""<yellow>Hypothesis Final answer extracted: {hypothesis_answer}</yellow>"""
    )
    # Verify agent behavior against checkpoints using llm-as-a-judge
    llm_judge = "openai/gpt-4o"
    checkpoint_results = verify_checkpoints(
        telemetry=telemetry,
        checkpoints=test_case.checkpoints,
        model=llm_judge,
    )

    hypothesis_answer_results = verify_hypothesis_answer(
        hypothesis_final_answer=hypothesis_answer,
        ground_truth_answer_dict=test_case.ground_truth,
        ground_truth_checkpoints=test_case.final_answer_criteria,
        model=llm_judge,
    )
    # Summarize results

    verification_results = checkpoint_results + hypothesis_answer_results
    failed_checks = [r for r in verification_results if not r.passed]
    passed_checks = [r for r in verification_results if r.passed]
    missed_points = sum([r.points for r in failed_checks])
    won_points = sum([r.points for r in passed_checks])
    if passed_checks:
        for check in passed_checks:
            message = dedent(
                f"""
                <green>Passed:
                - {check.criteria}
                - {check.reason}</green>"""
            )
            logger.info(message)
    if failed_checks:
        for check in failed_checks:
            message = dedent(
                f"""
                <red>Failed:
                - {check.criteria}
                - {check.reason}</red>"""
            )
            logger.error(message)
    else:
        logger.info("<green>All checkpoints passed!</green>")
    logger.info(
        f"<green>Passed checkpoints: {len(passed_checks)}/{len(verification_results)}</green>"
    )
    logger.info(
        f"<red>Failed checkpoints: {len(failed_checks)}/{len(verification_results)}</red>"
    )
    logger.info("<green>=====================================</green>")
    logger.info(f"<green>Score: {won_points}/{won_points + missed_points}</green>")
    logger.info("<green>=====================================</green>")


def evaluate(test_case_path: str, telemetry_path: Optional[str] = None) -> None:
    """
    Evaluate agent performance using either a provided telemetry file or by running the agent.

    Args:
        telemetry_path: Optional path to an existing telemetry file. If not provided,
                        the agent will be run to generate one.
    """
    test_case = TestCase.from_yaml(test_case_path)

    if telemetry_path is None:
        logger.info(
            "No telemetry path provided. Running agent to generate telemetry..."
        )
        telemetry_path = run_agent(test_case)
    else:
        logger.info(f"Using provided telemetry file: {telemetry_path}")
        logger.info(
            "For this to work, the telemetry file must align with the test case.",
        )

    evaluate_telemetry(test_case, telemetry_path)


def main():
    Fire(evaluate)


if __name__ == "__main__":
    main()

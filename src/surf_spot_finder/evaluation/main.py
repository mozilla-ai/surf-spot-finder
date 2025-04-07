import json
from textwrap import dedent
from typing import Any, Dict, List, Optional

from any_agent import AnyAgent
from any_agent.telemetry import TelemetryProcessor
from any_agent.tracing import setup_tracing
from fire import Fire

from surf_spot_finder.config import (
    Config,
)
from surf_spot_finder.evaluation.evaluators import (
    CheckpointEvaluator,
    HypothesisEvaluator,
    QuestionAnsweringSquadEvaluator,
)
from surf_spot_finder.evaluation.test_case import TestCase
from surf_spot_finder.evaluation.results_saver import save_evaluation_results
from surf_spot_finder.utils.logging import get_logger

# Replace the existing logger setup with the shared logger
logger = get_logger()


def run(agent_config: Config) -> str:
    logger.info("Setting up tracing")
    tracing_path = setup_tracing(agent_config.framework, "output")

    logger.info(f"Loading {agent_config.framework} agent")
    logger.info(f"{agent_config.managed_agents}")
    agent = AnyAgent.create(
        agent_framework=agent_config.framework,
        agent_config=agent_config.main_agent,
        managed_agents=agent_config.managed_agents,
    )

    query = agent_config.input_prompt_template.format(
        LOCATION=agent_config.location,
        MAX_DRIVING_HOURS=agent_config.max_driving_hours,
        DATE=agent_config.date,
    )
    logger.info(f"Running agent with query:\n{query}")
    agent.run(query)

    logger.success("Done!")

    return tracing_path


def evaluate_telemetry(test_case: TestCase, telemetry_path: str) -> bool:
    # load the json file
    with open(telemetry_path, "r") as f:
        telemetry: List[Dict[str, Any]] = json.loads(f.read())
    logger.info(f"Telemetry loaded from {telemetry_path}")

    agent_framework = TelemetryProcessor.determine_agent_framework(telemetry)

    # Extract the final answer from the telemetry
    processor = TelemetryProcessor.create(agent_framework)
    hypothesis_answer = processor.extract_hypothesis_answer(trace=telemetry)

    # Checkpoint evaluation
    checkpoint_evaluator = CheckpointEvaluator(model=test_case.llm_judge)
    checkpoint_results = checkpoint_evaluator.evaluate(
        telemetry=telemetry,
        checkpoints=test_case.checkpoints,
        processor=processor,
    )

    # Hypothesis answer evaluation
    hypothesis_evaluator = HypothesisEvaluator(model=test_case.llm_judge)
    hypothesis_answer_results = hypothesis_evaluator.evaluate(
        hypothesis_final_answer=hypothesis_answer,
        ground_truth_answer_dict=test_case.ground_truth,
        ground_truth_checkpoints=test_case.final_answer_criteria,
    )

    # Direct answer evaluation (new)
    if test_case.ground_truth:
        direct_evaluator = QuestionAnsweringSquadEvaluator()
        direct_results = direct_evaluator.evaluate(
            hypothesis_answer=hypothesis_answer,
            ground_truth_answer=test_case.ground_truth,
        )
    else:
        direct_results = []
    # Combine all results
    verification_results = (
        checkpoint_results + hypothesis_answer_results + direct_results
    )
    # Summarize results
    output_message = ""
    output_message += (
        f"""<yellow>Hypothesis Final answer extracted: {hypothesis_answer}</yellow>\n"""
    )
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
            output_message += message + "\n"
    if failed_checks:
        for check in failed_checks:
            message = dedent(
                f"""
                <red>Failed:
                - {check.criteria}
                - {check.reason}</red>"""
            )
            output_message += message + "\n"
    else:
        output_message += "<green>All checkpoints passed!</green>\n"
    output_message += f"<green>Passed checkpoints: {len(passed_checks)}</green>\n"
    output_message += f"<red>Failed checkpoints: {len(failed_checks)}</red>\n"
    output_message += "<green>=====================================</green>\n"
    output_message += (
        f"<green>Score: {won_points}/{won_points + missed_points}</green>\n"
    )
    output_message += "<green>=====================================</green>\n"
    logger.info(output_message)

    if won_points + missed_points == 0:
        raise ValueError("No points were defined in the test case")
    score = won_points / (won_points + missed_points) * 100

    # Save the evaluation results
    save_evaluation_results(
        test_case=test_case,
        output_path=test_case.output_path,
        output_message=output_message,
        telemetry_path=telemetry_path,
        hypothesis_answer=hypothesis_answer,
        passed_checks=len(passed_checks),
        failed_checks=len(failed_checks),
        score=score,
    )


def evaluate(
    test_case_path: str,
    agent_config_path: str = None,
    telemetry_path: Optional[str] = None,
) -> None:
    """
    Evaluate agent performance using either a provided telemetry file or by running the agent.

    Args:
        telemetry_path: Optional path to an existing telemetry file. If not provided,
                        the agent will be run to generate one.
    """
    test_case = TestCase.from_yaml(
        test_case_path=test_case_path, agent_config_path=agent_config_path
    )

    if telemetry_path is None:
        logger.info(
            "No telemetry path provided. Running agent to generate telemetry..."
        )
        assert (
            agent_config_path is not None
        ), "Agent config path must be provided if running agent"
        telemetry_path = run(test_case.agent_config)
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

import json
import os
import sys
from textwrap import dedent
from typing import Any, Dict, List, Optional
from loguru import logger
from fire import Fire
import pandas as pd
from surf_spot_finder.cli import find_surf_spot
from surf_spot_finder.config import (
    Config,
)
from surf_spot_finder.evaluation.telemetry import TelemetryProcessor
from surf_spot_finder.evaluation.evaluators import (
    CheckpointEvaluator,
    QuestionAnsweringSquadEvaluator,
    HypothesisEvaluator,
)
from surf_spot_finder.evaluation.test_case import TestCase

logger.remove()
logger = logger.opt(ansi=True)
logger.add(sys.stdout, colorize=True, format="{message}")


def run_agent(test_case: TestCase, agent_config_path: str) -> str:
    input_data = test_case.input

    logger.info("Loading config")
    config = Config.from_yaml(agent_config_path)
    config.location = input_data.location
    config.date = input_data.date
    config.max_driving_hours = input_data.max_driving_hours
    config.json_tracer = input_data.json_tracer
    return find_surf_spot(
        location=config.location,
        date=config.date,
        max_driving_hours=config.max_driving_hours,
        agent_type=config.agent_type,
        model_id=config.model_id,
        api_key_var=config.api_key_var,
        json_tracer=config.json_tracer,
        api_base=config.api_base,
        tools=config.tools,
        input_prompt_template=config.input_prompt_template,
    )


def evaluate_telemetry(test_case: TestCase, telemetry_path: str) -> bool:
    # load the json file
    with open(telemetry_path, "r") as f:
        telemetry: List[Dict[str, Any]] = json.loads(f.read())
    logger.info(f"Telemetry loaded from {telemetry_path}")

    agent_type = TelemetryProcessor.determine_agent_type(telemetry)

    # Extract the final answer from the telemetry
    processor = TelemetryProcessor.create(agent_type)
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
    direct_evaluator = QuestionAnsweringSquadEvaluator()
    direct_results = direct_evaluator.evaluate(
        hypothesis_answer=hypothesis_answer,
        ground_truth_answer=test_case.ground_truth,
    )

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
    # See if the test_case.output_path file exists.
    if os.path.exists(test_case.output_path):
        df = pd.read_json(test_case.output_path, orient="records", lines=True)
    else:
        df = pd.DataFrame()
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [
                    {
                        "test_case_path": test_case.test_case_path,
                        "output_message": output_message,
                        "telemetry_path": telemetry_path,
                        "hypothesis_answer": hypothesis_answer,
                        "passed_checks": len(passed_checks),
                        "failed_checks": len(failed_checks),
                        "score": round(
                            won_points / (won_points + missed_points) * 100, 2
                        ),
                    }
                ]
            ),
        ]
    )
    df.to_json(test_case.output_path, orient="records", lines=True)


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
    test_case = TestCase.from_yaml(test_case_path=test_case_path)

    if telemetry_path is None:
        logger.info(
            "No telemetry path provided. Running agent to generate telemetry..."
        )
        assert (
            agent_config_path is not None
        ), "Agent config path must be provided if running agent"
        telemetry_path = run_agent(test_case, agent_config_path)
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

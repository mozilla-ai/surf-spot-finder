import json
from loguru import logger

from surf_spot_finder.agents.smolagents import run_smolagent
from surf_spot_finder.config import (
    DEFAULT_PROMPT,
    Config,
)
from surf_spot_finder.tracing import setup_tracing
from .utils import extract_final_answer, verify_checkpoints, verify_final_answer
from .test_case import sample


# @pytest.mark.skipif(
#     "INTEGRATION_TESTS" not in os.environ or "OPENAI_API_KEY" not in os.environ,
#     reason="Integration tests require INTEGRATION_TESTS env var and OPENAI_API_KEY to be set"
# )
def test_surf_spot_finder():
    input_data = sample["input"]
    logger.info("Loading config")
    config = Config(
        location=input_data["location"],
        date=input_data["date"],
        max_driving_hours=input_data["max_driving_hours"],
        model_id=input_data["model_id"],
        api_key_var=input_data["api_key_var"],
        prompt=DEFAULT_PROMPT,
        json_tracer=input_data["json_tracer"],
        api_base=input_data["api_base"],
    )
    # project_name is a name + uuid
    project_name = "surf-spot-finder"

    logger.info("Setting up tracing")
    telemetry_path = setup_tracing(
        project_name=project_name, json_tracer=config.json_tracer
    )
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

    # load the json file
    with open(telemetry_path, "r") as f:
        telemetry = json.loads(f.read())
    logger.info("Telemetry loaded")

    # Extract the final answer from the telemetry
    final_answer = extract_final_answer(telemetry)
    logger.info(f"Final answer extracted: {final_answer}")

    # Verify agent behavior against checkpoints using llm-as-a-judge
    llm_judge = "openai/gpt-4o"
    checkpoint_results = verify_checkpoints(
        telemetry,
        final_answer,
        sample["checkpoints"],
        sample["output"],
        llm_judge,
    )

    final_answer_results = verify_final_answer(
        final_answer,
        sample["output"],
        sample["final_answer_criteria"],
        llm_judge,
    )
    # Summarize results

    verification_results = checkpoint_results + final_answer_results
    all_passed = all(result["passed"] for result in verification_results)
    failed_checks = [r for r in verification_results if not r["passed"]]

    # Log detailed results
    logger.info(f"All checkpoints passed: {all_passed}")
    if failed_checks:
        logger.error(
            f"Failed checkpoints: {len(failed_checks)}/{len(verification_results)}"
        )
        for check in failed_checks:
            logger.error(f"Failed: {check['criteria']} - {check['reason']}")

    # Assert that all checkpoints passed
    assert all_passed, f"{len(failed_checks)} checkpoints failed"

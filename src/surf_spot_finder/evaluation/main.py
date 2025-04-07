from typing import Optional
from fire import Fire
from pydantic import BaseModel, ConfigDict

from any_agent.evaluation.test_case import TestCase
from any_agent.evaluation.logging import get_logger
from any_agent.evaluation.evaluate import evaluate_telemetry

# Replace the existing logger setup with the shared logger
logger = get_logger()


class InputModel(BaseModel):
    """Input configuration for an evaluation test case"""

    model_config = ConfigDict(extra="forbid")
    location: str
    date: str
    max_driving_hours: int
    input_prompt_template: str | None = None


def evaluate(
    test_case_path: str,
    telemetry_path: Optional[str],
) -> None:
    """
    Evaluate agent performance using either a provided telemetry file or by running the agent.

    Args:
        telemetry_path: Optional path to an existing telemetry file. If not provided,
                        the agent will be run to generate one.
    """
    test_case = TestCase.from_yaml(test_case_path)

    logger.info(f"Using provided telemetry file: {telemetry_path}")
    logger.info(
        "For this to work, the telemetry file must align with the test case.",
    )

    evaluate_telemetry(test_case, telemetry_path)


def main():
    Fire(evaluate)


if __name__ == "__main__":
    main()

import json
from typing import Dict, List, Any, Optional
import re

from litellm import completion
from textwrap import dedent
from loguru import logger

from pydantic import BaseModel, ConfigDict
from surf_spot_finder.evaluation.telemetry_utils import extract_evidence
from surf_spot_finder.evaluation.test_case import CheckpointCriteria

from surf_spot_finder.agents import AgentType


def determine_agent_type(trace: List[Dict[str, Any]]) -> AgentType:
    """Determine the agent type based on the trace.
    These are not really stable ways to find it, because we're waiting on some
    reliable method for determining the agent type. This is a temporary solution.
    """
    for span in trace:
        if "langchain" in span.get("attributes", {}).get("input.value", ""):
            logger.info("Agent type is LANGCHAIN")
            return AgentType.LANGCHAIN
        if span.get("attributes", {}).get("smolagents.max_steps"):
            logger.info("Agent type is SMOLAGENTS")
            return AgentType.SMOLAGENTS
        # This is extremely fragile but there currently isn't
        # any specific key to indicate the agent type
        if span.get("name") == "response":
            logger.info("Agent type is OPENAI")
            return AgentType.OPENAI
    raise ValueError(
        "Could not determine agent type from trace, or agent type not supported"
    )


class EvaluationResult(BaseModel):
    """Represents the result of evaluating a criterion"""

    model_config = ConfigDict(extra="forbid")
    passed: bool
    reason: str
    criteria: str
    points: int


def evaluate_criterion(
    criteria: str,
    model: str,
    points: int,
    ground_truth_output: Optional[List[CheckpointCriteria] | Dict[str, Any]] = None,
    hypothesis_final_answer: Optional[str] = None,
    evidence: Optional[str] = None,
) -> EvaluationResult:
    """Evaluate a single criterion using LLM"""

    prompt = dedent(f"""
    Evaluate if the following criterion was met {"based on the provided evidence" if evidence else "in the agent's answer"}.

    Criterion: {criteria}
    """)

    if ground_truth_output:
        prompt += dedent(f"""
        Expected output: {json.dumps(ground_truth_output)}
        """)
    if hypothesis_final_answer:
        prompt += dedent(f"""
        Agent's  answer: {hypothesis_final_answer}
        """)

    if evidence:
        prompt += dedent(f"""
        Telemetry evidence:
        {evidence}
        """)

    prompt += f"""

    Based on the {"evidence" if evidence else "comparison between the expected output and the actual final answer"},
    was this criterion satisfied? Answer with:
    1. "passed": true or false
    2. "reason": Brief explanation for your decision
    """
    prompt += """
    Output valid JSON with these three fields only, in the format:
    ```json
    {
        "passed": true,
        "reason": "I have them"
    }
    ```
    """

    response = completion(model=model, messages=[{"role": "user", "content": prompt}])

    content = response.choices[0].message.content
    try:
        # Extract JSON from the response - looks for patterns like ```json {...} ``` or just {...}
        # Claude helped me with this one, regex is hard
        json_match = re.search(
            r"```(?:json)?\s*(\{.*?\})\s*```|(\{.*?\})", content, re.DOTALL
        )

        if json_match:
            # Use the first matching group that captured content
            json_str = next(group for group in json_match.groups() if group)
            evaluation = json.loads(json_str)
        else:
            # Fallback: try parsing the whole content as JSON
            evaluation = json.loads(content)

        evaluation["criteria"] = criteria
    except (json.JSONDecodeError, AttributeError, StopIteration) as e:
        evaluation = {
            "passed": False,
            "reason": f"Failed to evaluate due to parsing: {str(e)} \n Response: {content}",
            "criteria": criteria,
        }
    evaluation["points"] = points
    return EvaluationResult.model_validate(evaluation)


def verify_checkpoints(
    telemetry: List[Dict[str, Any]],
    checkpoints: List[CheckpointCriteria],
    model: str,
    agent_type: AgentType,
) -> List[EvaluationResult]:
    """Verify each checkpoint against the telemetry data using LLM
    These checkpoints do not take the ground truth or hyupothesis
    answers into account. They are only concerned with the trace and
    the specific criteria mentioned.
    """
    results = []

    evidence = extract_evidence(telemetry, agent_type)
    print(evidence)
    for checkpoint in checkpoints:
        criteria = checkpoint.criteria

        evaluation = evaluate_criterion(
            criteria=criteria,
            points=checkpoint.points,
            model=model,
            evidence=evidence,
        )

        results.append(evaluation)

    return results


def verify_hypothesis_answer(
    hypothesis_final_answer: str,
    ground_truth_answer_dict: Dict[str, Any],
    ground_truth_checkpoints: List[CheckpointCriteria],
    model: str,
) -> List[EvaluationResult]:
    """
    Verify if the final answer meets all specified criteria
    """
    results = []

    for criterion in ground_truth_checkpoints:
        evaluation = evaluate_criterion(
            criteria=criterion.criteria,
            points=criterion.points,
            ground_truth_output=ground_truth_answer_dict,
            hypothesis_final_answer=hypothesis_final_answer,
            model=model,
        )

        results.append(evaluation)

    return results

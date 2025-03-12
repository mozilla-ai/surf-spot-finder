import json
from typing import Dict, List, Any, Optional
import re

from litellm import completion


def extract_final_answer(telemetry: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extract the final answer from the telemetry data"""
    # Look for FinalAnswerTool spans
    for span in reversed(telemetry):
        if span.get("attributes", {}).get("openinference.span.kind") == "AGENT":
            final_answer = span.get("attributes", {}).get("output.value")
            return final_answer
    raise ValueError("Final answer not found in telemetry")


def evaluate_criterion(
    criteria: str,
    value: int,
    expected_output: Dict[str, Any],
    final_answer: Dict[str, Any],
    model: str,
    evidence: Optional[str] = None,
) -> Dict[str, Any]:
    """Evaluate a single criterion using LLM"""

    prompt = f"""
    Evaluate if the following {"checkpoint" if evidence else "criterion"} was met {"based on the provided evidence" if evidence else "in the agent's final answer"}.
    
    {"Checkpoint" if evidence else "Criterion"}: {criteria}
    Value: {value}
    
    Expected output: {json.dumps(expected_output)}
    
    Actual final answer: {json.dumps(final_answer)}
    """

    if evidence:
        prompt += f"""
        
        Telemetry evidence:
        {evidence}
        """

    prompt += f"""
    
    Based on the {"evidence" if evidence else "comparison between the expected output and the actual final answer"},
    was this {"checkpoint" if evidence else "criterion"} satisfied? Answer with:
    1. "passed": true or false
    2. "reason": Brief explanation for your decision
    3. "score": A score from 0 to {value} indicating how well the {"checkpoint" if evidence else "criterion"} was met
    """
    prompt += """
    Output valid JSON with these three fields only, in the format:
    ```json
    {
        "passed": true,
        "reason": "I have them",
        "score": 1
    }
    ```
    """

    response = completion(model=model, messages=[{"role": "user", "content": prompt}])

    try:
        content = response.choices[0].message.content

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
        evaluation["value"] = value
        return evaluation
    except (json.JSONDecodeError, AttributeError, StopIteration) as e:
        return {
            "passed": False,
            "reason": f"Failed to evaluate: {str(e)}",
            "score": 0,
            "criteria": criteria,
            "value": value,
        }


def verify_checkpoints(
    telemetry: List[Dict[str, Any]],
    final_answer: Dict[str, Any],
    checkpoints: List[Dict[str, Any]],
    expected_output: Dict[str, Any],
    model: str,
) -> List[Dict[str, Any]]:
    """Verify each checkpoint against the telemetry data using LLM"""
    results = []

    for checkpoint in checkpoints:
        criteria = checkpoint["criteria"]
        value = checkpoint["value"]
        evidence = extract_relevant_evidence(telemetry, criteria)

        evaluation = evaluate_criterion(
            criteria=criteria,
            value=value,
            expected_output=expected_output,
            final_answer=final_answer,
            model=model,
            evidence=evidence,
        )

        results.append(evaluation)

    return results


def verify_final_answer(
    final_answer: Dict[str, Any],
    expected_output: Dict[str, Any],
    criteria_list: List[Dict[str, Any]],
    model: str,
) -> List[Dict[str, Any]]:
    """
    Verify if the final answer meets all specified criteria
    """
    results = []

    for criterion in criteria_list:
        criteria = criterion["criteria"]
        value = criterion["value"]

        evaluation = evaluate_criterion(
            criteria=criteria,
            value=value,
            expected_output=expected_output,
            final_answer=final_answer,
            model=model,
        )

        results.append(evaluation)

    return results


def extract_relevant_evidence(telemetry: List[Dict[str, Any]], criteria: str) -> str:
    """Extract relevant telemetry evidence based on the checkpoint criteria
    TODO this is not a very robust implementation, since it requires knowledge about which tools have been
    implemented. We should abstract this so that it can dynamically figure out what tools may have been used
    and check for them appropriately."""
    evidence = ""

    # Look for evidence of tool usage
    if "DuckDuckGoSearchTool" in criteria:
        search_spans = [
            span for span in telemetry if span.get("name") == "DuckDuckGoSearchTool"
        ]
        evidence += f"Search tool was used {len(search_spans)} times.\n"
        for i, span in enumerate(search_spans):  # Limit to first 3 searches
            if "attributes" in span and "input.value" in span["attributes"]:
                try:
                    input_value = json.loads(span["attributes"]["input.value"])
                    if "kwargs" in input_value and "query" in input_value["kwargs"]:
                        evidence += (
                            f"Search query {i + 1}: {input_value['kwargs']['query']}\n"
                        )
                except (json.JSONDecodeError, TypeError):
                    pass

    # Look for evidence of website fetching
    if "fetched a website" in criteria:
        fetch_spans = [
            span
            for span in telemetry
            if span.get("attributes", {}).get("tool.name") == "fetch"
        ]
        evidence += f"Website fetch tool was used {len(fetch_spans)} times.\n"
        for i, span in enumerate(fetch_spans):  # Limit to first 3 fetches
            if "attributes" in span and "input.value" in span["attributes"]:
                try:
                    input_value = json.loads(span["attributes"]["input.value"])
                    if "kwargs" in input_value and "url" in input_value["kwargs"]:
                        evidence += (
                            f"Fetched URL {i + 1}: {input_value['kwargs']['url']}\n"
                        )
                except (json.JSONDecodeError, TypeError):
                    pass

    # Add general evidence about all tool calls
    tool_calls = {}
    for span in telemetry:
        if "name" in span and span["name"] not in tool_calls:
            tool_calls[span["name"]] = 1
        elif "name" in span:
            tool_calls[span["name"]] += 1

    evidence += "\nTool calls summary:\n"
    for tool, count in tool_calls.items():
        evidence += f"- {tool}: {count} call(s)\n"

    return evidence

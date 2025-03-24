from typing import Any, Dict, List
import json
from langchain_core.messages import BaseMessage
import re

from surf_spot_finder.agents import AgentType


def extract_hypothesis_answer(
    trace: List[Dict[str, Any]], agent_type: AgentType
) -> str:
    """Extract the hypothesis agent final answer from the trace"""
    for span in reversed(trace):
        if agent_type == AgentType.LANGCHAIN:
            if span["attributes"]["openinference.span.kind"] == "AGENT":
                content = span["attributes"]["output.value"]
                # If it's langchain, the actual content is a serialized langchain message that we need to extract.
                message = json.loads(content)["messages"][0]
                message = parse_generic_key_value_string(message)
                base_message = BaseMessage(**message, type="AGENT")
                print(base_message.text())
                return base_message.text()
        elif agent_type == AgentType.SMOLAGENTS:
            if span["attributes"]["openinference.span.kind"] == "AGENT":
                content = span["attributes"]["output.value"]
                # If it's langchain, the actual content is a serialized langchain message that we need to extract.
                return content
        elif agent_type == AgentType.OPENAI:
            # Looking for the final response that has the summary answer
            if (
                "attributes" in span
                and span.get("attributes", {}).get("openinference.span.kind") == "LLM"
            ):
                output_key = (
                    "llm.output_messages.0.message.contents.0.message_content.text"
                )
                if output_key in span["attributes"]:
                    return span["attributes"][output_key]
        else:
            raise ValueError(f"Unsupported agent type {agent_type}")
    raise ValueError("No agent final answer found in trace")


def parse_generic_key_value_string(text):
    """
    Parse a string that has items of a dict with key-value pairs separated by '='.
    Only splits on '=' signs, handling quoted strings properly.
    I think this is to compensate for a bug in openinference? https://github.com/Arize-ai/openinference/issues/1401
    """

    # Pattern to match key=value pairs, handling quoted values
    # This regex looks for word characters followed by = and then captures everything
    # until it finds another word character followed by = or the end of the string
    # Claude helped me with this one, regex is hard
    pattern = r"(\w+)=('.*?'|\".*?\"|[^'\"=]*?)(?=\s+\w+=|\s*$)"

    result = {}

    matches = re.findall(pattern, text)
    for key, value in matches:
        # Clean up the key
        key = key.strip()

        # Clean up the value - remove surrounding quotes if present
        if (value.startswith("'") and value.endswith("'")) or (
            value.startswith('"') and value.endswith('"')
        ):
            value = value[1:-1]

        # Store in result dictionary
        result[key] = value

    return result


def extract_evidence(telemetry: List[Dict[str, Any]], agent_type: AgentType) -> str:
    """Extract relevant telemetry evidence based on the agent type."""
    # Data extraction function for each agent type
    extractors = {
        AgentType.SMOLAGENTS: _extract_smolagents_data,
        AgentType.LANGCHAIN: _extract_langchain_data,
        AgentType.OPENAI: _extract_openai_data,
    }

    if agent_type not in extractors:
        raise ValueError(f"Unsupported agent type {agent_type}")

    # Extract raw data from telemetry
    calls = extractors[agent_type](telemetry)

    # Format data into a consistent structure
    return _format_evidence(calls, agent_type)


def _extract_smolagents_data(telemetry: List[Dict[str, Any]]) -> List[Dict]:
    """Extract LLM calls and tool calls from SmoL Agents telemetry."""
    calls = []

    for span in telemetry:
        # Skip spans without attributes
        if "attributes" not in span:
            continue

        attributes = span["attributes"]

        # Extract tool information
        if "tool.name" in attributes or span.get("name", "").startswith("SimpleTool"):
            tool_info = {
                "tool_name": attributes.get(
                    "tool.name", span.get("name", "Unknown tool")
                ),
                "status": "success"
                if span.get("status", {}).get("status_code") == "OK"
                else "error",
                "error": span.get("status", {}).get("description", None),
            }

            # Extract input if available
            if "input.value" in attributes:
                try:
                    input_value = json.loads(attributes["input.value"])
                    if "kwargs" in input_value:
                        # For SmoLAgents, the actual input is often in the kwargs field
                        tool_info["input"] = input_value["kwargs"]
                    else:
                        tool_info["input"] = input_value
                except (json.JSONDecodeError, TypeError):
                    tool_info["input"] = attributes["input.value"]

            # Extract output if available
            if "output.value" in attributes:
                try:
                    # Try to parse JSON output
                    output_value = (
                        json.loads(attributes["output.value"])
                        if isinstance(attributes["output.value"], str)
                        else attributes["output.value"]
                    )
                    tool_info["output"] = output_value
                except (json.JSONDecodeError, TypeError):
                    tool_info["output"] = attributes["output.value"]
            else:
                tool_info["output"] = "No output found"

            calls.append(tool_info)

        # Extract LLM calls to see reasoning
        elif "LiteLLMModel.__call__" in span.get("name", ""):
            # The LLM output may be in different places depending on the implementation
            output_content = None

            # Try to get the output from the llm.output_messages.0.message.content attribute
            if "llm.output_messages.0.message.content" in attributes:
                output_content = attributes["llm.output_messages.0.message.content"]

            # Or try to parse it from the output.value as JSON
            elif "output.value" in attributes:
                try:
                    output_value = json.loads(attributes["output.value"])
                    if "content" in output_value:
                        output_content = output_value["content"]
                except (json.JSONDecodeError, TypeError):
                    pass

            if output_content:
                calls.append(
                    {
                        "model": attributes.get("llm.model_name", "Unknown model"),
                        "output": output_content,
                        "type": "reasoning",
                    }
                )

    return calls


def _extract_langchain_data(telemetry: List[Dict[str, Any]]) -> List:
    """Extract LLM calls and tool calls from LangChain telemetry."""
    calls = []

    for span in telemetry:
        if "attributes" not in span:
            continue

        attributes = span.get("attributes", {})
        span_kind = attributes.get("openinference.span.kind", "")

        # Collect LLM calls
        if span_kind == "LLM" and "llm.output_messages.0.message.content" in attributes:
            llm_info = {
                "model": attributes.get("llm.model_name", "Unknown model"),
                "input": attributes.get("llm.input_messages.0.message.content", ""),
                "output": attributes.get("llm.output_messages.0.message.content", ""),
                "type": "reasoning",
            }
            calls.append(llm_info)

        # Try to find tool calls
        if "tool.name" in attributes or span.get("name", "").endswith("Tool"):
            tool_info = {
                "tool_name": attributes.get(
                    "tool.name", span.get("name", "Unknown tool")
                ),
                "status": "success"
                if span.get("status", {}).get("status_code") == "OK"
                else "error",
                "error": span.get("status", {}).get("description", None),
            }

            if "input.value" in attributes:
                try:
                    input_value = json.loads(attributes["input.value"])
                    tool_info["input"] = input_value
                except Exception:
                    tool_info["input"] = attributes["input.value"]

            if "output.value" in attributes:
                tool_info["output"] = parse_generic_key_value_string(
                    json.loads(attributes["output.value"])["output"]
                )["content"]

            calls.append(tool_info)

    return calls


def _extract_openai_data(telemetry: List[Dict[str, Any]]) -> list:
    """Extract LLM calls and tool calls from OpenAI telemetry."""
    calls = []

    for span in telemetry:
        if "attributes" not in span:
            continue

        attributes = span.get("attributes", {})
        span_kind = attributes.get("openinference.span.kind", "")

        # Collect LLM interactions - look for direct message content first
        if span_kind == "LLM":
            # Initialize the LLM info dictionary
            span_info = {}

            # Try to get input message
            input_key = "llm.input_messages.1.message.content"  # User message is usually at index 1
            if input_key in attributes:
                span_info["input"] = attributes[input_key]

            # Try to get output message directly
            output_content = None
            # Try in multiple possible locations
            for key in [
                "llm.output_messages.0.message.content",
                "llm.output_messages.0.message.contents.0.message_content.text",
            ]:
                if key in attributes:
                    output_content = attributes[key]
                    break

            # If we found direct output content, use it
            if output_content:
                span_info["output"] = output_content
                calls.append(span_info)
        elif span_kind == "TOOL":
            tool_name = attributes.get("tool.name", "Unknown tool")
            tool_output = attributes.get("output.value", "")

            span_info = {
                "tool_name": tool_name,
                "input": attributes.get("input.value", ""),
                "output": tool_output,
                "status": span.get("status", {}).get("status_code"),
            }
            span_info["input"] = json.loads(span_info["input"])

            calls.append(span_info)

    return calls


def _format_evidence(calls: List[Dict], agent_type: AgentType) -> str:
    """Format extracted data into a standardized output format."""
    evidence = f"## {agent_type.name} Agent Execution\n\n"

    for idx, call in enumerate(calls, start=1):
        evidence += f"### Call {idx}\n"

        # Truncate any values that are too long
        max_length = 400
        call = {
            k: (
                v[:max_length] + "..."
                if isinstance(v, str) and len(v) > max_length
                else v
            )
            for k, v in call.items()
        }

        # Use ensure_ascii=False to prevent escaping Unicode characters
        evidence += json.dumps(call, indent=2, ensure_ascii=False) + "\n\n"

    return evidence

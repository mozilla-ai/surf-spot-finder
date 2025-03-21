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
        if span["attributes"]["openinference.span.kind"] == "AGENT":
            content = span["attributes"]["output.value"]
            # If it's langchain, the actual content is a serialized langchain message that we need to extract.
            if agent_type == AgentType.LANGCHAIN:
                message = json.loads(content)["messages"][0]
                message = parse_generic_key_value_string(message)
                base_message = BaseMessage(**message, type="AGENT")
                print(base_message.text())
                return base_message.text()
            elif agent_type == AgentType.SMOLAGENTS:
                return content
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
        value = value.strip()
        if (value.startswith("'") and value.endswith("'")) or (
            value.startswith('"') and value.endswith('"')
        ):
            value = value[1:-1]

        # Store in result dictionary
        result[key] = value

    return result


def extract_evidence(telemetry: List[Dict[str, Any]], agent_type: AgentType) -> str:
    """Extract relevant telemetry evidence based on the agent type."""
    evidence = ""

    if agent_type == AgentType.SMOLAGENTS:
        # Extract information about all tools used
        tool_calls = []

        for span in telemetry:
            # Skip spans without attributes
            if "attributes" not in span:
                continue

            attributes = span.get("attributes", {})

            # Extract tool information
            if span.get("name", "").startswith("SimpleTool"):
                tool_info = {
                    "tool_name": attributes.get("tool.name", "Unknown tool"),
                    "status": "success"
                    if span.get("status", {}).get("status_code") == "OK"
                    else "error",
                    "error": span.get("status", {}).get("description", None),
                }

                # Extract input if available
                if "input.value" in attributes:
                    try:
                        input_value = json.loads(attributes["input.value"])
                        tool_info["input"] = input_value
                    except Exception:
                        tool_info["input"] = attributes["input.value"]

                # Extract output if available
                if "output.value" in attributes:
                    tool_info["output"] = attributes["output.value"]

                tool_calls.append(tool_info)

            # Extract LLM calls to see reasoning
            elif "LiteLLMModel.__call__" in span.get("name", ""):
                if "llm.output_messages.0.message.content" in attributes:
                    # We'll include just a sample of LLM reasoning for context
                    content = attributes["llm.output_messages.0.message.content"]
                    # Only include first 300 characters of each LLM call for brevity
                    tool_calls.append(
                        {
                            "type": "llm_reasoning",
                            "content": content[:300] + "..."
                            if len(content) > 300
                            else content,
                        }
                    )

        # Format the evidence as a string
        evidence += "## Tools Used and Their Results\n\n"
        for i, tool in enumerate(tool_calls):
            if tool.get("type") == "llm_reasoning":
                evidence += f"\n### LLM Reasoning Sample {i + 1}:\n{tool['content']}\n"
                continue

            evidence += f"\n### Tool Call {i + 1}: {tool.get('tool_name', 'Unknown')}\n"
            evidence += f"- Status: {tool.get('status', 'Unknown')}\n"

            if tool.get("error"):
                evidence += f"- Error: {tool.get('error')}\n"

            if "input" in tool:
                if isinstance(tool["input"], dict) and "kwargs" in tool["input"]:
                    evidence += (
                        f"- Input: {json.dumps(tool['input']['kwargs'], indent=2)}\n"
                    )
                else:
                    evidence += f"- Input: {tool['input']}\n"

            if "output" in tool:
                if isinstance(tool["output"], (dict, list)):
                    evidence += f"- Output: {json.dumps(tool['output'], indent=2)}\n"
                else:
                    output_str = str(tool["output"])
                    # Truncate long outputs
                    if len(output_str) > 500:
                        output_str = output_str[:500] + "...[truncated]"
                    evidence += f"- Output: {output_str}\n"
    elif agent_type == AgentType.LANGCHAIN:
        # Extract LLM calls and tool calls from LangChain telemetry
        llm_calls = []
        tool_calls = []

        for span in telemetry:
            if "attributes" not in span:
                continue

            attributes = span.get("attributes", {})
            span_kind = attributes.get("openinference.span.kind", "")

            # Collect LLM calls
            if (
                span_kind == "LLM"
                and "llm.output_messages.0.message.content" in attributes
            ):
                llm_info = {
                    "model": attributes.get("llm.model_name", "Unknown model"),
                    "input": attributes.get("llm.input_messages.0.message.content", ""),
                    "output": attributes.get(
                        "llm.output_messages.0.message.content", ""
                    ),
                    "tokens": {
                        "input": attributes.get("llm.token_count.prompt", 0),
                        "output": attributes.get("llm.token_count.completion", 0),
                        "total": attributes.get("llm.token_count.total", 0),
                    },
                }
                llm_calls.append(llm_info)

            # Try to find tool calls (may need to adjust based on actual structure)
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
                    tool_info["output"] = attributes["output.value"]

                tool_calls.append(tool_info)

        # Format the evidence
        evidence += "## LangChain Execution Flow\n\n"

        # Include the most relevant LLM reasoning
        if llm_calls:
            evidence += "### LLM Reasoning\n\n"
            # For brevity, include just the most significant LLM call
            # (usually the one with the longest output, which tends to have the most reasoning)
            most_significant = max(llm_calls, key=lambda x: len(x["output"]))

            evidence += f"**Model**: {most_significant['model']}\n\n"
            evidence += f"**Input**: {most_significant['input'][:150]}...\n\n"

            # Include full reasoning but format it for readability
            evidence += "**Reasoning**:\n"
            reasoning = most_significant["output"]
            # Truncate if extremely long
            if len(reasoning) > 1000:
                evidence += f"{reasoning[:1000]}...\n[truncated - full reasoning was {len(reasoning)} characters]\n\n"
            else:
                evidence += f"{reasoning}\n\n"

            evidence += (
                f"**Token Usage**: {most_significant['tokens']['total']} total tokens "
                f"({most_significant['tokens']['input']} input, {most_significant['tokens']['output']} output)\n\n"
            )

        # Include tool calls if detected
        if tool_calls:
            evidence += "### Tool Calls\n\n"
            for i, tool in enumerate(tool_calls):
                evidence += f"**Tool {i + 1}**: {tool.get('tool_name', 'Unknown')}\n"
                evidence += f"- Status: {tool.get('status', 'Unknown')}\n"

                if tool.get("error"):
                    evidence += f"- Error: {tool.get('error')}\n"

                if "input" in tool:
                    input_str = str(tool["input"])
                    if len(input_str) > 200:
                        input_str = input_str[:200] + "...[truncated]"
                    evidence += f"- Input: {input_str}\n"

                if "output" in tool:
                    output_str = str(tool["output"])
                    if len(output_str) > 200:
                        output_str = output_str[:200] + "...[truncated]"
                    evidence += f"- Output: {output_str}\n\n"
    else:
        raise ValueError(f"Unsupported agent type {agent_type}")
    return evidence

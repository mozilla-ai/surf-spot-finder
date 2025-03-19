import importlib
import os
from typing import Optional

from loguru import logger

from surf_spot_finder.prompts.smolagents import SYSTEM_PROMPT

try:
    from smolagents import (
        CodeAgent,
        LiteLLMModel,
        ToolCollection,
    )

    smolagents_available = True
except ImportError:
    smolagents_available = None


@logger.catch(reraise=True)
def run_smolagent(
    model_id: str,
    prompt: str,
    api_key_var: Optional[str] = None,
    api_base: Optional[str] = None,
    tools: Optional[list[str]] = None,
) -> CodeAgent:
    """
    Create and configure a Smolagents CodeAgent with the specified model.

    See https://docs.litellm.ai/docs/providers for details on available LiteLLM providers.

    Args:
        model_id (str): Model identifier using LiteLLM syntax (e.g., 'openai/o1', 'anthropic/claude-3-sonnet')
        prompt (str): Prompt to provide to the model
        api_key_var (Optional[str]): Name of environment variable containing the API key
        api_base (Optional[str]): Custom API base URL, if needed for non-default endpoints

    Returns:
        CodeAgent: Configured agent ready to process requests

    Example:

        >>> agent = run_smolagent("anthropic/claude-3-haiku", "my prompt here", "ANTHROPIC_API_KEY", None, None)
        >>> agent.run("Find surf spots near San Diego")
    """
    if not smolagents_available:
        raise ImportError("You need to `pip install smolagents` to use this agent")

    if tools is None:
        tools = [
            "smolagents.DuckDuckGoSearchTool",
            "smolagents.VisitWebpageTool",
            "smolagents.PythonInterpreterTool",
        ]

    imported_tools = []
    mcp_tool = None
    for tool in tools:
        if "mcp" in tool:
            mcp_tool = tool
        else:
            module, func = tool.rsplit(".", 1)
            module = importlib.import_module(module)
            tool = getattr(module, func)
            imported_tools.append(tool())

    model = LiteLLMModel(
        model_id=model_id,
        api_base=api_base if api_base else None,
        api_key=os.environ[api_key_var] if api_key_var else None,
    )

    if mcp_tool:
        from mcp import StdioServerParameters

        # We could easily use any of the MCPs at https://github.com/modelcontextprotocol/servers
        # or at https://glama.ai/mcp/servers
        # or at https://smithery.ai/
        server_parameters = StdioServerParameters(
            command="docker",
            args=["run", "-i", "--rm", mcp_tool],
            env={**os.environ},
        )
        # https://huggingface.co/docs/smolagents/v1.10.0/en/reference/tools#smolagents.ToolCollection.from_mcp
        with ToolCollection.from_mcp(server_parameters) as tool_collection:
            agent = CodeAgent(
                tools=imported_tools + tool_collection.tools,
                prompt_templates={"system_prompt": SYSTEM_PROMPT},
                model=model,
                add_base_tools=False,  # Turn this on if you want to let it run python code as it sees fit
            )
            agent.run(prompt)
    else:
        agent = CodeAgent(
            tools=imported_tools,
            prompt_templates={"system_prompt": SYSTEM_PROMPT},
            model=model,
        )
        agent.run(prompt)

    return agent

import os
from typing import Optional, TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from smolagents import CodeAgent


@logger.catch(reraise=True)
def load_smolagent(model_id: str, api_key_var: Optional[str]) -> "CodeAgent":
    """ """
    from smolagents import (
        CodeAgent,
        ToolCollection,
        DuckDuckGoSearchTool,
        VisitWebpageTool,
        LiteLLMModel,
    )
    from mcp import StdioServerParameters

    model = LiteLLMModel(
        model_id=model_id,
        api_key_var=os.environ[api_key_var] if api_key_var else None,
    )

    if "GOOGLE_MAPS_API_KEY" in os.environ:
        # We could easily use any of the MCPs at https://github.com/modelcontextprotocol/servers
        # or at https://glama.ai/mcp/servers
        # or at https://smithery.ai/
        # https://github.com/modelcontextprotocol/servers/tree/main/src/google-maps
        server_parameters = StdioServerParameters(
            command="npx",
            args=["@modelcontextprotocol/server-google-maps"],
            env={**os.environ},
        )
        # https://huggingface.co/docs/smolagents/v1.10.0/en/reference/tools#smolagents.ToolCollection.from_mcp
        with ToolCollection.from_mcp(server_parameters) as tool_collection:
            agent = CodeAgent(
                tools=[
                    *tool_collection.tools,
                    DuckDuckGoSearchTool(),
                    VisitWebpageTool(),
                ],
                model=model,
                add_base_tools=True,
                additional_authorized_imports=["json"],
            )
    else:
        logger.debug(
            "GOOGLE_MAPS_api_key_var not set, running without Google Maps tool"
        )
        agent = CodeAgent(
            tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
            model=model,
            add_base_tools=True,
            additional_authorized_imports=["json"],
        )

    return agent

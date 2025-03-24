import inspect
import importlib

from loguru import logger

try:
    from langchain.chat_models import init_chat_model
    from langchain_core.messages import HumanMessage
    from langchain_core.tools import BaseTool, tool
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.prebuilt import create_react_agent

    langchain_available = True
except ImportError:
    langchain_available = False

DEFAULT_RECURSION_LIMIT = 50


@logger.catch(reraise=True)
def run_lanchain_agent(
    model_id: str, prompt: str, tools: list[str] | None = None, **kwargs
):
    """Runs an langchain ReAct agent with the given prompt and configuration.

    Uses [create_react_agent](https://langchain-ai.github.io/langgraph/reference/prebuilt/#langgraph.prebuilt.chat_agent_executor.create_react_agent).

    Args:
        model_id: The ID of the model to use.
            See [init_chat_model](https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html).
        prompt: The prompt to be given to the agent.
    """
    if not langchain_available:
        raise ImportError(
            "You need to `pip install langchain langgraph` to use this agent"
        )

    if tools is None:
        tools = [
            "surf_spot_finder.tools.search_web",
            "surf_spot_finder.tools.visit_webpage",
        ]

    imported_tools = []
    for imported_tool in tools:
        module, func = imported_tool.rsplit(".", 1)
        module = importlib.import_module(module)
        imported_tool = getattr(module, func)
        if inspect.isclass(imported_tool):
            imported_tool = imported_tool()
        if not isinstance(imported_tool, BaseTool):
            imported_tool = tool(imported_tool)
        imported_tools.append((imported_tool))
    if "/" in model_id:
        model_provider, model_id = model_id.split("/")
        model = init_chat_model(model_id, model_provider=model_provider)
    else:
        model = init_chat_model(model_id)
    agent = create_react_agent(
        model=model, tools=imported_tools, checkpointer=MemorySaver()
    )
    for step in agent.stream(
        {"messages": [HumanMessage(content=prompt)]},
        {
            "configurable": {"thread_id": "abc123"},
            "recursion_limit": DEFAULT_RECURSION_LIMIT,
        },
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()

import os
from typing import Optional

from loguru import logger

from surf_spot_finder.prompts.openai import (
    SINGLE_AGENT_SYSTEM_PROMPT,
    MULTI_AGENT_SYSTEM_PROMPT,
)
from surf_spot_finder.tools.wrappers import import_and_wrap_tools, wrap_tool_openai


try:
    from agents import (
        Agent,
        AsyncOpenAI,
        OpenAIChatCompletionsModel,
        Runner,
        RunResult,
        function_tool,
    )

    agents_available = True
except ImportError:
    agents_available = None

DEFAULT_MAX_TURNS = 20


@logger.catch(reraise=True)
def run_openai_agent(
    model_id: str,
    prompt: str,
    name: str = "surf-spot-finder",
    instructions: Optional[str] = SINGLE_AGENT_SYSTEM_PROMPT,
    api_key_var: Optional[str] = None,
    api_base: Optional[str] = None,
    tools: Optional[list[str]] = None,
    max_turns: Optional[int] = DEFAULT_MAX_TURNS,
) -> RunResult:
    """Runs an OpenAI agent with the given prompt and configuration.

    It leverages the 'agents' library to create and manage the agent
    execution.

    See https://openai.github.io/openai-agents-python/ref/agent/ for more details.


    Args:
        model_id (str): The ID of the OpenAI model to use (e.g., "gpt4o").
            See https://platform.openai.com/docs/api-reference/models.
        prompt (str): The prompt to be given to the agent.
        name (str, optional): The name of the agent. Defaults to "surf-spot-finder".
        instructions (Optional[str], optional): Initial instructions to give the agent.
            Defaults to [SINGLE_AGENT_SYSTEM_PROMPT][surf_spot_finder.prompts.openai.SINGLE_AGENT_SYSTEM_PROMPT].
        api_key_var (Optional[str], optional): The name of the environment variable
            containing the OpenAI API key. If provided, along with `base_url`, an
            external OpenAI client will be used. Defaults to None.
        api_base (Optional[str], optional): The base URL for the OpenAI API.
            Required if `api_key_var` is provided to use an external OpenAI client.
            Defaults to None.


    Returns:
        RunResult: A RunResult object containing the output of the agent run.
            See https://openai.github.io/openai-agents-python/ref/result/#agents.result.RunResult.
    """
    if not agents_available:
        raise ImportError("You need to `pip install openai-agents` to use this agent")

    if tools is None:
        tools = [
            "surf_spot_finder.tools.search_web",
            "surf_spot_finder.tools.visit_webpage",
        ]

    imported_tools = import_and_wrap_tools(tools, wrap_tool_openai)

    if api_key_var and api_base:
        external_client = AsyncOpenAI(
            api_key=os.environ[api_key_var],
            base_url=api_base,
        )
        agent = Agent(
            name=name,
            instructions=instructions,
            model=OpenAIChatCompletionsModel(
                model=model_id,
                openai_client=external_client,
            ),
            tools=imported_tools,
        )
    else:
        agent = Agent(
            model=model_id,
            instructions=instructions,
            name=name,
            tools=imported_tools,
        )
    result = Runner.run_sync(starting_agent=agent, input=prompt, max_turns=max_turns)
    logger.info(result.final_output)
    return result


@logger.catch(reraise=True)
def run_openai_multi_agent(
    model_id: str,
    prompt: str,
    name: str = "surf-spot-finder",
    instructions: Optional[str] = MULTI_AGENT_SYSTEM_PROMPT,
    max_turns: Optional[int] = DEFAULT_MAX_TURNS,
    **kwargs,
) -> RunResult:
    """Runs multiple OpenAI agents orchestrated by a main agent.

    It leverages the 'agents' library to create and manage the agent
    execution.

    See https://openai.github.io/openai-agents-python/ref/agent/ for more details.


    Args:
        model_id (str): The ID of the OpenAI model to use (e.g., "gpt4o").
            See https://platform.openai.com/docs/api-reference/models.
        prompt (str): The prompt to be given to the agent.
        name (str, optional): The name of the main agent. Defaults to "surf-spot-finder".
        instructions (Optional[str], optional): Initial instructions to give the agent.
            Defaults to [MULTI_AGENT_SYSTEM_PROMPT][surf_spot_finder.prompts.openai.MULTI_AGENT_SYSTEM_PROMPT].

    Returns:
        RunResult: A RunResult object containing the output of the agent run.
            See https://openai.github.io/openai-agents-python/ref/result/#agents.result.RunResult.
    """
    if not agents_available:
        raise ImportError("You need to `pip install openai-agents` to use this agent")

    from surf_spot_finder.tools import (
        ask_user_verification,
        show_final_answer,
        show_plan,
        search_web,
        visit_webpage,
    )

    user_verification_agent = Agent(
        model=model_id,
        instructions="Interact with the user by showing information and asking for verification.",
        name="user-verification-agent",
        tools=[function_tool(ask_user_verification), function_tool(show_plan)],
    )

    search_web_agent = Agent(
        model=model_id,
        instructions="Find relevant information about the provided task by using your tools.",
        name="search-web-agent",
        tools=[function_tool(search_web), function_tool(visit_webpage)],
    )

    communication_agent = Agent(
        model=model_id,
        instructions="Communicate the final answer to the user.",
        name="communication-agent",
        tools=[function_tool(show_final_answer)],
    )

    main_agent = Agent(
        model=model_id,
        instructions=instructions,
        name=name,
        handoffs=[communication_agent],
        tools=[
            search_web_agent.as_tool(
                tool_name="search_web_with_agent",
                tool_description=search_web_agent.instructions,
            ),
            user_verification_agent.as_tool(
                tool_name="ask_user_verification_with_agent",
                tool_description=user_verification_agent.instructions,
            ),
        ],
    )

    result = Runner.run_sync(
        starting_agent=main_agent, input=prompt, max_turns=max_turns
    )
    logger.info(result.final_output)
    return result

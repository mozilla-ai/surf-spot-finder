import os
from typing import Optional

from agents import (
    Agent,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    Runner,
    RunResult,
    function_tool,
)
from loguru import logger
from smolagents import (
    DuckDuckGoSearchTool,
    VisitWebpageTool,
    FinalAnswerTool,
)


@function_tool
def search_web(query: str) -> str:
    """Performs a duckduckgo web search based on your query (think a Google search) then returns the top search results.

    Args:
        query: The search query to perform.
    """
    logger.debug(f"Calling search_web: {query}")
    search_tool = DuckDuckGoSearchTool()
    return search_tool.forward(query)


@function_tool
def visit_webpage(url: str) -> str:
    """Visits a webpage at the given url and reads its content as a markdown string. Use this to browse webpages.

    Args:
        url: The url of the webpage to visit.
    """
    logger.debug(f"Calling visit_webpage: {url}")
    visit_tool = VisitWebpageTool()
    return visit_tool.forward(url)


@function_tool
def final_answer(answer: str) -> str:
    """Provides a final answer to the given problem.

    Args:
        answer: The answer to the problem.
    """
    logger.debug("Calling final_answer")
    final_answer_tool = FinalAnswerTool()
    return final_answer_tool.forward(answer)


@function_tool
def user_verification(query: str) -> str:
    """Asks user to verify the given `query`.

    Args:
        query: The question that requires verification.
    """
    logger.debug("Calling user_verification")
    return input(f"{query} => Type your answer here:")


@logger.catch(reraise=True)
def run_openai_agent(
    model_id: str,
    prompt: str,
    name: str = "surf-spot-finder",
    instructions: Optional[str] = None,
    api_key_var: Optional[str] = None,
    base_url: Optional[str] = None,
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
            Defaults to None.
        api_key_var (Optional[str], optional): The name of the environment variable
            containing the OpenAI API key. If provided, along with `base_url`, an
            external OpenAI client will be used. Defaults to None.
        base_url (Optional[str], optional): The base URL for the OpenAI API.
            Required if `api_key_var` is provided to use an external OpenAI client.
            Defaults to None.

    Returns:
        RunResult: A RunResult object containing the output of the agent run.
            See https://openai.github.io/openai-agents-python/ref/result/#agents.result.RunResult.
    """

    if api_key_var and base_url:
        external_client = AsyncOpenAI(
            api_key=os.environ[api_key_var],
            base_url=base_url,
        )
        agent = Agent(
            name=name,
            instructions=instructions,
            model=OpenAIChatCompletionsModel(
                model=model_id,
                openai_client=external_client,
            ),
            tools=[search_web, visit_webpage],
        )
    else:
        agent = Agent(
            model=model_id,
            instructions=instructions,
            name=name,
            tools=[search_web, visit_webpage],
        )
    result = Runner.run_sync(agent, prompt)
    logger.info(result.final_output)
    return result


DEFAULT_MULTIAGENT_INSTRUCTIONS = """
You will be asked to perform a task.

Always follow this steps:

First, before solving the task, look at the available agent/tools and plan a sequence of actions using the available tools.
Second, show the plan of actions and ask for user verification. If the user does not verify the plan, come up with a better plan.
Third, execute the plan using the available tools, until you get a final answer.

Once you get a final answer, show it and ask for user verification.  If the user does not verify the answer, come up with a better answer.

Finally, use the available handoff tool (`transfer_to_<agent_name>`) to communicate it to the user.
"""


@logger.catch(reraise=True)
def run_openai_multi_agent(
    model_id: str,
    prompt: str,
    name: str = "surf-spot-finder",
    instructions: Optional[str] = DEFAULT_MULTIAGENT_INSTRUCTIONS,
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
            Defaults to [DEFAULT_MULTIAGENT_INSTRUCTIONS][surf_spot_finder.agents.openai.DEFAULT_MULTIAGENT_INSTRUCTIONS].

    Returns:
        RunResult: A RunResult object containing the output of the agent run.
            See https://openai.github.io/openai-agents-python/ref/result/#agents.result.RunResult.
    """
    user_verification_agent = Agent(
        model=model_id,
        instructions="Display the current output to the user, then ask for verification.",
        name="user-verification-agent",
        tools=[user_verification],
    )

    search_web_agent = Agent(
        model=model_id,
        instructions="Find relevant information about the provided task by combining web searches with visiting webpages.",
        name="search-web-agent",
        tools=[search_web, visit_webpage],
    )

    communication_agent = Agent(
        model=model_id,
        instructions=None,
        name="communication-agent",
        tools=[final_answer],
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

    result = Runner.run_sync(main_agent, prompt)
    logger.info(result.final_output)
    return result

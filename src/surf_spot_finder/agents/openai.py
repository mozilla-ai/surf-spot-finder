import os
from typing import Optional, TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from agents import RunResult


@logger.catch(reraise=True)
def run_openai_agent(
    model_id: str,
    prompt: str,
    name: str = "surf-spot-finder",
    instructions: Optional[str] = None,
    api_key_var: Optional[str] = None,
    base_url: Optional[str] = None,
) -> "RunResult":
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
    from agents import (
        Agent,
        AsyncOpenAI,
        OpenAIChatCompletionsModel,
        Runner,
        WebSearchTool,
    )

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
            tools=[WebSearchTool()],
        )
    else:
        agent = Agent(
            model=model_id,
            instructions=instructions,
            name=name,
            tools=[WebSearchTool()],
        )
    result = Runner.run_sync(agent, prompt)
    logger.info(result.final_output)
    return result

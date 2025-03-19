from unittest.mock import patch, MagicMock

from surf_spot_finder.agents.langchain import (
    run_lanchain_agent,
)
from surf_spot_finder.tools import (
    search_web,
    visit_webpage,
)


def test_run_langchain_agent_default():
    model_mock = MagicMock()
    create_mock = MagicMock()
    agent_mock = MagicMock()
    create_mock.return_value = agent_mock
    memory_mock = MagicMock()
    tool_mock = MagicMock()

    with (
        patch("surf_spot_finder.agents.langchain.create_react_agent", create_mock),
        patch("surf_spot_finder.agents.langchain.init_chat_model", model_mock),
        patch("surf_spot_finder.agents.langchain.MemorySaver", memory_mock),
        patch("surf_spot_finder.agents.langchain.tool", tool_mock),
    ):
        run_lanchain_agent("gpt-4o", "Test prompt")
        model_mock.assert_called_once_with("gpt-4o")
        create_mock.assert_called_once_with(
            model=model_mock.return_value,
            tools=[tool_mock(search_web), tool_mock(visit_webpage)],
            checkpointer=memory_mock.return_value,
        )
        agent_mock.stream.assert_called_once()

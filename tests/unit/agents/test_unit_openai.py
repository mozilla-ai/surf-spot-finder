import os
import pytest
from unittest.mock import patch, MagicMock, ANY

from surf_spot_finder.agents.openai import run_openai_agent


@pytest.fixture
def mock_agents_module():
    agents_mocks = {
        name: MagicMock()
        for name in (
            "Agent",
            "AsyncOpenAI",
            "OpenAIChatCompletionsModel",
            "Runner",
            "WebSearchTool",
        )
    }
    with patch.dict(
        "sys.modules",
        {
            "agents": MagicMock(**agents_mocks),
        },
    ):
        yield agents_mocks


def test_run_openai_agent_default(mock_agents_module):
    run_openai_agent("gpt-4o", "Test prompt")
    mock_agents_module["Agent"].assert_called_once_with(
        model="gpt-4o",
        instructions=None,
        name="surf-spot-finder",
        tools=ANY,
    )


def test_run_openai_agent_base_url_and_api_key_var(mock_agents_module):
    with patch.dict(os.environ, {"TEST_API_KEY": "test-key-12345"}):
        run_openai_agent(
            "gpt-4o", "Test prompt", base_url="FOO", api_key_var="TEST_API_KEY"
        )
        mock_agents_module["AsyncOpenAI"].assert_called_once_with(
            api_key="test-key-12345",
            base_url="FOO",
        )
        mock_agents_module["OpenAIChatCompletionsModel"].assert_called_once()


def test_run_smolagent_environment_error():
    """Test that passing a bad api_key_var throws an error"""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(KeyError, match="MISSING_KEY"):
            run_openai_agent(
                "test-model", "Test prompt", base_url="FOO", api_key_var="MISSING_KEY"
            )

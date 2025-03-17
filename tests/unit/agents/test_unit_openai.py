import os
import pytest
from unittest.mock import patch, MagicMock, ANY

from surf_spot_finder.agents.openai import (
    final_answer,
    run_openai_agent,
    run_openai_multi_agent,
    search_web,
    user_verification,
    visit_webpage,
    DEFAULT_MULTIAGENT_INSTRUCTIONS,
)


def test_run_openai_agent_default():
    mock_agent = MagicMock()

    with (
        patch("surf_spot_finder.agents.openai.Agent", mock_agent),
        patch("surf_spot_finder.agents.openai.Runner", MagicMock()),
    ):
        run_openai_agent("gpt-4o", "Test prompt")
        mock_agent.assert_called_once_with(
            model="gpt-4o",
            instructions=None,
            name="surf-spot-finder",
            tools=[search_web, visit_webpage],
        )


def test_run_openai_agent_base_url_and_api_key_var():
    async_openai_mock = MagicMock()
    openai_chat_completions_model = MagicMock()
    with (
        patch("surf_spot_finder.agents.openai.Agent", MagicMock()),
        patch("surf_spot_finder.agents.openai.Runner", MagicMock()),
        patch("surf_spot_finder.agents.openai.AsyncOpenAI", async_openai_mock),
        patch(
            "surf_spot_finder.agents.openai.OpenAIChatCompletionsModel",
            openai_chat_completions_model,
        ),
        patch.dict(os.environ, {"TEST_API_KEY": "test-key-12345"}),
    ):
        run_openai_agent(
            "gpt-4o", "Test prompt", base_url="FOO", api_key_var="TEST_API_KEY"
        )
        async_openai_mock.assert_called_once_with(
            api_key="test-key-12345",
            base_url="FOO",
        )
        openai_chat_completions_model.assert_called_once()


def test_run_openai_environment_error():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(KeyError, match="MISSING_KEY"):
            run_openai_agent(
                "test-model", "Test prompt", base_url="FOO", api_key_var="MISSING_KEY"
            )


def test_run_openai_multiagent():
    mock_agent = MagicMock()

    with (
        patch("surf_spot_finder.agents.openai.Agent", mock_agent),
        patch("surf_spot_finder.agents.openai.Runner", MagicMock()),
    ):
        run_openai_multi_agent("gpt-4o", "Test prompt")
        mock_agent.assert_any_call(
            model="gpt-4o",
            instructions="Display the current output to the user, then ask for verification.",
            name="user-verification-agent",
            tools=[user_verification],
        )

        mock_agent.assert_any_call(
            model="gpt-4o",
            instructions="Find relevant information about the provided task by combining web searches with visiting webpages.",
            name="search-web-agent",
            tools=[search_web, visit_webpage],
        )

        mock_agent.assert_any_call(
            model="gpt-4o",
            instructions=None,
            name="communication-agent",
            tools=[final_answer],
        )

        mock_agent.assert_any_call(
            model="gpt-4o",
            instructions=DEFAULT_MULTIAGENT_INSTRUCTIONS,
            name="surf-spot-finder",
            # TODO: add more elaborated checks
            handoffs=ANY,
            tools=ANY,
        )

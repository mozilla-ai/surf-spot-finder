import os
import pytest
from unittest.mock import patch, MagicMock
import contextlib

from surf_spot_finder.agents.smolagents import run_smolagent


@pytest.fixture
def common_patches():
    litellm_model_mock = MagicMock()
    code_agent_mock = MagicMock()
    patch_context = contextlib.ExitStack()
    mock_tool_collection = MagicMock()

    mock_tool_collection.from_mcp.return_value.__enter__.return_value = (
        mock_tool_collection
    )
    mock_tool_collection.from_mcp.return_value.__exit__.return_value = None
    mock_tool_collection.tools = ["mock_tool"]
    patch_context.enter_context(
        patch("surf_spot_finder.agents.smolagents.StdioServerParameters", MagicMock())
    )
    patch_context.enter_context(
        patch("surf_spot_finder.agents.smolagents.CodeAgent", code_agent_mock)
    )
    patch_context.enter_context(
        patch("surf_spot_finder.agents.smolagents.LiteLLMModel", litellm_model_mock)
    )
    patch_context.enter_context(
        patch("surf_spot_finder.agents.smolagents.ToolCollection", mock_tool_collection)
    )
    yield patch_context, litellm_model_mock, code_agent_mock, mock_tool_collection
    patch_context.close()


def test_run_smolagent_with_api_key_var(common_patches):
    patch_context, litellm_model_mock, code_agent_mock, *_ = common_patches

    with patch_context, patch.dict(os.environ, {"TEST_API_KEY": "test-key-12345"}):
        run_smolagent("openai/gpt-4", "Test prompt", api_key_var="TEST_API_KEY")

        litellm_model_mock.assert_called()
        model_call_kwargs = litellm_model_mock.call_args[1]
        assert model_call_kwargs["model_id"] == "openai/gpt-4"
        assert model_call_kwargs["api_key"] == "test-key-12345"
        assert model_call_kwargs["api_base"] is None

        code_agent_mock.assert_called_once()
        code_agent_mock.return_value.run.assert_called_once_with("Test prompt")


def test_run_smolagent_with_custom_api_base(common_patches):
    patch_context, litellm_model_mock, *_ = common_patches

    with patch_context, patch.dict(os.environ, {"TEST_API_KEY": "test-key-12345"}):
        run_smolagent(
            "anthropic/claude-3-sonnet",
            "Test prompt",
            api_key_var="TEST_API_KEY",
            api_base="https://custom-api.example.com",
        )
        last_call = litellm_model_mock.call_args_list[-1]

        assert last_call[1]["model_id"] == "anthropic/claude-3-sonnet"
        assert last_call[1]["api_key"] == "test-key-12345"
        assert last_call[1]["api_base"] == "https://custom-api.example.com"


def test_run_smolagent_without_api_key(common_patches):
    patch_context, litellm_model_mock, *_ = common_patches

    with patch_context:
        run_smolagent("ollama_chat/deepseek-r1", "Test prompt")

    last_call = litellm_model_mock.call_args_list[-1]
    assert last_call[1]["model_id"] == "ollama_chat/deepseek-r1"
    assert last_call[1]["api_key"] is None


def test_run_smolagent_environment_error():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(KeyError, match="MISSING_KEY"):
            run_smolagent("test-model", "Test prompt", api_key_var="MISSING_KEY")

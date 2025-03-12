import os
import pytest
from unittest.mock import patch, MagicMock

from surf_spot_finder.agents.smolagents import run_smolagent


@pytest.fixture
def mock_smolagents_imports():
    """Mock the smolagents imports to avoid actual instantiation."""
    mock_code_agent = MagicMock()
    mock_ddg_tool = MagicMock()
    mock_litellm_model = MagicMock()
    mock_tool_collection = MagicMock()

    # Configure the mock tool collection to work as a context manager
    mock_tool_collection.from_mcp.return_value.__enter__.return_value = (
        mock_tool_collection
    )
    mock_tool_collection.from_mcp.return_value.__exit__.return_value = None
    mock_tool_collection.tools = ["mock_tool"]

    with patch.dict(
        "sys.modules",
        {
            "smolagents": MagicMock(
                CodeAgent=mock_code_agent,
                DuckDuckGoSearchTool=mock_ddg_tool,
                LiteLLMModel=mock_litellm_model,
                ToolCollection=mock_tool_collection,
            ),
            "mcp": MagicMock(
                StdioServerParameters=MagicMock(),
            ),
        },
    ):
        yield {
            "CodeAgent": mock_code_agent,
            "DuckDuckGoSearchTool": mock_ddg_tool,
            "LiteLLMModel": mock_litellm_model,
            "ToolCollection": mock_tool_collection,
        }


@pytest.mark.usefixtures("mock_smolagents_imports")
def test_run_smolagent_with_api_key_var():
    """Test smolagent creation with an API key from environment variable."""
    # The patch.dict(os.environ, {"TEST_API_KEY": "test-key-12345"})
    # is a testing construct that temporarily modifies the environment variables
    # for the duration of the test.
    # some tests use TEST_API_KEY while others don't
    with patch.dict(os.environ, {"TEST_API_KEY": "test-key-12345"}):
        from smolagents import CodeAgent, LiteLLMModel

        run_smolagent("openai/gpt-4", "Test prompt", api_key_var="TEST_API_KEY")

        LiteLLMModel.assert_called()
        model_call_kwargs = LiteLLMModel.call_args[1]
        assert model_call_kwargs["model_id"] == "openai/gpt-4"
        assert model_call_kwargs["api_key"] == "test-key-12345"
        assert model_call_kwargs["api_base"] is None

        CodeAgent.assert_called_once()
        CodeAgent.return_value.run.assert_called_once_with("Test prompt")


@pytest.mark.usefixtures("mock_smolagents_imports")
def test_run_smolagent_with_custom_api_base():
    """Test smolagent creation with a custom API base."""
    with patch.dict(os.environ, {"TEST_API_KEY": "test-key-12345"}):
        from smolagents import LiteLLMModel

        # Act
        run_smolagent(
            "anthropic/claude-3-sonnet",
            "Test prompt",
            api_key_var="TEST_API_KEY",
            api_base="https://custom-api.example.com",
        )
        last_call = LiteLLMModel.call_args_list[-1]

        assert last_call[1]["model_id"] == "anthropic/claude-3-sonnet"
        assert last_call[1]["api_key"] == "test-key-12345"
        assert last_call[1]["api_base"] == "https://custom-api.example.com"


@pytest.mark.usefixtures("mock_smolagents_imports")
def test_run_smolagent_without_api_key():
    """You should be able to run the smolagent without an API key."""
    from smolagents import LiteLLMModel

    run_smolagent("ollama_chat/deepseek-r1", "Test prompt")

    last_call = LiteLLMModel.call_args_list[-1]
    assert last_call[1]["model_id"] == "ollama_chat/deepseek-r1"
    assert last_call[1]["api_key"] is None


def test_run_smolagent_environment_error():
    """Test that passing a bad api_key_var throws an error"""
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(KeyError, match="MISSING_KEY"):
            run_smolagent("test-model", "Test prompt", api_key_var="MISSING_KEY")

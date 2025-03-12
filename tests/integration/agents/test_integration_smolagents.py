import os
import pytest
from unittest.mock import patch

from surf_spot_finder.agents.smolagents import run_smolagent

# TODO I'd rather not use openai
INTEGRATION_MODEL = "openai/gpt-3.5-turbo"
API_KEY_VAR = "OPENAI_API_KEY"


@pytest.mark.skipif(
    "INTEGRATION_TESTS" not in os.environ,
    reason="Integration tests require INTEGRATION_TESTS env var",
)
def test_smolagent_integration():
    """
    Full integration test of the smolagent functionality.

    Requires:
    - Docker to be running
    - OPENAI_API_KEY in environment variables
    - INTEGRATION_TESTS env var to be set
    """
    with patch("smolagents.CodeAgent") as MockCodeAgent:
        # Create a mock agent that returns itself from run()
        mock_agent = MockCodeAgent.return_value
        mock_agent.run.return_value = mock_agent

        # Run the agent
        result = run_smolagent(
            INTEGRATION_MODEL,
            "Find popular surf spots in California",
            api_key_var=API_KEY_VAR,
        )

        # Verify the agent was created and run
        MockCodeAgent.assert_called_once()
        mock_agent.run.assert_called_once_with("Find popular surf spots in California")
        assert result is mock_agent


@pytest.mark.skipif(
    "INTEGRATION_TESTS" not in os.environ,
    reason="Full integration tests require INTEGRATION_TESTS env var",
)
def test_smolagent_real_execution():
    """
    Tests the actual execution of the agent against real APIs.

    WARNING: This will make actual API calls and incur costs.
    Only run when explicitly needed for full system testing.

    Requires:
    - Docker to be running
    - OPENAI_API_KEY in environment variables
    - INTEGRATION_TESTS env var to be set
    """
    # Run with a simple, inexpensive request
    agent = run_smolagent(
        INTEGRATION_MODEL,
        "What are three popular surf spots in California?",
        api_key_var=API_KEY_VAR,
    )

    # Basic verification that we got an agent back
    assert agent is not None

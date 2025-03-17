from unittest.mock import patch, MagicMock

import pytest

from surf_spot_finder.tracing import get_tracer_provider, setup_tracing


@pytest.mark.parametrize("json_tracer", [True, False])
def test_get_tracer_provider(tmp_path, json_tracer):
    mock_trace = MagicMock()
    mock_tracer_provider = MagicMock()
    mock_register = MagicMock()

    with (
        patch("surf_spot_finder.tracing.trace", mock_trace),
        patch("surf_spot_finder.tracing.TracerProvider", mock_tracer_provider),
        patch("surf_spot_finder.tracing.register", mock_register),
    ):
        get_tracer_provider(
            project_name="test_project",
            json_tracer=json_tracer,
            output_dir=tmp_path / "telemetry",
        )
        assert (tmp_path / "telemetry").exists() == json_tracer
        if json_tracer:
            mock_trace.set_tracer_provider.assert_called_once_with(
                mock_tracer_provider.return_value
            )
        else:
            mock_register.assert_called_once_with(
                project_name="test_project", set_global_tracer_provider=True
            )


@pytest.mark.parametrize(
    "agent_type,instrumentor",
    [
        ("openai", "openai.OpenAIInstrumentor"),
        ("openai_multi_agent", "openai.OpenAIInstrumentor"),
        ("smolagents", "smolagents.SmolagentsInstrumentor"),
    ],
)
def test_setup_tracing(agent_type, instrumentor):
    with patch(f"openinference.instrumentation.{instrumentor}") as mock_instrumentor:
        setup_tracing(MagicMock(), agent_type)
        mock_instrumentor.assert_called_once()


def test_invalid_agent_type():
    with pytest.raises(ValueError, match="agent_type must be one of"):
        setup_tracing(MagicMock(), "invalid_agent_type")

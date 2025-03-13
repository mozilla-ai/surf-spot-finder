import os
from datetime import datetime

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export import SpanExporter
from phoenix.otel import register


class JsonFileSpanExporter(SpanExporter):
    def __init__(self, file_name: str):
        self.file_name = file_name

    def export(self, spans) -> None:
        with open(self.file_name, "a") as f:
            for span in spans:
                f.write(
                    span.to_json() + "\n"
                )  # Ensure to_json() method is properly implemented

    def shutdown(self):
        pass


def get_tracer_provider(
    project_name: str, json_tracer: bool, output_dir: str = "telemetry_output"
) -> TracerProvider:
    """
    Create a tracer_provider based on the selected mode.

    Args:
        project_name: Name of the project for tracing
        json_tracer: Whether to use the custom JSON file exporter (True) or Phoenix (False)
        output_dir: The directory where the telemetry output will be stored.
            Only used if `json_tracer=True`.
            Defaults to "telemetry_output".

    Returns:
        TracerProvider: The configured tracer provider
    """
    if json_tracer:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        tracer_provider = TracerProvider()
        trace.set_tracer_provider(tracer_provider)

        json_file_exporter = JsonFileSpanExporter(
            file_name=f"{output_dir}/{project_name}-{timestamp}.json"
        )
        span_processor = SimpleSpanProcessor(json_file_exporter)
        tracer_provider.add_span_processor(span_processor)
    else:
        tracer_provider = register(project_name=project_name)

    return tracer_provider


def setup_tracing(tracer_provider: TracerProvider, agent_type: str) -> None:
    """Setup tracing for `agent_type` by instrumenting `trace_provider`.

    Args:
        tracer_provider (TracerProvider): The configured tracer provider from
            [get_tracer_provider][surf_spot_finder.tracing.get_tracer_provider].
        agent_type (str): The type of agent being used.
            Must be one of the supported types in [RUNNERS][surf_spot_finder.agents.RUNNERS].
    """
    from surf_spot_finder.agents import validate_agent_type

    validate_agent_type(agent_type)

    if agent_type == "openai":
        from openinference.instrumentation.openai import OpenAIInstrumentor

        OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
    elif agent_type == "smolagents":
        from openinference.instrumentation.smolagents import SmolagentsInstrumentor

        SmolagentsInstrumentor().instrument(tracer_provider=tracer_provider)

from datetime import datetime
import os

from opentelemetry import trace
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
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


def setup_tracing(project_name: str, json_tracer: bool = True) -> TracerProvider:
    """
    Set up tracing configuration based on the selected mode.

    Args:
        project_name: Name of the project for tracing
        json_tracer: Whether to use the custom JSON file exporter (True) or Phoenix (False)

    Returns:
        TracerProvider: The configured tracer provider
    """
    if json_tracer:
        local_folder: str = "telemetry_output"
        if not os.path.exists(local_folder):
            os.makedirs(local_folder)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        tracer_provider = TracerProvider()
        trace.set_tracer_provider(tracer_provider)

        json_file_exporter = JsonFileSpanExporter(
            file_name=f"{local_folder}/{project_name}-{timestamp}.json"
        )
        span_processor = SimpleSpanProcessor(json_file_exporter)
        tracer_provider.add_span_processor(span_processor)
    else:
        tracer_provider = register(project_name=project_name)

    SmolagentsInstrumentor().instrument(tracer_provider=tracer_provider)

    return tracer_provider

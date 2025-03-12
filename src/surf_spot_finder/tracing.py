from datetime import datetime
import os
import json

from opentelemetry import trace
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export import SpanExporter
from phoenix.otel import register



class JsonFileSpanExporter(SpanExporter):
    def __init__(self, file_name: str):
        self.file_name = file_name
        # Initialize with an empty array if file doesn't exist
        if not os.path.exists(self.file_name):
            with open(self.file_name, "w") as f:
                json.dump([], f)
    
    def export(self, spans) -> None:
        # Read existing spans
        try:
            with open(self.file_name, "r") as f:
                all_spans = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            all_spans = []
        
        # Add new spans
        for span in spans:
            try:
                # Try to parse the span data from to_json() if it returns a string
                span_data = json.loads(span.to_json())
            except (json.JSONDecodeError, TypeError, AttributeError):
                # If span.to_json() doesn't return valid JSON string
                span_data = {"error": "Could not serialize span", "span_str": str(span)}
            
            all_spans.append(span_data)
        
        # Write all spans back to the file as a proper JSON array
        with open(self.file_name, "w") as f:
            json.dump(all_spans, f, indent=2)
    
    def shutdown(self):
        pass

def setup_tracing(project_name: str, json_tracer: bool) -> str | None:
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

        file_name = f"{local_folder}/{project_name}-{timestamp}.json"
        json_file_exporter = JsonFileSpanExporter(
            file_name=file_name
        )
        span_processor = SimpleSpanProcessor(json_file_exporter)
        tracer_provider.add_span_processor(span_processor)
    else:
        tracer_provider = register(
            project_name=project_name, set_global_tracer_provider=True
        )
        file_name=None

    SmolagentsInstrumentor().instrument(tracer_provider=tracer_provider)

    return file_name

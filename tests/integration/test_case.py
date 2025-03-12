"""
TODO

More work is needed to further refine a test case template,
but this is a starting place
"""
sample = {}

input = {
    "location": "Vigo",
    "date": "2025-03-15 22:00",
    "max_driving_hours": 3,
    "model_id": "openai/gpt-4o",
    "api_key_var": "OPENAI_API_KEY",
    "json_tracer": True,
    "api_base": None,
}

output = {
    "location": "Playa de Patos",
    "Weather forecast": {
        "temperature": "about 14°C +-5°C",
        "wave height": "about 1 meter",
    },
}

# Base checkpoints for agent behavior
checkpoints = [
    {
        "value": 1,
        "criteria": "Check if the agent consulted DuckDuckGoSearchTool for locations near Vigo.",
    },
    {
        "value": 1,
        "criteria": "Check if the agent fetched a website for forecasting, not relying on text from a DuckDuckGo search.",
    }
]

final_answer_criteria = []


# Add checkpoints for each output value
def add_output_final_answer_criteria(output_dict, prefix=""):
    """Recursively add checkpoints for each value in the output dictionary"""
    for key, value in output_dict.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            add_output_final_answer_criteria(value, path)
        else:
            final_answer_criteria.append(
                {
                    "value": 1,
                    "criteria": f"Check if {path} approximately matches the expected value '{value}'.",
                }
            )


# Add final_answer_criteria for all output values
add_output_final_answer_criteria(output)

sample = {
    "input": input,
    "checkpoints": checkpoints,
    "output": output,
    "final_answer_criteria": final_answer_criteria,
}

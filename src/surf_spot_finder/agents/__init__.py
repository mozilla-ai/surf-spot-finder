from .openai import run_openai_agent, run_openai_multi_agent
from .smolagents import run_smolagent

RUNNERS = {
    "openai": run_openai_agent,
    "smolagents": run_smolagent,
    "openai_multi_agent": run_openai_multi_agent,
}


def validate_agent_type(value) -> str:
    if value not in RUNNERS:
        raise ValueError(f"agent_type must be one of {RUNNERS.keys()}")

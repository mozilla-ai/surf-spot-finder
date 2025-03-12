from .openai import run_openai_agent
from .smolagents import run_smolagent

RUNNERS = {
    "openai": run_openai_agent,
    "smolagents": run_smolagent,
}

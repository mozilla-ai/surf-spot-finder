from enum import Enum
from .langchain import run_lanchain_agent
from .openai import run_openai_agent, run_openai_multi_agent
from .smolagents import run_smolagent


# Define the available agent type enums
class AgentType(str, Enum):
    LANGCHAIN = "langchain"
    OPENAI = "openai"
    OPENAI_MULTI_AGENT = "openai_multi_agent"
    SMOLAGENTS = "smolagents"


RUNNERS = {
    AgentType.LANGCHAIN: run_lanchain_agent,
    AgentType.OPENAI: run_openai_agent,
    AgentType.SMOLAGENTS: run_smolagent,
    AgentType.OPENAI_MULTI_AGENT: run_openai_multi_agent,
}


def validate_agent_type(value: str) -> str:
    try:
        agent_type = AgentType(value)
        if agent_type not in RUNNERS:
            raise ValueError(f"agent_type {value} is valid but has no runner")
        return value
    except ValueError:
        raise ValueError(f"agent_type must be one of {[e.value for e in AgentType]}")

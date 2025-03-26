from enum import Enum


class AgentType(str, Enum):
    LANGCHAIN = "langchain"
    OPENAI = "openai"
    OPENAI_MULTI_AGENT = "openai_multi_agent"
    SMOLAGENTS = "smolagents"

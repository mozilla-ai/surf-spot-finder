from typing import Annotated

from any_agent import AgentFramework
from any_agent.config import AgentConfig
from pydantic import AfterValidator, BaseModel, ConfigDict, FutureDatetime, PositiveInt
import yaml


INPUT_PROMPT_TEMPLATE = """
According to the forecast, what will be the best spot to surf around {LOCATION},
in a {MAX_DRIVING_HOURS} hour driving radius,
at {DATE}?"
""".strip()


def validate_prompt(value) -> str:
    for placeholder in ("{LOCATION}", "{MAX_DRIVING_HOURS}", "{DATE}"):
        if placeholder not in value:
            raise ValueError(f"prompt must contain {placeholder}")
    return value


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    location: str
    max_driving_hours: PositiveInt
    date: FutureDatetime
    input_prompt_template: Annotated[str, AfterValidator(validate_prompt)] = (
        INPUT_PROMPT_TEMPLATE
    )

    framework: AgentFramework

    main_agent: AgentConfig
    managed_agents: list[AgentConfig] | None = None

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)    yaml_path: Path to the YAML configuration file

        Returns:
            Config: A new Config instance populated with values from the YAML file
        """
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        # for each tool listed in main_agent.tools, use import lib to import it and replace the str with the callable
        callables = []
        for tool in data["main_agent"]["tools"]:
            if isinstance(tool, str):
                module_name, func_name = tool.rsplit(".", 1)
                module = __import__(module_name, fromlist=[func_name])
                print(f"Importing {tool}")
                callables.append(getattr(module, func_name))
            else:
                # this means it must be an MCPStdioParams
                callables.append(tool)
        data["main_agent"]["tools"] = callables
        for agent in data.get("managed_agents", []):
            callables = []
            for tool in agent.get("tools", []):
                if isinstance(tool, str):
                    module_name, func_name = tool.rsplit(".", 1)
                    module = __import__(module_name, fromlist=[func_name])
                    print(f"Importing {tool}")
                    callables.append(getattr(module, func_name))
                else:
                    # this means it must be an MCPStdioParams
                    callables.append(tool)
            agent["tools"] = callables

        data["framework"] = AgentFramework[data["framework"]]
        return cls(**data)

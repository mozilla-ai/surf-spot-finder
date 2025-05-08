from typing import Annotated
from datetime import datetime, timedelta
from any_agent import AgentFramework
from any_agent.config import AgentConfig
from pydantic import AfterValidator, BaseModel, ConfigDict, FutureDatetime, PositiveInt
import yaml
from rich.prompt import Prompt
from loguru import logger
import geocoder

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


def ask_framework() -> AgentFramework:
    """
    Ask the user which framework they would like to use. They must select one of the Agent Frameworks
    """
    frameworks = [framework.name for framework in AgentFramework]
    frameworks_str = "\n".join(
        [f"{i}: {framework}" for i, framework in enumerate(frameworks)]
    )
    prompt = f"Select the agent framework to use:\n{frameworks_str}\n"
    choice = Prompt.ask(prompt, default="3")
    try:
        choice = int(choice)
        if choice < 0 or choice >= len(frameworks):
            raise ValueError("Invalid choice")
        return AgentFramework[frameworks[choice]]
    except ValueError:
        raise ValueError("Invalid choice")


def date_picker() -> FutureDatetime:
    """
    Ask the user to select a date in the future. The date must be at least 1 day in the future.
    """
    prompt = "Select a date in the future (YYYY-MM-DD-HH)"
    # the default should be the current date + 1 day
    now = datetime.now()
    default_val = (now + timedelta(days=1)).strftime("%Y-%m-%d-%H")
    date_str = Prompt.ask(prompt, default=default_val)
    try:
        year, month, day, hour = map(int, date_str.split("-"))
        date = datetime(year, month, day, hour)
        return date
    except ValueError:
        raise ValueError("Invalid date format. Please use YYYY-MM-DD-HH.")


def location_picker() -> str:
    """
    Ask the user to input a location. By default use the current location based on the IP address.
    """
    prompt = "Enter a location"
    g = geocoder.ip("me")
    default_val = f"{g.city} {g.state}, {g.country}"
    location = Prompt.ask(prompt, default=default_val)
    if not location:
        raise ValueError("location cannot be empty")
    return location


def max_driving_hours_picker() -> int:
    """
    Ask the user to input the maximum driving hours. The default is 2 hours.
    """
    prompt = "Enter the maximum driving hours"
    default_val = 2
    max_driving_hours = Prompt.ask(prompt, default=default_val)
    try:
        max_driving_hours = int(max_driving_hours)
        if max_driving_hours <= 0:
            raise ValueError("Invalid choice")
        return max_driving_hours
    except ValueError:
        raise ValueError("Invalid choice")


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
        if not data.get("framework"):
            data["framework"] = ask_framework()
        else:
            logger.info(f"Using framework {data['framework']}")
        if not data.get("location"):
            data["location"] = location_picker()
        else:
            logger.info(f"Using location {data['location']}")
        if not data.get("max_driving_hours"):
            data["max_driving_hours"] = max_driving_hours_picker()
        else:
            logger.info(f"Using max driving hours {data['max_driving_hours']}")
        if not data.get("date"):
            data["date"] = date_picker()
        else:
            logger.info(f"Using date {data['date']}")
        return cls(**data)

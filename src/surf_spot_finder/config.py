from typing import Annotated

from any_agent.schema import AgentSchema
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

    framework: str

    main_agent: AgentSchema
    managed_agents: list[AgentSchema] | None = None

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
        return cls(**data)

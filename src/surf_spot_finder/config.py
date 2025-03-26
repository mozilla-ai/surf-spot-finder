from typing import Annotated

from any_agent.schema import AgentSchema
from pydantic import AfterValidator, BaseModel, FutureDatetime, PositiveInt

from surf_spot_finder.prompts.shared import INPUT_PROMPT


def validate_prompt(value) -> str:
    for placeholder in ("{LOCATION}", "{MAX_DRIVING_HOURS}", "{DATE}"):
        if placeholder not in value:
            raise ValueError(f"prompt must contain {placeholder}")
    return value


class Config(BaseModel):
    location: str
    max_driving_hours: PositiveInt
    date: FutureDatetime
    input_prompt_template: Annotated[str, AfterValidator(validate_prompt)] = (
        INPUT_PROMPT
    )

    framework: str

    main_agent: AgentSchema
    managed_agents: list[AgentSchema] | None = None

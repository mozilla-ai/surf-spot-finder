from typing import Annotated, Optional
from pydantic import AfterValidator, BaseModel, FutureDatetime, PositiveInt

from surf_spot_finder.prompts.shared import INPUT_PROMPT


def validate_prompt(value) -> str:
    for placeholder in ("{LOCATION}", "{MAX_DRIVING_HOURS}", "{DATE}"):
        if placeholder not in value:
            raise ValueError(f"prompt must contain {placeholder}")
    return value


def validate_agent_type(value) -> str:
    from surf_spot_finder.agents import validate_agent_type

    validate_agent_type(value)
    return value


class Config(BaseModel):
    input_prompt_template: Annotated[str, AfterValidator(validate_prompt)] = (
        INPUT_PROMPT
    )
    location: str
    max_driving_hours: PositiveInt
    date: FutureDatetime
    model_id: str
    agent_type: Annotated[str, AfterValidator(validate_agent_type)]
    api_key_var: Optional[str] = None
    json_tracer: bool = True
    api_base: Optional[str] = None

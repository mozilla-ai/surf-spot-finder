from typing import Annotated, Optional
from pydantic import AfterValidator, BaseModel, FutureDatetime, PositiveInt


DEFAULT_PROMPT = (
    "What will be the best surf spot around {LOCATION}"
    ", in a radio of {MAX_DRIVING_HOURS} hours driving"
    ", at {DATE}?"
)


def validate_prompt(value):
    for placeholder in ("{LOCATION}", "{MAX_DRIVING_HOURS}"):
        if placeholder not in value:
            raise ValueError(f"prompt must contain {placeholder}")
    return value


class Config(BaseModel):
    prompt: str = Annotated[str, AfterValidator(validate_prompt)]
    location: str
    max_driving_hours: PositiveInt
    date: FutureDatetime
    model_id: str
    api_key_var: Optional[str] = None
    json_tracer: bool = True

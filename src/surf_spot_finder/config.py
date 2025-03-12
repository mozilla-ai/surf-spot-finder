from typing import Annotated, Optional
from pydantic import AfterValidator, BaseModel, FutureDatetime, PositiveInt
from datetime import datetime

CURRENT_DATE = datetime.now().strftime("%Y-%m-%d")

DEFAULT_PROMPT = (
    "What will be the best surf spot around {LOCATION}"
    ", in a {MAX_DRIVING_HOURS} hour driving radius"
    ", at {DATE}? it is currently "
    + CURRENT_DATE
    + ". find me the best surf spot and the"
    " up to date weather forecast for that day."
)


def validate_prompt(value) -> str:
    for placeholder in ("{LOCATION}", "{MAX_DRIVING_HOURS}", "{DATE}"):
        if placeholder not in value:
            raise ValueError(f"prompt must contain {placeholder}")
    return value


class Config(BaseModel):
    prompt: Annotated[str, AfterValidator(validate_prompt)]
    location: str
    max_driving_hours: PositiveInt
    date: FutureDatetime
    model_id: str
    api_key_var: Optional[str] = None
    json_tracer: bool = True
    api_base: Optional[str] = None

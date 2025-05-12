from datetime import datetime, timedelta
import json
import requests
import streamlit as st
from any_agent import AgentFramework
from any_agent.tracing.trace import _is_tracing_supported
from any_agent.evaluation import EvaluationCase

from constants import MODEL_OPTIONS


def create_evaluation_case() -> EvaluationCase:
    """Create an EvaluationCase from the user configuration.

    Args:
        case_config (dict): The evaluation case configuration from the user

    Returns:
        EvaluationCase: The created evaluation case
    """

    return EvaluationCase(
        llm_judge="openai/gpt-4.1-mini",
        checkpoints=[
            {
                "criteria": "Check if the agent used the get_surfing_spots tool and it succeeded, and that the tool was used before the get_wave_forecast and get_wind_forecast tools",
                "points": 1,
            },
            {
                "criteria": "Check if the agent used the get_wave_forecast tool and it succeeded",
                "points": 1,
            },
            {
                "criteria": "Check if the agent used the get_wind_forecast tool and it succeeded",
                "points": 1,
            },
            {
                "criteria": "Check if the agent used the get_area_lat_lon tool and it succeeded",
                "points": 1,
            },
            {
                "criteria": "Check if the agent used the driving_hours_to_meters tool to convert the driving hours to meters and it succeeded",
                "points": 1,
            },
            {
                "criteria": "Check if the final answer contains any description about the weather at the chosen location",
                "points": 1,
            },
            {
                "criteria": "Check if the final answer contains one of the surf spots found by a call of the get_surfing_spots tool",
                "points": 1,
            },
            {
                "criteria": "Check that the agent completed in fewer than 10 steps",
                "points": 1,
            },
        ],
    )


@st.cache_resource
def get_area(area_name: str) -> dict:
    """Get the area from Nominatim.

    Uses the [Nominatim API](https://nominatim.org/release-docs/develop/api/Search/).

    Args:
        area_name (str): The name of the area.

    Returns:
        dict: The area found.
    """
    response = requests.get(
        f"https://nominatim.openstreetmap.org/search?q={area_name}&format=json",
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=5,
    )
    response.raise_for_status()
    response_json = json.loads(response.content.decode())
    return response_json


def get_user_inputs() -> dict:
    default_val = "Los Angeles California, US"

    col1, col2 = st.columns([3, 1])
    with col1:
        location = st.text_input("Enter a location", value=default_val)
    with col2:
        if location:
            location_check = get_area(location)
            if not location_check:
                st.error("❌")
            else:
                st.success("✅")

    max_driving_hours = st.number_input(
        "Enter the maximum driving hours", min_value=1, value=2
    )

    col_date, col_time = st.columns([2, 1])
    with col_date:
        date = st.date_input(
            "Select a date in the future", value=datetime.now() + timedelta(days=1)
        )
    with col_time:
        # default to 9am
        time = st.time_input(
            "Select a time", value=datetime.now().time().replace(hour=9, minute=0)
        )
    date = datetime.combine(date, time)

    supported_frameworks = [
        framework for framework in AgentFramework if _is_tracing_supported(framework)
    ]

    framework = st.selectbox(
        "Select the agent framework to use",
        supported_frameworks,
        index=2,
        format_func=lambda x: x.name,
    )

    model_id = st.selectbox(
        "Select the model to use",
        MODEL_OPTIONS,
        index=0,
        format_func=lambda x: "/".join(x.split("/")[-3:]),
    )

    # Add evaluation case section
    with st.expander("Evaluation Case"):
        evaluation_case = create_evaluation_case()
        st.write(evaluation_case.model_dump(), expanded=True)

    return {
        "location": location,
        "max_driving_hours": max_driving_hours,
        "date": date,
        "framework": framework,
        "model_id": model_id,
        "evaluation_case": evaluation_case
        if st.checkbox("Run Evaluation", value=True)
        else None,
    }

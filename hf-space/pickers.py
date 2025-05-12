from datetime import datetime, timedelta
import json
import requests
import streamlit as st
from any_agent import AgentFramework
from any_agent.tracing.trace import _is_tracing_supported

from constants import MODEL_OPTIONS


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
    st.title("Surf Spot Finder")
    st.write(
        "This app finds the best surf spots in your area based on the weather forecast."
    )
    default_val = "Los Angeles California, US"

    location = st.text_input("Enter a location", value=default_val)
    if location:
        location_check = get_area(location)
        if not location_check:
            st.error("Invalid location. Please enter a valid location.")
            return None
        else:
            # display a checkmark saying that the location is valid
            st.success("Valid location")
    max_driving_hours = st.number_input(
        "Enter the maximum driving hours", min_value=1, value=2
    )
    date = st.date_input(
        "Select a date in the future", value=datetime.now() + timedelta(days=1)
    )

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

    return {
        "location": location,
        "max_driving_hours": max_driving_hours,
        "date": date,
        "framework": framework,
        "model_id": model_id,
    }

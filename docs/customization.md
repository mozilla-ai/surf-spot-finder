# 🎨 **Customization Guide**

The surf-spot-finder blueprint is designed to be flexible and adaptable for different use cases through modifying the YAML configuration file and mofifying the `config.py` and `cli.py` files.

---

# 🎯 **Adapting it for your use-case**

## Part 1 - Modifying the YAML config file

### 1. Customize the Prompt
The `input_prompt_template` is the key for setting the agent's instructions. Here are a couple of examples of how you could modify it:

```yaml
# Example 1: Hiking spot Finder
input_prompt_template: |
  Help me find the perfect hiking spot for {DATE} and {LOCATION}. Consider:
  - Trail difficulty and length
  - Current weather conditions
  - elevation and terrain type
  - Scenic viewpoints

# Example 2: Recipe finder
input_prompt_template: |
  I have these ingredients: {INGREDIENTS}
  Time available: {TIME_AVAILABLE}

  Suggest recipes that:
  1. Use my available ingredients
  2. Match my dietary preferences
  3. Can be prepared within my time limit
  4. Consider seasonal availability
  5. Minimize food waste
```
### 2. Customize the model
You can swap either the model to see how it affects the agent's performance:

```yaml
main_agent:
  model_id: "openai/gpt-4.1-mini"
```

### 3. Create Custom Tools
You can create your own tools to support your use case. This can be added to a file in the tools folder.
Here's a mock example tool for a hiking spot finder:

```python
# hiking_tools.py
import json
from datetime import datetime, timedelta
import requests


def get_weather_forecast(location: str, date: str) -> List[Dict]:
    """Get weather forecast for a location.

    Args:
        location: Trail location
        date: Date to check in ISO format (YYYY-MM-DD)

    Returns:
        Hourly weather data with:
        - Temperature
        - Conditions
        - Wind
        - Precipitation chance
    """
    # Example API call (similar to openmeteo.py)
    url = "https://api.weather-service.com/forecast"
    params = {
        "location": location,
        "date": date,
        "hourly": ["temperature", "conditions", "wind", "precipitation"]
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

```

Ensure you update the `__init__.py` file in the tools folder:
```python
from .hike_finder_tools import get_weather_forecast

__all__ = [
    "get_weather_forecast",
]
```

You can then add the tools to the yaml file
```yaml
# For the hiking spot finder
tools:
  - "surf_spot_finder.tools.get_weather_forecast"
```

### 4. Customize evaluation cases and evaluation model
You can define evaluation criteria to ensure your agent performs as expected. Here's an example for the hiking spot finder:

```yaml
evaluation_cases:
  - llm_judge: "openai/gpt-4o-mini"
    checkpoints:

      - criteria: "Check if the agent used the get_trail_conditions tool and it succeeded"
        points: 1

      # Tool Order
      - criteria: "Check if the agent checked weather conditions before suggesting a trail"
        points: 1

```

### 5. Putting it all together - Final YAML file
Here's how a complete YAML configuration file might look for the hiking spot finder:

```yaml
# hiking_spot_finder.yaml

input_prompt_template: |
  Help me find the perfect hiking spot for {DATE} and {LOCATION}. Consider:
  - Trail difficulty and length
  - Current weather conditions
  - Elevation and terrain type
  - Scenic viewpoints

main_agent:
  model_id: "openai/gpt-4.1-mini"
  tools:
    - "hiking_tools.get_weather_forecast"
    - "hiking_tools.get_trail_conditions"
    - "any_agent.tools.search_web"
    - "any_agent.tools.visit_webpage"

evaluation_cases:
  - llm_judge: "openai/gpt-4.1-nano"
    checkpoints:
      - criteria: "Check if the agent used the get_weather_forecast tool and it succeeded"
        points: 1
      - criteria: "Check if the agent used the get_trail_conditions tool and it succeeded"
        points: 1
```

## Part 2 - Modifying the config file and the cli file

You'll need to edit `src/surf_spot_finder/config.py` to:

  - Update the validation requirements for your use case
  - Add support for any new parameters you need
  - Modify the Config class to handle your specific parameters requirements

*For example, for the Hiking Spot Finder example, you may want to remove references to `{MAX_DRIVING_HOURS}`, as its not in the prompt.*

You'll likely also need to edit `src/surf_spot_finder/cli.py` to:

- Update the input prompt template depending on your specific parameter requirements.

*For example, for the Hiking Spot Finder example, you may want to remove references to `{MAX_DRIVING_HOURS}`, as its not in the prompt.*

## Part 3 - Running the agent

With the new YAML file and the updated `config.py`, you can run the agent with:

```bash
surf-spot-finder path_to_yaml/hiking_spot_finder.yaml
```

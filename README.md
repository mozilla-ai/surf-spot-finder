<p align="center">
  <picture>
    <!-- When the user prefers dark mode, show the white logo -->
    <source media="(prefers-color-scheme: dark)" srcset="./images/Blueprint-logo-white.png">
    <!-- When the user prefers light mode, show the black logo -->
    <source media="(prefers-color-scheme: light)" srcset="./images/Blueprint-logo-black.png">
    <!-- Fallback: default to the black logo -->
    <img src="./images/Blueprint-logo-black.png" width="35%" alt="Project logo"/>
  </picture>
</p>

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![](https://dcbadge.limes.pink/api/server/YuMNeuKStr?style=flat)](https://discord.gg/YuMNeuKStr) <br>
[![Try on Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Try%20on-Spaces-blue)](https://huggingface.co/spaces/mozilla-ai/surf-spot-finder)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LiteLLM](https://img.shields.io/badge/🚅%20LiteLLM-1E90FF)](https://www.litellm.ai/)
[![Any-Agent](https://img.shields.io/badge/🎨%20Any--Agent-white)](https://github.com/mozilla-ai/any-agent) <br>
[![Docs](https://github.com/mozilla-ai/surf-spot-finder/actions/workflows/docs.yaml/badge.svg)](https://github.com/mozilla-ai/surf-spot-finder/actions/workflows/docs.yaml/)
[![Ruff](https://github.com/mozilla-ai/surf-spot-finder/actions/workflows/lint.yaml/badge.svg?label=Ruff)](https://github.com/mozilla-ai/surf-spot-finder/actions/workflows/lint.yaml/)

[Blueprints Hub](https://developer-hub.mozilla.ai/)
| [Documentation](https://mozilla-ai.github.io/surf-spot-finder/)
| [Getting Started](https://mozilla-ai.github.io/surf-spot-finder/getting-started)
| [Contributing](CONTRIBUTING.md)

</div>

# Surf Spot Finder: a Blueprint for comparing agent frameworks on a specific task

Many Large Language Model (LLM) capabilities are unlocked when they are given access to tools and given control of their
own runtime and execution path. However, it's important that as they are given greater capabilities, they are properly
evaluated and controlled.

In this Blueprint, we demonstrate an AI agent designed for an extremely specific task (some refer to this as a "Vertical Agent")
that is given the web and searching access it needs to find an answer the same way you would find the answer as a human.

This agent is designed for help in finding the next great surf spot near you: the agent is provided with a location, a distance,
a timestamp, and it's able to independently search and browse the web to recommend the best spot to you along with the
relevant information!

Although this exact use-case may not be useful to you directly, the framework we provide here is intended to be easily
adapted to the Agent use case you have in mind.

This implementation uses the [smolagents](https://huggingface.co/docs/smolagents/index) library for Agentic capabilities, alongside
of the increasingly Model Context Protocol (MCP) which allows for a standard access communication standard for a large number of tools.


## 🚀 Quick Start

Try out our demo on HF Spaces: [![Try on Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Try%20on-Spaces-blue)](https://huggingface.co/spaces/mozilla-ai/surf-spot-finder)

### 1️⃣ Clone the Project
```bash
git clone https://github.com/mozilla-ai/surf-spot-finder.git
cd surf-spot-finder
```

### 2️⃣ Update submodule and install dependencies
```bash
pip install -e .  # Install root project dependencies
```

### 3️⃣ Run

```bash
surf-spot-finder examples/single_agent_with_tools.yaml
```

## How it Works


## Pre-requisites

- **System requirements**:
  - OS: Windows, macOS, or Linux
  - Python 3.10 or higher
  - Minimum RAM:
  - Disk space:

- **Dependencies**:
  - Docker
  - Dependencies listed in `pyproject.toml`

## Run Tests

```bash
pip install -e .[tests]
```

### Unit Tests

```bash
pytest
```

## Troubleshooting


## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! To get started, you can check out the [CONTRIBUTING.md](CONTRIBUTING.md) file.

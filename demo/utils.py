import json
from typing import Any
import streamlit as st
from surf_spot_finder.tools import (
    driving_hours_to_meters,
    get_area_lat_lon,
    get_surfing_spots,
    get_wave_forecast,
    get_wind_forecast,
)
from surf_spot_finder.config import Config
from any_agent import AgentConfig, AnyAgent, TracingConfig
from any_agent.tracing.trace import AgentTrace
from any_agent.tracing.otel_types import StatusCode
from any_agent.evaluation import evaluate, TraceEvaluationResult


async def run_agent(user_inputs: dict[str, Any]):
    st.markdown("### 🔍 Running Surf Spot Finder...")

    if "huggingface" in user_inputs["model_id"]:
        model_args = {
            "extra_headers": {"X-HF-Bill-To": "mozilla-ai"},
            "temperature": 0.0,
        }
    else:
        model_args = {}

    agent_config = AgentConfig(
        model_id=user_inputs["model_id"],
        model_args=model_args,
        tools=[
            get_wind_forecast,
            get_wave_forecast,
            get_area_lat_lon,
            get_surfing_spots,
            driving_hours_to_meters,
        ],
    )

    config = Config(
        location=user_inputs["location"],
        max_driving_hours=user_inputs["max_driving_hours"],
        date=user_inputs["date"],
        framework=user_inputs["framework"],
        main_agent=agent_config,
        managed_agents=[],
        evaluation_cases=[user_inputs.get("evaluation_case")]
        if user_inputs.get("evaluation_case")
        else None,
    )

    agent = await AnyAgent.create_async(
        agent_framework=config.framework,
        agent_config=config.main_agent,
        managed_agents=config.managed_agents,
        tracing=TracingConfig(console=True, cost_info=True),
    )

    query = config.input_prompt_template.format(
        LOCATION=config.location,
        MAX_DRIVING_HOURS=config.max_driving_hours,
        DATE=config.date,
    )

    st.markdown("#### 📝 Query")
    st.code(query, language="text")

    with st.spinner("🤔 Analyzing surf spots..."):
        agent_trace: AgentTrace = await agent.run_async(query)
        agent.exit()

    st.markdown("### 🏄 Results")
    st.markdown("#### Final Output")
    st.info(agent_trace.final_output)

    # Display the agent trace in a more organized way
    with st.expander("### 🧩 Agent Trace"):
        for span in agent_trace.spans:
            # Header with name and status
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{span.name}**")
                if span.attributes:
                    # st.json(span.attributes, expanded=False)
                    if "input.value" in span.attributes:
                        input_value = json.loads(span.attributes["input.value"])
                        if isinstance(input_value, list):
                            st.write(f"Input: {input_value[-1]}")
                        else:
                            st.write(f"Input: {input_value}")
                    if "output.value" in span.attributes:
                        output_value = json.loads(span.attributes["output.value"])
                        if isinstance(output_value, list):
                            st.write(f"Output: {output_value[-1]}")
                        else:
                            st.write(f"Output: {output_value}")
            with col2:
                status_color = (
                    "green" if span.status.status_code == StatusCode.OK else "red"
                )
                st.markdown(
                    f"<span style='color: {status_color}'>● {span.status.status_code.name}</span>",
                    unsafe_allow_html=True,
                )

    if config.evaluation_cases is not None:
        assert (
            len(config.evaluation_cases) == 1
        ), "Only one evaluation case is supported in the demo"
        st.markdown("### 📊 Evaluation Results")

        with st.spinner("Evaluating results..."):
            case = config.evaluation_cases[0]
            result: TraceEvaluationResult = evaluate(
                evaluation_case=case,
                trace=agent_trace,
                agent_framework=config.framework,
            )

            all_results = (
                result.checkpoint_results
                + result.hypothesis_answer_results
                + result.direct_results
            )

            # Create columns for better layout
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Criteria Results")
                for checkpoint in all_results:
                    if checkpoint.passed:
                        st.success(f"✅ {checkpoint.criteria}")
                    else:
                        st.error(f"❌ {checkpoint.criteria}")

            with col2:
                st.markdown("#### Overall Score")
                total_points = sum([result.points for result in all_results])
                if total_points == 0:
                    msg = "Total points is 0, cannot calculate score."
                    raise ValueError(msg)
                passed_points = sum(
                    [result.points for result in all_results if result.passed]
                )

                # Create a nice score display
                st.markdown(f"### {passed_points}/{total_points}")
                percentage = (passed_points / total_points) * 100
                st.progress(percentage / 100)
                st.markdown(f"**{percentage:.1f}%**")

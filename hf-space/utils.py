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
from any_agent.evaluation import evaluate, TraceEvaluationResult


async def run_agent(user_inputs):
    st.write("Running surf spot finder...")
    if "huggingface" in user_inputs["model_id"]:
        model_args = {
            "extra_headers": {"X-HF-Bill-To": "mozilla-ai"},
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
        evaluation_cases=None,
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
    st.write("Running agent with query:\n", query)

    with st.spinner("Running..."):
        agent_trace = await agent.run_async(query)
        agent.exit()

    st.write("Final output from agent:\n", agent_trace.final_output)

    # Display the agent trace
    with st.expander("Agent Trace", expanded=True):
        st.write(agent_trace.spans)

    if config.evaluation_cases is not None:
        results = []
        st.write("Found evaluation cases, running trace evaluation")
        for i, case in enumerate(config.evaluation_cases):
            st.write("Evaluating case: ", case)
            result: TraceEvaluationResult = evaluate(
                evaluation_case=case,
                trace=agent_trace,
                agent_framework=config.framework,
            )
            for list_of_checkpoints in [
                result.checkpoint_results,
                result.direct_results,
                result.hypothesis_answer_results,
            ]:
                for checkpoint in list_of_checkpoints:
                    msg = (
                        f"Checkpoint: {checkpoint.criteria}\n"
                        f"\tPassed: {checkpoint.passed}\n"
                        f"\tReason: {checkpoint.reason}\n"
                        f"\tScore: {'%d/%d' % (checkpoint.points, checkpoint.points) if checkpoint.passed else '0/%d' % checkpoint.points}"
                    )
                    st.write(msg)
            st.write("==========================")
            st.write("Overall Score: %d%%", 100 * result.score)
            st.write("==========================")
            results.append(result)
    st.write("Surf spot finder finished running.")

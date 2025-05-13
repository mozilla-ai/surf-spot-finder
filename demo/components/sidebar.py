from components.inputs import UserInputs, get_user_inputs
import streamlit as st


def ssf_sidebar() -> UserInputs:
    st.markdown("### Configuration")
    st.markdown("Built using [Any-Agent](https://github.com/mozilla-ai/any-agent)")
    user_inputs = get_user_inputs()
    return user_inputs

import streamlit as st
import asyncio
import nest_asyncio
from pickers import get_user_inputs
from utils import run_agent


nest_asyncio.apply()


with st.sidebar:
    user_inputs = get_user_inputs()
    is_valid = user_inputs is not None
    run_button = st.button("Run", disabled=not is_valid)


async def main():
    if run_button:
        await run_agent(user_inputs)
    else:
        st.write("Click run to start!")


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    loop.run_until_complete(main())

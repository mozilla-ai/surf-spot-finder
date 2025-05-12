import streamlit as st
import asyncio
import nest_asyncio
from pickers import get_user_inputs
from utils import run_agent


nest_asyncio.apply()

# Set page config
st.set_page_config(page_title="Surf Spot Finder", page_icon="🏄", layout="wide")

# Add title and header
st.title("🏄 Surf Spot Finder")
st.markdown(
    "Find the best surfing spots based on your location and preferences! [Github Repo](https://github.com/mozilla-ai/surf-spot-finder)"
)

# Sidebar
with st.sidebar:
    st.markdown("### Configuration")
    st.markdown("Built using [Any-Agent](https://github.com/mozilla-ai/any-agent)")
    user_inputs = get_user_inputs()
    is_valid = user_inputs is not None
    run_button = st.button("Run", disabled=not is_valid, type="primary")


# Main content
async def main():
    if run_button:
        await run_agent(user_inputs)
    else:
        st.info(
            "👈 Configure your search parameters in the sidebar and click Run to start!"
        )


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    loop.run_until_complete(main())

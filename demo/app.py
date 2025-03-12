import gradio as gr
import spaces

from surf_spot_finder.cli import find_surf_spot


@spaces.GPU
def run_find_surf_spot(location, date, max_driving_hours):
    result = find_surf_spot(
        location=location,
        date=date,
        max_driving_hours=max_driving_hours,
        model_id="gemini/gemini-2.0-flash-thinking-exp-01-21",
        api_key_var="GEMINI_API_KEY",
    )
    return result


with gr.Blocks() as demo:
    gr.Markdown("# Surf Spot Finder")

    location = gr.Textbox(label="Input Location")
    date = gr.Textbox(label="Input Date")
    max_driving_hours = gr.Number(label="Max hours to drive")

    with gr.Row():
        process_button = gr.Button("Find Surf Spot")
    with gr.Row():
        output_message = gr.Textbox(label="Output Message")

    process_button.click(
        fn=run_find_surf_spot,
        inputs=[location, date, max_driving_hours],
        outputs=output_message,
    )


demo.launch()

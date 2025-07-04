import gradio as gr
import ollama_llms as llms
import prompts


def run_pipeline():
    return f"run done"


with gr.Blocks() as interface:
    gr.Markdown("Benchmark and Prompt Optimization Platform")
    inputs = [  # https://www.gradio.app/main/docs/gradio/slider
        # set with default values.
        gr.Slider(0, 1, step=0.1, value=0.7, label="Temperature",
                  info="The temperature of the model. Increasing the temperature will make the model answer more creatively. (Default: 0.8)"),
        gr.Slider(0, 1, step=0.01, value=0.9, label="Top-p",
                  info="Works together with top-k. A higher value(e.g., 0.95) will lead to more diverse text, while a lower value(e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)"),
        gr.Slider(1, 1000, step=1, value=40, label="Top-k",
                  info="Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 40)"),
        gr.Slider(0, 1, step=0.01, value=1, label="Tfs-z",
                  info="Tail free sampling is used to reduce the impact of less probable tokens from the output. A higher value(e.g., 2.0) will reduce the impact more, while a value of 1.0 disables this setting. (default: 1)"),
        gr.Dropdown(llms.available_llm_models, label="Model",
                    info="Models used for this project"),
        gr.Checkbox(label="Reason", info="Reasoning block usage"),
        gr.CheckboxGroup(prompts.available_prompt_engineeing, label="Prompt",
                         label="Prompt Engineering Methods"),
        gr.Textbox(lines=3, label="Question", info="input to model")
    ]
    outputs = [gr.Textbox(label="Model Output", lines=5),]
    run_btn = gr.Button("Run")
    run_btn.click(fn=run_pipeline, inputs=inputs, outputs=outputs)
    # reset_btn = gr.Button("Reset")
    # reset_btn.click(fn=reset_inputs, inputs=[], outputs=inputs + outputs)

interface.launch()

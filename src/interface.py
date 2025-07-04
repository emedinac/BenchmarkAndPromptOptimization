import gradio as gr
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import ollama_llms as llms
import prompts
from pathlib import Path
import numpy as np


exp_path = Path("results")


def norm(model_metrics, models, metrics):
    for me in metrics:
        values = []
        for mo in models:
            values.append(model_metrics[mo][me])
        maxval, minval = np.max(values), np.min(values)
        values = values/maxval
        for mo, val in zip(models, values):
            model_metrics[mo][me + f"  - max_val: {maxval:.2f}"] = val
            del model_metrics[mo][me]
    return model_metrics


def plotly_radar_multiple_models(model_metrics: dict):
    """
    model_metrics = {
        "llama3": { stats, blabla },
        "mistral": {stats, blabla },
        "ollama": {stats, blabla },
    }
    """
    all_models = list(model_metrics.keys())
    all_metrics = list(model_metrics[all_models[0]])
    model_metrics = norm(model_metrics, all_models, all_metrics)
    all_models = list(model_metrics.keys())
    all_metrics = list(model_metrics[all_models[0]])
    categories = all_metrics + [all_metrics[0]]
    categories = [cat.replace("_", " ") for cat in categories]  # rename :)

    fig = go.Figure()
    for model, metrics in model_metrics.items():
        values = list(metrics.values()) + [list(metrics.values())[0]]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=model
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Relative Model Comparison"
    )

    return fig


def run_pipeline(temperature, top_p, top_k, tfs_z, model, reasoning, prompt_types, question):
    # Reasoning checkbox: Convert to boolean doesnt work as tetx
    use_reasoning = "Reason" in reasoning if reasoning else False

    pipeline_results = ["zero-cot_llama3.2_th-False_7",
                        "few-shot_llama3_th-True_12",
                        "auto-cot_llama3_th-False_0",
                        "auto-cot_llama3_th-False_12",
                        ]
    model_stats = {}
    model_message = ""
    for model_res in pipeline_results:
        res = np.load(exp_path.joinpath(model_res + ".npy"),
                      allow_pickle=True).all()
        stats = res["stats"]
        message = res["message"]
        metrics = res["metrics"]
        # question = res["question"]
        # gt = res["gt"]
        if stats is None:
            print(
                f"model: {model_res} is empty. Expriment failed or incomplete!")
            continue

        model_stats[model_res] = stats
        model_message += f"### {model} Output:\nPrompt: {prompt_types}\nReasoning: {use_reasoning}\nQ: {question}\nA: {message}\n\n"
    fig = plotly_radar_multiple_models(model_stats)
    return model_message, fig


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
        gr.Dropdown(llms.available_llm_models, value="llama3", label="Model",
                    info="Models used for this project"),
        gr.CheckboxGroup(["Reason"], label="Reasoning",
                         info="Reasoning block usage"),
        gr.CheckboxGroup(prompts.available_prompt_engineeing, value=["zero-shot"], label="Prompt",
                         info="Prompt Engineering Methods"),
        gr.Textbox(lines=3, label="Question", info="input to model")
    ]
    outputs = [gr.Textbox(label="Model Output", lines=5),
               gr.Plot()
               ]
    run_btn = gr.Button("Run")
    run_btn.click(fn=run_pipeline, inputs=inputs, outputs=outputs)
    # reset_btn = gr.Button("Reset")
    # reset_btn.click(fn=reset_inputs, inputs=[], outputs=inputs + outputs)

interface.launch(share=False)

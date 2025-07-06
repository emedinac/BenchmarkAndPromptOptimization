import plotly.express as px
import gradio as gr
import plotly.graph_objects as go
import pandas as pd
import ollama_llms as llms
import prompts
from pathlib import Path
import numpy as np
import copy
import app


def plot_generic_bars(model_metrics, normed=False):
    model_metrics = unlink_and_clean_metrics(model_metrics)
    if normed:
        model_metrics = norm(model_metrics)
    df = pd.DataFrame(model_metrics)
    df = df.reset_index().rename(columns={"index": "metric"})
    df_long = df.melt(id_vars="metric", var_name="model", value_name="score")

    # Plot
    fig = px.bar(df_long, x="metric", y="score", color="model",
                 barmode="group", title="Evaluation Metrics per Model")
    fig.update_layout(xaxis_tickangle=-45,
                      height=600)
    return fig


def norm(model_metrics):
    all_models = list(model_metrics.keys())
    all_metrics = list(model_metrics[all_models[0]])
    for me in all_metrics:
        values = []
        for mo in all_models:
            values.append(model_metrics[mo][me])
        maxval, minval = np.max(values), np.min(values)
        values = values/maxval
        for mo, val in zip(all_models, values):
            model_metrics[mo][me + f"  - max_val: {maxval:.2f}"] = val
            del model_metrics[mo][me]
    return model_metrics


def unlink_and_clean_metrics(model_metrics):
    # Unfortunatelly, this is hardcoded.
    # logic: get metric if exists else None
    all_models = list(model_metrics.keys())
    all_metrics = list(model_metrics[all_models[0]])
    # Patch to clean wrong saved metrics.
    for me in all_metrics:
        if not isinstance(me, str):
            for mo in all_models:
                model_metrics[mo][me[0]] = model_metrics[mo][me]
                del model_metrics[mo][me]
    new_model_metrics = {}
    for model_name, metrics in model_metrics.items():
        row = {
            "bleu": metrics["bleu"]["bleu"] if "bleu" in metrics else None,
            "sacrebleu": metrics["sacrebleu"]["score"] if "sacrebleu" in metrics else None,
            "chrf": metrics["chrf"]["score"] if "chrf" in metrics else None,
            "google_bleu": metrics["google_bleu"]["google_bleu"] if "google_bleu" in metrics else None,
            "meteor": metrics["meteor"]["meteor"] if "meteor" in metrics else None,
            "rouge1": metrics["rouge"]["rouge1"] if "rouge" in metrics else None,
            "rouge2": metrics["rouge"]["rouge2"] if "rouge" in metrics else None,
            "rougeL": metrics["rouge"]["rougeL"] if "rouge" in metrics else None,
            "wer": metrics["wer"] if "wer" in metrics else None,
            "cer": metrics["cer"] if "cer" in metrics else None,
            "cer_score": metrics["character"]["cer_score"] if "character" in metrics else None,
            "ter": metrics["ter"]["score"] if "ter" in metrics else None,
            "mauve": metrics["mauve"].mauve if "mauve" in metrics else None,
            "bertscore_f1": metrics["bertscore"]["f1"][0] if "bertscore" in metrics else None,
            "perplexity": metrics["perplexity"]["mean_perplexity"] if "perplexity" in metrics else None,
            "frugalscore": metrics["frugalscore"]["scores"][0] if "frugalscore" in metrics else None,
            "bleurt": metrics["bleurt"]["scores"][0] if "bleurt" in metrics else None,
        }
        new_model_metrics[model_name] = row
    return new_model_metrics


def plotly_radar_metrics(model_metrics: dict):
    model_metrics = unlink_and_clean_metrics(model_metrics)
    fig = plotly_generic_radar_multiple_models(model_metrics)
    return fig


def plotly_generic_radar_multiple_models(model_metrics: dict):
    """
    model_metrics = {
        "llama3": { stats, blabla },
        "mistral": {stats, blabla },
        "ollama": {stats, blabla },
    }
    """
    model_metrics = norm(model_metrics)
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
        height=600,
        showlegend=True,
        title="Execution Performance Model Comparison (Relative)"
    )

    return fig


def run_pipeline(temperature,
                 top_p,
                 top_k,
                 tfs_z,
                 model,
                 reasoning,
                 prompt,
                 question,
                 samples,
                 seed_number,
                 ):
    global exp_path
    # Reasoning checkbox: Convert to boolean doesnt work as tetx
    use_reasoning = "Reason" in reasoning if reasoning else False

    # Manual tests
    # pipeline_results = ["zero-cot_llama3.2_th-False_7",
    #                     "few-shot_llama3_th-True_12",
    #                     "auto-cot_llama3_th-False_0",
    #                     "auto-cot_llama3_th-False_12",
    #                     ]
    pipeline_results = app.run_experiments(exp_path,
                                           temperature,
                                           top_p,
                                           top_k,
                                           tfs_z,
                                           model,
                                           use_reasoning,
                                           prompt,
                                           question,
                                           samples,
                                           seed_number,
                                           )
    pipeline_results = exp_path.glob("*.npy")
    model_stats = {}
    model_metrics = {}
    model_message = ""
    for model_res in pipeline_results:
        model_res = model_res.name
        res = np.load(exp_path.joinpath(model_res), allow_pickle=True).all()
        stats = res["stats"]
        message = res["message"]
        metrics = res["metrics"]
        question = res["question"]
        exp_dict = res["exp_dict"]
        gt = res["gt"]
        if stats is None:
            print(
                f"model: {model_res[:-4]} is empty. Expriment failed or incomplete!")
            continue

        model_stats[model_res[:-4]] = stats
        model_metrics[model_res[:-4]] = metrics
        model_message = model_message + \
            f"### {model} - Output:" + \
            f"\nExperiment configuration: {str(exp_dict)}" + \
            f"\nPrompt: {model_res.split('_')[0]}" + \
            f"\nReasoning: {use_reasoning}" + \
            f"\n--GT: {gt}" + \
            f"\n--Q: {question}" + \
            f"\n--A: {message}\n=========================\n\n\n"

    fig1 = plotly_generic_radar_multiple_models(copy.deepcopy(model_stats))
    fig2 = plotly_radar_metrics(copy.deepcopy(model_metrics))
    fig3 = plot_generic_bars(copy.deepcopy(model_metrics), normed=False)
    fig4 = plot_generic_bars(copy.deepcopy(model_metrics), normed=True)
    return model_message, fig1, fig2, fig3, fig4


def main(args):
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
        ]
        with gr.Row():
            with gr.Row():
                input_text = gr.Textbox(lines=7, label="Question",
                                        info="input to model")
            with gr.Column():
                input_number = gr.Number(value=1, label="Samples",
                                         info="Number of samples to use obtained from the financial databases. If you want to use test cases, set this to 0.")
                seed_number = gr.Number(value=0, label="Seed",
                                        info="Seed to obtain samples from the database")
        inputs = inputs + [input_text, input_number, seed_number]

        run_btn = gr.Button("Run")  # Run the pipeline

        with gr.Row():
            outputs_text = gr.Textbox(label="Model Output", lines=5)
            with gr.Row():
                plot1 = gr.Plot(label="Execution Stats")
                plot2 = gr.Plot(label="Metrics")
            with gr.Row():
                plot3 = gr.Plot(label="Metrics 2")
                plot4 = gr.Plot(label="Metrics 3")
        outputs = [outputs_text, plot1, plot2, plot3, plot4]

        run_btn.click(fn=run_pipeline, inputs=inputs, outputs=outputs)

    interface.launch(share=False)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Run the Benchmark and Prompt Optimization Platform")
    parser.add_argument("--exp_path", type=str, default="results",
                        help="Path to save the results of the experiments")
    args = parser.parse_args()
    exp_path = Path(args.exp_path)
    exp_path.mkdir(parents=True, exist_ok=True)
    main(exp_path)

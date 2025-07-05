# Benchmark and Prompt Optimization Platform for Offline Models

Benchmark and Prompt Optimization Platform for Offline Models (Locally‑Hosted) LLMs for Financial Q&A in the stock market text data.

## Locally‑Hosted LLM for Financial Q&A

A fully offline system leveraging Ollama to run local LLMs for experimenting with prompt engineering techniques, including Chain-of-Thought (CoT) and Self-Consistency on answering finance-related questions, particularly around stock market data and trends. This tool is designed for fast iteration, performance benchmarking, and analysis of prompt effectiveness in a domain where precision matters. Includes interactive parameter tuning (for the LLM and metrics), metrics visualization, and an experimentation workflow.

Tech stacks and concepts applied:

- Ollama: LLM framework with request and BeautifulSoup for url search.
- Chain‑of‑Thought (CoT): Guides the model to generate intermediate reasoning steps before answering, improving clarity and problem-solving accuracy
- Self‑Consistency: Runs multiple zero/few-shot CoT generations and picks the most similar and voted answer (via cosine similarity for embeddings), reducing single‑path bias and boosting reliability.
- Huggungface environemnt: Load and evaluate different dataset.
- Gradio: minimal and simple Front-end framework for software integration between AI and other software skills. It was used for visualziation. Not only test prompts, but facilitate iterative design with metrics.

Specifically, Add:

- [X] Ollama: with different input parameters (e.g. temperature, context, etc).
- [X] Simple and minimal url search functionality (request and BeautifulSoup).
- [X] zero-shot.
- [X] few-shot.
- [X] zero-shot CoT.
- [X] few-shot CoT.
- [X] Self-Consistency.
- [X] several metrics using Huggingface lib.
- [X] Dataset Loader (Huggingface datasets only)
- [X] Interface and Visualization with Gradio.
- [X] Interactivity in the visualizations.
- [ ] Set main pipeline and argument customazation.
- [ ] Support for multiple parallel experiments.



## Install

```bash
python3.10 -m venv mvp1
source mvp1/bin/activate
pip install --upgrade pip
pip install uv
uv init . 
uv sync --active
```

Add more packages, not all metrics (from evaluate) were installed for this demo.

```bash
uv add <package> --active
```

Install Ollama - Should be done via Dockerfile later on.

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

## Run

Tested models: `llama3`, `llama3.2`, `deepseek-r1`, `gemma3`, `qwen3`

Run `python3 src/prepare_dataset.py` to get the embeddings, but the main code also compute them if they are not present :) 

Run the main logic: `python3 main.py`

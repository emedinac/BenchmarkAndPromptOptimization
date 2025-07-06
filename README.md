# Prompt Benchmark Interface for Offline Models

Prompt Benchmark Interface for Offline Models (Locally‑Hosted) LLMs for Financial Q&A in the stock market text data.

![Demo animation](figures/demo.gif)

A fully offline system leveraging Ollama to run local LLMs for experimenting with prompt engineering techniques, including Chain-of-Thought (CoT) and Self-Consistency on answering finance-related questions, particularly around stock market data and trends. This tool is designed for fast iteration, performance benchmarking, and analysis of prompt effectiveness in a domain where precision matters. Includes interactive parameter tuning (for the LLM and metrics), metrics visualization, and an experimentation workflow.

Tech stacks and concepts applied:

- Ollama: LLM framework with request and BeautifulSoup for url search.
- Chain‑of‑Thought (CoT): Guides the model to generate intermediate reasoning steps before answering, improving clarity and problem-solving accuracy
- Self‑Consistency: Runs multiple zero/few-shot CoT generations and picks the most similar and voted answer (via cosine similarity for embeddings), reducing single‑path bias and boosting reliability.
- Huggungface environemnt: Load and evaluate different dataset.
- Gradio: minimal and simple Front-end framework for software integration between AI and other software skills. It was used for visualziation. Not only test prompts, but facilitate iterative design with metrics.

Specifically, Added:

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
- [X] Set main pipeline and argument customazation.
- [X] Support for multiple parallel experiments.
- [X] Support for non-dataset samples (setting 1 manual ground truth)

## Install

```bash
python3.10 -m venv prj1
source prj1/bin/activate
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

### Run

Run the main logic:

```bash
python3 interface.py
```

IMPORTANT: Set samples to 0 in the interface to accept input texts (Question section).
IMPORTANT 2: The code is slow in the first run, later it uses precomputed values.

### Short explaination

This repo is base on 4 finantial datasets collected from huggingface (online available): `jyanimaulik/yahoo_finance_stock_market_news`, `davzoku/moecule-stock-market-outlook`, `Ubaidbhat/stock_market_basics`, `yc4142/stockmarketCoT`

Tested models: `llama3`, `llama3.2`, `deepseek-r1`, `gemma3`, `qwen3`

Tested metrics: `exact_match`,  `bertscore`, `bleu`, `bleurt`, `cer`, `character`, `chrf`, `frugalscore`, `google_bleu`, `mauve`, `meteor`, `"perplexity`, `rouge`, `sacrebleu`, `ter`, `wer`. NOTE: All the metrics are computed, metric selection was not implemented. Change metric configuration is needed (some warning appear here), my GPU was small (8GB): for instance: ('bleurt', 'bleurt-large-512').

The parameters (e.g. Temperature) showed in Gradio are saved directly but not displayed nicely :'(. A folder called `restuls` (or optionally set this input via argument) is created to record all experiments. The code automatically load all `npy` files located inside.

If the input text is not part fromthe dataset, different metrics are computed. For embedding similarity, etc.

### Embeddings Dataset Preprocessing

If this is the first time running the code. Run `python3 src/prepare_dataset.py` to get the embeddings, but the main code also compute them if they are not present :)

This is used for the auto-CoT, this algorithm computes the K-means of the embedding space and is used to generate the CoT before calling the LLM response.


from prompts import Prompter, available_prompt_engineeing_tpye
import ollama_llms as llms
import pandas as pd
from metrics import Metrics
import prepare_dataset


def save_results():
    results[f"{prompt}_{model}_th-{th}_{idx}"] = {}
    results[f"{prompt}_{model}_th-{th}_{idx}"]["message"] = message
    results[f"{prompt}_{model}_th-{th}_{idx}"]["stats"] = stats
    results[f"{prompt}_{model}_th-{th}_{idx}"]["metrics"] = scores


if __name__ == '__main__':
    from inputs import test_texts
    used_metrics = ["exact_match",
                    ("bertscore", "distilbert-base-uncased"),
                    # ("bertscore", "prajjwal1/bert-small"),
                    "bleu",
                    "bleurt",
                    "cer",
                    "character",
                    "chrf",
                    "frugalscore",
                    # ("glue", "stsb"),
                    # ("glue", "rte"),
                    # ("glue", "wnli"),
                    # ("glue", "cola"),
                    "google_bleu",
                    "mauve",
                    "meteor",
                    ("perplexity", "metric"),
                    # "poseval",
                    "rouge",
                    # "squad_v2",
                    "sacrebleu",
                    # ("super_glue", "cb"),
                    # ("super_glue", "rte"),
                    # ("super_glue", "wic"),
                    # ("super_glue", "copa"),
                    "ter",
                    "wer",
                    ]
    metrics = Metrics(used_metrics)

    try_test_cases_scanerios = False
    if try_test_cases_scanerios:
        input_texts = test_texts
        ground_truth = [None] * test_texts
    else:
        input_texts, ground_truth = prepare_dataset.build_validation(samples=15,
                                                                     seed=0)
    test_models = ['llama3',
                   'llama3.2',
                   'deepseek-r1',
                   'gemma3',
                   'qwen3',
                   ]

    think = [False, True]

    # Running a loop with several models for massive tests.
    # test_texts can be passsed in a single forward pass, but independent tests are wanted in this demo.

    results = {}
    beshort = True
    prompter = Prompter(beshort=beshort, examples_to_use=None)
    for model in test_models:
        skip_model = False
        for th in think:
            for prompt in available_prompt_engineeing_tpye:
                # for prompt in ["self-consistency"]:
                for idx, question in enumerate(input_texts):
                    print(
                        f"{prompt}_{model}_th-{th}_{idx}: {question['content']}")
                    if skip_model:
                        results[f"{prompt}_{model}_th-{th}_{idx}"] = {}
                        results[f"{prompt}_{model}_th-{th}_{idx}"]["message"] = ""
                        results[f"{prompt}_{model}_th-{th}_{idx}"]["stats"] = None
                        results[f"{prompt}_{model}_th-{th}_{idx}"]["metrics"] = None
                        continue
                    reference = ground_truth[idx]
                    prompted_question = prompter.apply(question, prompt)
                    message, stats = llms.runLLM(model=model,
                                                 messages=[
                                                     prompted_question
                                                 ],
                                                 think=th)
                    if metrics is None:
                        skip_model = True
                    if reference is not None and len(message) > 0:
                        scores = metrics.compute(message, reference)
                    else:
                        scores = None
                    # save experiment
                    results[f"{prompt}_{model}_th-{th}_{idx}"] = {}
                    results[f"{prompt}_{model}_th-{th}_{idx}"]["message"] = message
                    results[f"{prompt}_{model}_th-{th}_{idx}"]["stats"] = stats
                    results[f"{prompt}_{model}_th-{th}_{idx}"]["metrics"] = scores
                    print(f"{scores}")
    print("end")

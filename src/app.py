
import numpy as np
import prepare_dataset
from metrics import Metrics
import ollama_llms as llms
from prompts import Prompter, available_prompt_engineeing
import time
from inputs import test_texts

# Test on Laptop GPU
# Temporal solution for 1 GPU.
# Ollama doesnt have space to run on large text inputs.
# However it works pretty well :D
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
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


def run_experiments(temperature,
                    top_p,
                    top_k,
                    tfs_z,
                    model,
                    reasoning,
                    prompt_types,
                    question,
                    try_test_cases_scanerios=False):

    if try_test_cases_scanerios:
        input_texts = test_texts
        ground_truth = [None] * test_texts
    else:
        input_texts, ground_truth = prepare_dataset.build_validation(samples=15,
                                                                     seed=0)
    metrics = Metrics(used_metrics)
    think = [False, True]

    results = {}
    beshort = True
    prompter = Prompter(beshort=beshort, examples_to_use=None)
    exp = f"{prompt}_{model}_th-{th}_{idx}"
    for idx, question in enumerate(input_texts):
        print(
            f"{exp}: {question['content']}")
        if skip_model:
            results[f"{exp}"] = {}
            results[f"{exp}"]["message"] = ""
            results[f"{exp}"]["stats"] = None
            results[f"{exp}"]["metrics"] = None
            continue
        reference = ground_truth[idx]
        t1 = time.time()
        prompted_question = prompter.apply(question, prompt)
        message, stats = llms.runLLM(model=model,
                                        messages=[
                                            prompted_question
                                        ],
                                        think=th)
        t2 = time.time()
        stats["total_execution_time_sec"] = t2-t1
        if metrics is None:
            skip_model = True
        if reference is not None and len(message) > 0:
            print("computing metrics")
            scores = metrics.compute(message, reference)
        else:
            scores = None
        # save experiment
        results[f"{exp}"] = {}
        results[f"{exp}"]["message"] = message
        results[f"{exp}"]["question"] = question['content']
        results[f"{exp}"]["gt"] = reference
        results[f"{exp}"]["stats"] = stats
        results[f"{exp}"]["metrics"] = scores
        np.save(f"results/{exp}.npy",
                results[f"{exp}"]
                )
        print(f"{scores}")

    np.save("results/results.npy", results)
    print("end")


import numpy as np
import prepare_dataset
from metrics import Metrics
import ollama_llms as llms
from prompts import Prompter, available_prompt_engineeing
import time

# Test on Laptop GPU
# Temporal solution for 1 GPU.
# Ollama doesnt have space to run on large text inputs.
# However it works pretty well :D
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

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

    think = [False, True]

    # Running a loop with several models for massive tests.
    # test_texts can be passsed in a single forward pass, but independent tests are wanted in this demo.

    results = {}
    beshort = True
    prompter = Prompter(beshort=beshort, examples_to_use=None)
    for model in llms.available_llm_models:
        skip_model = False
        for th in think:
            for prompt in available_prompt_engineeing:
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
                    results[f"{prompt}_{model}_th-{th}_{idx}"] = {}
                    results[f"{prompt}_{model}_th-{th}_{idx}"]["message"] = message
                    results[f"{prompt}_{model}_th-{th}_{idx}"]["question"] = question['content']
                    results[f"{prompt}_{model}_th-{th}_{idx}"]["gt"] = reference
                    results[f"{prompt}_{model}_th-{th}_{idx}"]["stats"] = stats
                    results[f"{prompt}_{model}_th-{th}_{idx}"]["metrics"] = scores
                    np.save(f"results/{prompt}_{model}_th-{th}_{idx}.npy",
                            results[f"{prompt}_{model}_th-{th}_{idx}"]
                            )
                    print(f"{scores}")

    np.save("results/results.npy", results)
    print("end")

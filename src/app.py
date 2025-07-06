
import numpy as np
import prepare_dataset
from metrics import Metrics
import ollama_llms as llms
from prompts import Prompter
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


def run_experiments(exp_path,
                    temperature,
                    top_p,
                    top_k,
                    tfs_z,
                    model,
                    reasoning,
                    prompt,
                    question,
                    samples,
                    seed,
                    ):
    if samples == 0:
        input_texts = test_texts
        ground_truth = [None] * test_texts
    else:
        input_texts, ground_truth, idxs = prepare_dataset.build_validation(samples=samples,
                                                                           seed=seed)
    metrics = Metrics(used_metrics)

    results = {}
    beshort = True
    prompter = Prompter(beshort=beshort, examples_to_use=None)
    for pro in prompt:
        for i, (idx, question) in enumerate(zip(idxs, input_texts)):
            exp = f"{pro}_{model}_th-{reasoning}_id{idx}"
            if exp_path.joinpath(exp + ".npy").exists():
                print(f"Experiment {exp} already exists, skipping...")
                continue
            print(f"{exp}: {question['content']}")
            reference = ground_truth[i]
            t1 = time.time()
            prompted_question = prompter.apply(question, pro)
            message, stats = llms.runLLM(model=model,
                                         messages=[
                                             prompted_question
                                         ],
                                         think=reasoning,
                                         options={
                                             "temperature": temperature,
                                             "top_k": top_k,
                                             "top_p": top_p,
                                             "tfs_z": tfs_z,
                                             # for some tests.
                                             "repeat_penalty": 1.2,
                                             "mirostat": 1.0, }
                                         )

            t2 = time.time()
            stats["total_execution_time_sec"] = t2-t1
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
            exp_dict = {"temperature": temperature,
                        "top_p": top_p,
                        "top_k": top_k,
                        "tfs_z": tfs_z,
                        "model": model,
                        "reasoning": reasoning,
                        "prompt": pro,
                        "samples": samples,
                        "seed": seed
                        }
            results[f"{exp}"]["exp_dict"] = exp_dict
            np.save(f"results/{exp}.npy",
                    results[f"{exp}"]
                    )
            print(f"{scores}")
    print("Processing done.")

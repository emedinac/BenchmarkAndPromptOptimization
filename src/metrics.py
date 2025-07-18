import evaluate
import torch
from sentence_transformers import SentenceTransformer

# f1_metric = evaluate.load("f1")
# em_metric = evaluate.load("exact_match")
# rouge_metric = evaluate.load("rouge")
# meteor_metric = evaluate.load("meteor")


class Metrics:
    def __init__(self, list_of_metrics: list[str],
                 model_id: str = "all-mpnet-base-v2"
                 ):
        """
        Get metric names from: https://huggingface.co/metrics
        """
        self.embedder = SentenceTransformer(model_id)
        self.list_of_metrics = list_of_metrics
        self.metrics = {}
        for m in list_of_metrics:
            if isinstance(m, str) or m == "bertscore":
                self.metrics[m] = evaluate.load(m)
            elif isinstance(m, (tuple, list)):
                name, conf = m
                self.metrics[f"{name}-{conf}"] = evaluate.load(name,
                                                               config_name=conf)

    def get_embeddings(self, input_text):
        return self.embedder.encode(input_text)

    def compute(self, pred, ref):
        # Ensure format consistency
        pred = pred.strip()
        ref = ref.strip()
        results = {}
        with torch.no_grad():  # no idea why this works for metrics :D
            for m in self.list_of_metrics:
                if isinstance(m, str):
                    metric = self.metrics[m]
                elif isinstance(m, (tuple, list)):
                    name, conf = m
                    metric = self.metrics[f"{name}-{conf}"]
                kwargs = {}
                if "bertscore" in m:
                    kwargs = {"model_type": conf}
                elif "perplexity" in m:
                    kwargs = {"model_id": "distilgpt2", "device": "cpu"}
                results[m] = metric.compute(predictions=[pred],
                                            references=[ref],
                                            **kwargs)
        torch.cuda.empty_cache()
        return results

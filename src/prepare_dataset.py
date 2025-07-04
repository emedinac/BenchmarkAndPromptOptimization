# My own implementation of Auto-cot

from datasets import load_dataset, load_from_disk, concatenate_datasets
from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path


def get_dataset():
    # test dataset - publicly available for stock market.
    ds1 = load_dataset(
        "jyanimaulik/yahoo_finance_stock_market_news", split="train")
    ds2 = load_dataset(
        "davzoku/moecule-stock-market-outlook", split="train")
    ds3 = load_dataset("Ubaidbhat/stock_market_basics", split="train")
    ds4 = load_dataset("yc4142/stockmarketCoT", split="train")
    ds1 = ds1.rename_columns({"instruction": "Question",
                              "input": "Answer"}).select_columns(["Question", "Answer"])
    ds2 = ds2.select_columns(["Question", "Answer"])
    ds3 = ds3.rename_columns({
        "question": "Question",
        "answer":   "Answer"
    }).select_columns(["Question", "Answer"])
    ds4 = ds4.rename_columns({"instruction": "Question",
                              "output": "Answer"}).select_columns(["Question", "Answer"])

    merged_dataset = concatenate_datasets([ds1, ds2, ds3, ds4])
    return merged_dataset


def build_validation(samples: int, seed: int = 0):
    np.random.seed(seed)
    dataset = get_dataset()
    dataset_texts = []
    ground_truth = []
    dataset_lenght = len(dataset)
    for _ in range(samples):
        data = dataset[np.random.randint(dataset_lenght)]
        dataset_texts.append({'role': 'user',
                              'context': 'I am an active investor and I want to increase my profit in the next years',
                              'content': data["Question"]})
        ground_truth.append(data["Answer"])
    return dataset_texts, ground_truth


def get_dataset_embeddings(embeddings_path: str = "../stock_market_embeddings.npy",
                           dataset_path: str = "../stock_market_datasets"):
    if not Path(embeddings_path).exists():
        merged_dataset = get_dataset()
        questions = merged_dataset["Question"]
        # The all-mpnet-base-v2 model provides the best quality, while all-MiniLM-L6-v2 is 5 times faster and still offers good quality.
        # https://sbert.net/docs/sentence_transformer/pretrained_models.html
        embedder = SentenceTransformer('all-mpnet-base-v2')
        embeddings = embedder.encode(questions, show_progress_bar=True)
        np.save("stock_market_embeddings.npy", embeddings)
    else:
        embeddings = np.load(embeddings_path)
        merged_dataset = load_from_disk(dataset_path)
    return embeddings, merged_dataset


if __name__ == '__main__':
    get_dataset_embeddings()
    print("npy file with bmbeddings created...")

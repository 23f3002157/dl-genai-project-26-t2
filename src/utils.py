import os
import random
import numpy as np
import yaml


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def map_at_3(predictions: list, labels: list) -> float:
    """
    predictions: list of ["A","B","C"] per sample
    labels: list of ground truth e.g. "A"
    """
    scores = []
    for preds, label in zip(predictions, labels):
        score = 0.0
        for rank, pred in enumerate(preds[:3], start=1):
            if pred == label:
                score = 1.0 / rank
                break
        scores.append(score)
    return float(np.mean(scores))
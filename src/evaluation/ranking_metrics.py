# %%
import numpy as np


def ranking_metrics_at_5_percent(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if y_true.ndim != 1 or y_score.ndim != 1:
        raise ValueError("y_true e y_score devem ser vetores de uma dimensão.")

    if len(y_true) == 0 or len(y_true) != len(y_score):
        raise ValueError("y_true e y_score devem ter o mesmo tamanho e não estar vazios.")

    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("y_true deve conter apenas 0 e 1.")

    if not np.all(np.isfinite(y_score)):
        raise ValueError("y_score deve conter apenas valores finitos.")

    top_n = int(np.ceil(len(y_true) * 0.05))
    top_indices = np.argsort(-y_score, kind="stable")[:top_n]
    positives_at_5_percent = int(y_true[top_indices].sum())
    total_positives = int(y_true.sum())

    precision = positives_at_5_percent / top_n
    recall = positives_at_5_percent / total_positives if total_positives else 0.0
    positive_rate = total_positives / len(y_true)
    lift = precision / positive_rate if positive_rate else 0.0

    return {
        "precision_at_5_percent": precision,
        "recall_at_5_percent": recall,
        "lift_at_5_percent": lift,
    }

# %
import mlflow
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# %%

def evaluate_model(model, X, y, split, threshold = 0.5):
    if mlflow.active_run() is None:
        raise RuntimeError("A avaliação deve ocorrer dentro de mlflow.start_run().")

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels = [0, 1]
    ).ravel()

    metrics = {
        f"{split}_average_precision": average_precision_score(y, probabilities),
        f"{split}_precision": precision_score(y, predictions, zero_division = 0),
        f"{split}_recall": recall_score(y, predictions, zero_division = 0),
        f"{split}_f1": f1_score(y, predictions, zero_division = 0),
        f"{split}_positive_rate": float(y.mean()),
        f"{split}_threshold": threshold,
        f"{split}_tn": int(tn),
        f"{split}_fp": int(fp),
        f"{split}_fn": int(fn),
        f"{split}_tp": int(tp),
    }

    mlflow.log_metrics(metrics)
    return metrics

# %%

def evaluate_candidates(
    candidates,
    X_test,
    y_test,
    split = "test",
    threshold = 0.5
):
    results = {}

    for name, run_id in candidates.items():

        model_uri = f"runs:/{run_id}/model"
        model = mlflow.sklearn.load_model(model_uri)

        with mlflow.start_run(run_id = run_id):
            metrics = evaluate_model(
                model = model,
                X = X_test,
                y = y_test,
                split = split,
                threshold = threshold
            )

        results[name] = metrics

        print(f"\nModelo: {name} | Conjunto: {split}")
        print(metrics)

    return results

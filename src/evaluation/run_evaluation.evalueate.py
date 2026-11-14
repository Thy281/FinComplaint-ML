# %%

import mlflow
import sys
from pathlib import Path

project_root = Path.cwd()
while project_root != project_root.parent and not (project_root / "src/training/split.py").exists():
    project_root = project_root.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.evaluation.evaluate import evaluate_candidates
from src.training.split import load_data, validate_dates, split_data, split_features_target

# %%

mlflow.set_tracking_uri("http://127.0.0.1:5000")

df = validate_dates(load_data())
_, test, _ = split_data(df)
X_test, y_test = split_features_target(test)

experiment = mlflow.get_experiment_by_name("model-received")
if experiment is None:
    raise ValueError("Experimento 'model-received' não encontrado no MLflow.")

candidate_names = {
    "BernoulliNB_original",
    "XGBoost_over"
}
candidates = {}

runs = mlflow.search_runs(
    experiment_ids = [experiment.experiment_id],
    filter_string = "attributes.status = 'FINISHED'",
    order_by = ["attributes.start_time DESC"],
    output_format = "list"
)

for run in runs:
    name = run.data.tags.get("mlflow.runName")
    if name in candidate_names and name not in candidates:
        candidates[name] = run.info.run_id

missing = candidate_names - candidates.keys()
if missing:
    raise ValueError(f"Execuções concluídas não encontradas: {', '.join(sorted(missing))}")

test_results = evaluate_candidates(
    candidates = candidates,
    X_test = X_test,
    y_test = y_test
)
# %%

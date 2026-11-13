# %%
import mlflow
import mlflow.sklearn
import sys
from pathlib import Path

project_root = Path.cwd()
while project_root != project_root.parent and not (project_root / "src/training/split.py").exists():
    project_root = project_root.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.training.split import (
    load_data,
    validate_dates,
    split_data,
    split_features_target,
    validate_split
)
from src.training.pipeline import get_models, create_grid_search

# %%
def main():
    # 1. Carregar a ABT
    df = load_data()

    # 2. Validar e ordenar cronologicamente
    df = validate_dates(df)

    # 3. Separar treino, teste e OOT
    train, test, oot = split_data(df)

    # 4. Separar features e target
    X_train, y_train = split_features_target(train)

    # 5. Guardar as datas para validação temporal
    train_dates = train["date_received"].reset_index(drop = True)

    # 6. Conferir os dados
    validate_split(X_train, y_train, "Treino")

    return train_models(X_train, y_train, train_dates)

# %%

def train_models(X_train, y_train, train_dates):
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    mlflow.set_experiment("model-received")

    models = get_models()
    results = {}

    sampling_methods = [None, "over", "under"]

    for name, config in models.items():
        for sampling in sampling_methods:

            sampling_name = sampling or "original"
            run_name = f"{name}_{sampling_name}"

            print(f"\nTreinando: {run_name}")

            with mlflow.start_run(run_name = run_name):
                grid = create_grid_search(
                    model = config["model"],
                    params = config["params"],
                    train_dates = train_dates,
                    sampling = sampling
                )
                grid.fit(X_train, y_train)

                mlflow.log_params(grid.best_params_)
                mlflow.log_params({
                    "algorithm": name,
                    "sampling": sampling_name
                })
                mlflow.log_metrics({
                    "cv_average_precision": grid.best_score_,
                    "cv_f1": grid.cv_results_["mean_test_F1"][grid.best_index_],
                    "cv_recall": grid.cv_results_["mean_test_Recall"][grid.best_index_],
                    "cv_precision": grid.cv_results_["mean_test_Precision"][grid.best_index_]
                })
                mlflow.sklearn.log_model(
                    sk_model = grid.best_estimator_,
                    name = "model",
                    serialization_format = "cloudpickle"
                )

                results[run_name] = {
                    "best_model": grid.best_estimator_,
                    "best_params": grid.best_params_,
                    "best_score": grid.best_score_
                }

                print(f"AP: {grid.best_score_:.4f}")

    return results

# %%
if __name__ == "__main__":
    main()
    
# %%

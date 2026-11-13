# %%

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
    models = get_models()
    results = {}

    for name, config in models.items():
        print(f"\nTreinando: {name}")

        model = config["model"]
        params = config["params"]

        grid = create_grid_search(
            model = model,
            params = params,
            train_dates = train_dates,
            sampling = "over"
        )

        grid.fit(X_train, y_train)

        results[name] = {
            "best_model": grid.best_estimator_,
            "best_params": grid.best_params_,
            "best_score": grid.best_score_,
            "grid": grid
        }

        print(f"Melhor AP: {grid.best_score_:.4f}")
        print(f"Melhores parâmetros: {grid.best_params_}")

    return results

# %%
if __name__ == "__main__":
    main()
    
# %%

# %%

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.naive_bayes import BernoulliNB
import numpy as np
import pandas as pd

# %%

def get_features():
    numeric_features = [
        "received_year",
        "received_month",
        "received_dayofweek"
    ]

    categorical_features = [
        "product",
        "sub_product",
        "issue",
        "sub_issue",
        "company",
        "state",
        "submitted_via"
    ]

    return numeric_features, categorical_features

# %%

def create_preprocessor():
    numeric_features, categorical_features = get_features()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy = "median"
        ))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy = "constant",
            fill_value = "Missing"
        )),
        ("onehot", OneHotEncoder(
            handle_unknown = "ignore"
        ))
    ])

    # Direciona cada grupo para seu Pipeline especifico
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

# %%

def create_bernoulli_preprocessor():
    numeric_features, categorical_features = get_features()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy = "most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown = "ignore"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(
            strategy = "constant",
            fill_value = "Missing"
        )),
        ("onehot", OneHotEncoder(handle_unknown = "ignore"))
    ])

    # Direciona cada grupo para seu Pipeline especifico
    return ColumnTransformer([
        ("calendar", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

# %%

def create_pipeline(model):
    if isinstance(model, BernoulliNB):
        preprocessor = create_bernoulli_preprocessor()
    else:
        preprocessor = create_preprocessor()

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

# %%

def get_models():
    return {
        "LogisticRegression": {
            "model": LogisticRegression(
                max_iter = 2000,
                class_weight = "balanced"
            ),
            "params": {
                "model__C": [0.1, 1, 10]
            }
        },

        "RandomForest": {
            "model": RandomForestClassifier(
                random_state = 42,
                class_weight = "balanced",
                n_jobs = 1
            ),
            "params": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [10, 20]
            }
        },

        "XGBoost": {
            "model": XGBClassifier(
                random_state = 42,
                tree_method = "hist",
                eval_metric = "logloss",
                n_jobs = 1
            ),
            "params": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [3, 6],
                "model__learning_rate": [0.05, 0.1]
            }
        },

        # Navie-Bayes
        "BernoulliNB": {
            "model": BernoulliNB(binarize = None),
            "params": {
                "model__alpha": [0.1, 0.5, 1.0],
            }
        }
    }

# %%


import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


def create_temporal_cv(dates, n_splits=5):
    dates = pd.to_datetime(dates).reset_index(drop=True)

    unique_dates = np.sort(dates.unique())

    tscv = TimeSeriesSplit(n_splits=n_splits)

    folds = []

    for train_idx, val_idx in tscv.split(unique_dates):
        train_dates = unique_dates[train_idx]
        val_dates = unique_dates[val_idx]

        train_rows = np.flatnonzero(
            dates.isin(train_dates).to_numpy()
        )

        val_rows = np.flatnonzero(
            dates.isin(val_dates).to_numpy()
        )

        folds.append((train_rows, val_rows))

    return folds


# %%

def create_grid_search(model, params):
    pipeline = create_pipeline(model)

    cv = create_temporal_cv(
        dates = train_dates,
        n_split = 5
    )

    return GridSearchCV(
        estimator = pipeline,
        param_grid = params,
        scoring = "average_precision",
        cv = cv,
        n_jobs = 2,
        verbose = 2
    )

# %%

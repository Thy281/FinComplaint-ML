# %%

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

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

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

# %%

def create_pipeline(model):
    return Pipeline([
        ("preprocessor", create_preprocessor()),
        ("model", model)
    ])

# %%

def get_param_grid():
    return {
        "model__n_estimators": [100, 200],
        "model__max_depth": [10, 20],
        "model__min_samples_leaf": [2, 5]
    }

# %%

def create_grid_search():
    model = RandomForestClassifier(
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    pipeline = create_pipeline(model)

    cv = TimeSeriesSplit(n_splits=5)

    return GridSearchCV(
        estimator=pipeline,
        param_grid=get_param_grid(),
        cv=cv,
        scoring="average_precision",
        n_jobs=2,
        verbose=2
    )

# %%

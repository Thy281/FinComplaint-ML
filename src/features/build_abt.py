# %%

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data/interim/complaints_clean.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data/processed/ABT.parquet"
# %%

def build_abt():
    df = pd.read_parquet(INPUT_PATH)

    print("Dimensoes:", df.shape)

    print("\nColunas disponiveis:")
    print(df.columns.to_list())

    print("\nDistribuicao do target:")
    print(df["late_response"].value_counts(dropna = False))

    return df

# %%

def create_features(df):
    df = df.copy()

    df["date_received"] = pd.to_datetime(
        df["date_received"],
        errors = "coerce" # NaT
    )

    # Extrair info da data da reclamacao
    df["received_year"] = df["date_received"].dt.year
    df["received_month"] = df["date_received"].dt.month
    df["received_dayofweek"] = df["date_received"].dt.dayofweek

    return df

# %%
def select_features(df):
    df = df.copy()

    target = "late_response"

    features = [
        "product",
        "sub_product",
        "issue",
        "sub_issue",
        "company",
        "state",
        "submitted_via",
        "received_year",
        "received_month",
        "received_dayofweek"
    ]

    expected_columns = ["date_received"] + features + [target]
    missing_columns = [col for col in expected_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Colunas esperadas ausentes na ABT: {', '.join(missing_columns)}")

    abt = df[expected_columns].copy()

    return abt

# %%

def save_abt(df):
    OUTPUT_PATH.parent.mkdir(parents = True, exist_ok = True)
    df.to_parquet(OUTPUT_PATH, index = False)

    print("ABT salva com sucesso!")

if __name__ == "__main__":
    df = build_abt()
    df = create_features(df)
    abt = select_features(df)
    print("Dimensoes:", abt.shape)
    print("Colunas:", abt.columns.to_list())
    save_abt(abt)

# %%

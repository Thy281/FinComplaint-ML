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
# Inputacao de features

def select_features(df):
    df = df.copy()

    target = "late_response"

    drop_columns = [
        "timely_response",
        "company_response_to_consumer",
        "company_public_response",
        "consumer_disputed?",
        "date_sent_to_company",
        "complaint_id",
        "consumer_complaint_narrative",
        "consumer_consent_provided",
        "tags",
        "zipcode"
    ]

    # Selecao indireta das features
    features = df.columns.drop(
        [target, "date_received"] + drop_columns
    )

    abt = df[["date_received"] + features.tolist() + [target]].copy()

    return abt

abt = select_features(df)

print("Dimensoes:", abt.shape)
print("Colunas:", abt.columns.to_list())


# %%

def save_abt(df):
    OUTPUT_PATH.parent.mkdir(parents = True, exist_ok = True)
    df.to_parquet(OUTPUT_PATH, index = False)

    print("ABT salva com sucesso!")

if __name__ == "__main__":
    df = build_abt()
    df = create_features(df)
    abt = select_features(df)
    save_abt(abt)

# %%

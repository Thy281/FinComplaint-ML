# %%

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data/raw/consumer_complaints.csv"
OUTPUT_PATH = PROJECT_ROOT / "data/interim/complaints_clean.parquet"

df = pd.read_csv(RAW_PATH, low_memory = False)

print(df.shape)
print(df.columns.to_list())

# %%
# Corrigindo Datas

def convert_dates(df):
    df = df.copy()

    date_columns = [
        "date_received",
        "date_sent_to_company"
    ]

    for col in date_columns:
        df[col] = pd.to_datetime(
            df[col],
            errors = "coerce"
        )
    return df

df = convert_dates(df)

print(df[[
    "date_received",
    "date_sent_to_company"
]].dtypes)
df.head()

# %%
# Validar Datas

def validate_dates(df):
    columns = [
        "date_received",
        "date_sent_to_company"
    ]

    print("Valores ausentes nas datas: ")
    print(df[columns].isna().sum())

    invalid_order = (
        df["date_sent_to_company"]
        < df["date_received"]
    )

    print(
        "Datas de envio anteriores ao recebimento:",
        invalid_order.sum()
    )
    return df

df = validate_dates(df)

# %%
# Duplicadas

def remove_duplicates(df):
    df = df.copy()
    print(f"Antes: {len(df)}")

    df = df.drop_duplicates()
    print(f"Depois: {len(df)}")

    return df
df = remove_duplicates(df)

# %%





# %%

import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH =  PROJECT_ROOT / "data/processed/ABT.parquet"

# %%

def load_data():
    df = pd.read_parquet(INPUT_PATH)

    print("Dimensoes:", df.shape)
    print("Periodo inicial:", df["date_received"].min())
    print("Periddo final:", df["date_received"].max())

    return df

# %%

def validate_dates(df):
    df = df.copy()

    missing_dates = df["date_received"].isna().sum()
    print("Datas ausentes:", missing_dates)

    # Impedir divisao temporal com datas ausentes
    if missing_dates > 0:
        raise ValueError("Existem datas ausentes no dataset.")

    # Ordenar da data mais antiga para a mais recente
    df = df.sort_values(
        by = "date_received"
    ).reset_index(drop = True)

    return df

# %%

def split_data(df):
    df = df.copy()

    # Definir proporcoes
    train_size = 0.7
    test_size = 0.15

    # Calcular os pontos de corte
    n = len(df)

    # N = numero de linhas do df (555.957)
    train_end = int(n * train_size)
    test_end = int(n * (train_size + test_size))

    # Obter as datas de corte
    # Aqui vamos pegar a linha ao qual temos o valor de 70% do treino e 85% do teste com o iloc
    train_cutoff = df["date_received"].iloc[train_end]
    oot_cutoff = df["date_received"].iloc[test_end]

    # Separar por datas, sem dividir o mesmo dia
    # Selecione todas as reclamações cuja data seja anterior a 29/04/2015.
    # Portanto temos 01/12/2011 <-> 29/04/2015
    train = df[df["date_received"] < train_cutoff].copy()

    # Valores 29/04/2015 <-> 18/10/2015
    test = df[
        (df["date_received"] >= train_cutoff)
        & (df["date_received"] < oot_cutoff)
    ].copy()

    # Valores 19/10/2015 <-> 25/04/2016
    oot = df[df["date_received"] >= oot_cutoff].copy()

    return train, test, oot

# %%

def split_features_target(df):
    target = "late_response"

    X = df.drop(
        columns = [target, "date_received"]
    ).copy()

    y = df[target].copy()

    return X, y

# %%

def validate_split(X, y, name):
    print(f"\n{name}")
    print("Dimensões de X:", X.shape)
    print("Dimensões de y:", y.shape)
    print("Valores ausentes em y:", y.isna().sum())
    print("Distribuição do target:")
    print(y.value_counts(normalize = True))

# %%

if __name__ == "__main__":
    df = load_data()
    df = validate_dates(df)

    train, test, oot = split_data(df)

    X_train, y_train = split_features_target(train)
    X_test, y_test = split_features_target(test)
    X_oot, y_oot = split_features_target(oot)

    validate_split(X_train, y_train, "Treino")
    validate_split(X_test, y_test, "Teste")
    validate_split(X_oot, y_oot, "OOT")

    print("\nX_train:", X_train.shape)
    print("y_train:", y_train.shape)

    print("\nX_test:", X_test.shape)
    print("y_test:", y_test.shape)

    print("\nX_oot:", X_oot.shape)
    print("y_oot:", y_oot.shape)

    for name, data in [
        ("Treino", train),
        ("Teste", test),
        ("OOT", oot)
    ]:
        print(f"\n{name}: {data.shape}")
        print("Início:", data["date_received"].min())
        print("Fim:", data["date_received"].max())
        print("Atrasos:", data["late_response"].sum())

# %%

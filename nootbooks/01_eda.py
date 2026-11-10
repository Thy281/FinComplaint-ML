# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

DATA_PATH = Path("../data/raw/consumer_complaints.csv")

df = pd.read_csv(
    DATA_PATH,
    low_memory = False,
    parse_dates = ["date_received", "date_sent_to_company"]
)
df.head()
# %%
# Dimension & Duplicated

print("\nLinhas e colunas:", df.shape)

def information():
    df.info()
    df.describe(include = "all").T
information()

print(
    "\nIDs duplicados:",
    df["complaint_id"].duplicated().sum()
)
# %%
# Missing Values

missing = (
    df.isna().mean().mul(100) # % Value
    .sort_values(ascending = False)
)
missing.to_frame(name = "missing_pct")

plt.figure(figsize = (11, 5))
missing.plot(kind = "bar")

plt.title("Percentual de valores ausentes")
plt.ylabel("Ausentes (%)")
plt.xlabel("Colunas")
plt.xticks(rotation=70, ha="right")
plt.tight_layout()
plt.show()

# %%
# Variavel de Anaslise

df["late_response"] = (
    df["timely_response"]
    .map({"Yes": 0, "No": 1})
)

print(df["late_response"].value_counts())
print(
    df["late_response"]
    .value_counts(normalize = True)
    .mul(100) # % value
    .round(2) # arredondamento em 2 casas
)

# %%
# Produtos mais reclamados

top_products = (
    df["product"]
    .value_counts()
    .head(10)
)

plt.figure(figsize = (11, 5))

sns.barplot(
    x = top_products.values,
    y = top_products.index
)

plt.title("Produtos com mais reclamações")
plt.xlabel("Quantidade de reclamações")
plt.ylabel("Produto")
plt.tight_layout()
plt.show()

# %%
# Taxa de atraso por produto

product_analysis = (
    df.groupby("product")
    .agg(
        total = ("complaint_id", "count"),
        late_count = ("late_response", "sum"),
        late_rate = ("late_response", "mean")
    )
)

product_analysis["late_rate_pct"] = (
    product_analysis["late_rate"] * 100
)

product_analysis = (
    product_analysis
    .query("total >= 1000")
    .sort_values("late_rate_pct", ascending = False)
)
product_analysis

sns.barplot(
    data = product_analysis.reset_index(),
    x = "late_rate_pct",
    y = "product"
)

plt.title("Taxa de atraso por produto")
plt.xlabel("Reclamações atrasadas (%)")
plt.ylabel("Produto")
plt.tight_layout()
plt.show()

# %%
# Evolucao Temporal

monthly = (
    df.set_index("date_received")
    .resample("MS")
    .agg(
        complaints = ("complaint_id", "count"),
        late_rate = ("late_response", "mean")
    )
)

monthly["late_rate_pct"] = monthly["late_rate"] * 100

monthly["complaints"].plot(
    figsize = (12, 4),
    title = "Reclamações recebidas por mês"
)
plt.xlabel("Anos")
plt.ylabel("Quantidade de reclamações")
plt.show()

monthly["late_rate_pct"].plot(
    figsize = (12, 4),
    title = "Taxa mensal de atraso (%)"
)
plt.xlabel("Anos")
plt.ylabel("Taxa de atraso (%)")
plt.show()

# %%
# Empresa com maior taxa de atraso

company_analysis = (
    df.groupby("company")
    .agg(
        total = ("complaint_id", "count"),
        late_rate = ("late_response", "mean")
    )
    .query("total >= 500")
    .sort_values("late_rate", ascending = False)
)

company_analysis["late_rate_pct"] = (
    company_analysis["late_rate"] * 100
)
company_analysis.head(15)

# %%
# Possivel vazamento de dados

post_event_columns = [
    "company_response_to_consumer",
    "company_public_response",
    "consumer_disputed?",
    "date_sent_to_company"
]

df[post_event_columns]
df.head(10)


# %%

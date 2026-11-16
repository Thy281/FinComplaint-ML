# %%

from pathlib import Path
import sqlite3

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATABASE = ROOT / "mlflow.db"
OUTPUT = ROOT / "reports" / "figures"
MODELS = (
    ("LogisticRegression", "Regressão\nlogística"),
    ("RandomForest", "Random\nForest"),
    ("XGBoost", "XGBoost"),
    ("BernoulliNB", "Bernoulli\nNB"),
)
SAMPLING = ("original", "over", "under")
COLORS = ("#2374ab", "#e18b3a", "#67a773")


def metric(connection, run_name, metric_name):
    row = connection.execute(
        """
        SELECT m.value
        FROM runs AS r
        JOIN experiments AS e ON e.experiment_id = r.experiment_id
        JOIN latest_metrics AS m ON m.run_uuid = r.run_uuid
        WHERE e.name = ? AND r.name = ? AND r.status = 'FINISHED'
          AND m.key = ?
        ORDER BY r.start_time DESC, r.run_uuid DESC
        LIMIT 1
        """,
        ("model-received", run_name, metric_name),
    ).fetchone()
    if row is None:
        raise ValueError(f"Métrica ausente: {run_name} / {metric_name}")
    return row[0]


def plot_training(connection):
    positions = np.arange(len(MODELS))
    width = 0.23
    fig, ax = plt.subplots(figsize=(10, 5.2))
    for index, sampling in enumerate(SAMPLING):
        values = [
            metric(connection, f"{model}_{sampling}", "cv_average_precision")
            for model, _ in MODELS
        ]
        bars = ax.bar(
            positions + (index - 1) * width,
            values,
            width,
            label={"original": "Original", "over": "Over", "under": "Under"}[sampling],
            color=COLORS[index],
        )
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax.set_xticks(positions, [label for _, label in MODELS])
    ax.set_ylim(0, 0.16)
    ax.set_ylabel("Average Precision (AP)")
    ax.set_title("Validação temporal dos treinamentos", loc="left", fontweight="bold")
    ax.legend(ncol=3, loc="upper left")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUTPUT / "training_cv_ap.png", dpi=160)
    plt.close(fig)


def plot_evaluation(connection):
    labels = ["BernoulliNB\nTeste", "BernoulliNB\nOOT", "XGBoost + over\nTeste"]
    values = [
        metric(connection, "BernoulliNB_original", "test_average_precision"),
        metric(connection, "BernoulliNB_original", "oot_average_precision"),
        metric(connection, "XGBoost_over", "test_average_precision"),
    ]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars = ax.bar(labels, values, color=("#2374ab", "#67a773", "#e18b3a"), width=0.55)
    ax.bar_label(bars, fmt="%.3f", padding=5)
    ax.set_ylim(0, 0.46)
    ax.set_ylabel("Average Precision (AP)")
    ax.set_title("Avaliação fora do treino", loc="left", fontweight="bold")
    ax.text(0.99, 0.96, "XGBoost: OOT não avaliado", transform=ax.transAxes,
            ha="right", va="top", fontsize=10, color="#555555")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUTPUT / "test_oot_ap.png", dpi=160)
    plt.close(fig)


def plot_ranking(connection):
    cases = (
        ("BernoulliNB_original", "test", "BernoulliNB\nTeste"),
        ("XGBoost_over", "test", "XGBoost + over\nTeste"),
        ("BernoulliNB_original", "oot", "BernoulliNB\nOOT"),
    )
    measures = (
        ("precision_at_5_percent", "Precision@5%", 100, "%"),
        ("recall_at_5_percent", "Recall@5%", 100, "%"),
        ("lift_at_5_percent", "Lift@5%", 1, "×"),
    )
    colors = ("#2374ab", "#e18b3a", "#67a773")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))

    for ax, (key, title, scale, unit) in zip(axes, measures):
        values = [
            metric(connection, run_name, f"{split}_{key}") * scale
            for run_name, split, _ in cases
        ]
        bars = ax.bar(
            [label for _, _, label in cases], values, color=colors, width=0.62
        )
        ax.bar_label(
            bars,
            labels=[f"{value:.1f}{unit}" for value in values],
            padding=4,
            fontsize=9,
        )
        ax.set_ylim(0, max(values) * 1.2)
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)

    fig.suptitle("Ranking dos 5% com maior score", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT / "test_oot_ranking_at_5_percent.png", dpi=160)
    plt.close(fig)


def main():
    if not DATABASE.is_file():
        raise FileNotFoundError(f"Banco MLflow não encontrado: {DATABASE}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DATABASE.as_uri() + "?mode=ro", uri=True) as connection:
        plot_training(connection)
        plot_evaluation(connection)
        plot_ranking(connection)
    print(f"Gráficos salvos em {OUTPUT}")


if __name__ == "__main__":
    main()

# %%

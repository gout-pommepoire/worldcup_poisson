"""Visualisations : heatmap des scores, barres des probabilités."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns


def plot_score_heatmap(result: dict, ax=None) -> plt.Axes:
    """Heatmap de la matrice de scores pour un match donné."""
    mat = result["score_matrix"]
    home, away = result["home_team"], result["away_team"]
    n = min(7, mat.shape[0])

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        mat[:n, :n] * 100,
        annot=True, fmt=".1f", cmap="YlOrRd",
        xticklabels=range(n), yticklabels=range(n),
        ax=ax, linewidths=0.5, cbar_kws={"label": "Probabilité (%)"},
    )
    ax.set_xlabel(f"Buts {away}", fontsize=12)
    ax.set_ylabel(f"Buts {home}", fontsize=12)
    ax.set_title(f"{home} vs {away}\nDistribution des scores (%)", fontsize=13, fontweight="bold")
    return ax


def plot_outcome_bars(result: dict, ax=None) -> plt.Axes:
    """Barres des probabilités victoire/nul/défaite."""
    home, away = result["home_team"], result["away_team"]

    labels = [f"Victoire\n{home}", "Match nul", f"Victoire\n{away}"]
    probs  = [result["prob_home_win"], result["prob_draw"], result["prob_away_win"]]
    colors = ["#2196F3", "#9E9E9E", "#F44336"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))

    bars = ax.bar(labels, [p * 100 for p in probs], color=colors, edgecolor="white", linewidth=1.5)
    for bar, prob in zip(bars, probs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{prob:.1%}",
            ha="center", va="bottom", fontweight="bold", fontsize=11,
        )
    ax.set_ylim(0, max(probs) * 140)
    ax.set_ylabel("Probabilité (%)")
    ax.set_title(f"{home} vs {away}", fontsize=12, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    return ax


def plot_match(result: dict, save_path: str | None = None):
    """Affiche heatmap + barres côte à côte."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plt.suptitle(
        f"🏆 Coupe du Monde 2026 — Prédiction Dixon-Coles\n"
        f"{result['home_team']} vs {result['away_team']}",
        fontsize=14, fontweight="bold", y=1.02,
    )
    plot_score_heatmap(result, ax=ax1)
    plot_outcome_bars(result, ax=ax2)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"📊 Graphique sauvegardé : {save_path}")
    plt.show()


def plot_strengths(df_strengths, top_n: int = 20, save_path: str | None = None):
    """Scatter attaque vs défense pour les top N équipes."""
    df = df_strengths.head(top_n)

    fig, ax = plt.subplots(figsize=(9, 6))
    scatter = ax.scatter(
        df["defense"], df["attack"],
        c=df["overall"], cmap="RdYlGn", s=120, edgecolors="black", linewidths=0.5,
    )
    for _, row in df.iterrows():
        ax.annotate(row["team"], (row["defense"], row["attack"]),
                    fontsize=8, ha="left", va="bottom",
                    xytext=(4, 4), textcoords="offset points")

    plt.colorbar(scatter, ax=ax, label="Force globale (att/déf)")
    ax.set_xlabel("Force défensive (↑ = meilleure défense)")
    ax.set_ylabel("Force offensive (↑ = meilleure attaque)")
    ax.set_title(f"Forces des équipes — Top {top_n}", fontsize=13, fontweight="bold")
    ax.invert_xaxis()   # faible déf à droite = mauvais
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

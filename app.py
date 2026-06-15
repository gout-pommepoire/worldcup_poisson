"""
Interface Streamlit — Modèle Dixon-Coles Coupe du Monde 2026
Lance avec : streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from itertools import product

from data_loader import load_all
from model import DixonColesModel

# ---------------------------------------------------------------------------
# Config page
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="⚽ Prédicteur CdM 2026",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ Prédicteur Coupe du Monde 2026")
st.caption("Modèle de Poisson Dixon-Coles • Pondération temporelle • 5 350 matchs historiques")

# ---------------------------------------------------------------------------
# Chargement & entraînement (caché)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Entraînement du modèle…")
def get_model():
    df = load_all()
    model = DixonColesModel()
    model.fit(df)
    return model, df

model, df = get_model()
teams = sorted(model.att.keys())

# ---------------------------------------------------------------------------
# Sidebar — sélection des équipes
# ---------------------------------------------------------------------------

st.sidebar.header("🎯 Choisir le match")

default_home = teams.index("France") if "France" in teams else 0
default_away = teams.index("Brazil") if "Brazil" in teams else 1

home = st.sidebar.selectbox("Équipe domicile", teams, index=default_home)
away = st.sidebar.selectbox("Équipe extérieure", teams, index=default_away)

if home == away:
    st.sidebar.warning("Choisir deux équipes différentes.")
    st.stop()

st.sidebar.markdown("---")
max_goals_display = st.sidebar.slider("Buts max affichés (heatmap)", 4, 8, 6)
top_n = st.sidebar.slider("Top N scores", 5, 15, 10)

# ---------------------------------------------------------------------------
# Prédiction
# ---------------------------------------------------------------------------

result = model.predict(home, away)
mat = result["score_matrix"]

# ---------------------------------------------------------------------------
# Ligne du haut — métriques
# ---------------------------------------------------------------------------

st.subheader(f"{home}  vs  {away}")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(f"Victoire {home}", f"{result['prob_home_win']:.1%}")
c2.metric("Match nul", f"{result['prob_draw']:.1%}")
c3.metric(f"Victoire {away}", f"{result['prob_away_win']:.1%}")
c4.metric("Score le + probable", result["most_likely_score"])
c5.metric("Buts attendus", f"{result['expected_home_goals']} – {result['expected_away_goals']}")

st.markdown("---")

# ---------------------------------------------------------------------------
# Ligne du bas — heatmap + top scores + forces
# ---------------------------------------------------------------------------

col_heat, col_scores, col_strength = st.columns([2, 1, 1])

# Heatmap
with col_heat:
    st.markdown("#### Distribution des scores (%)")
    n = max_goals_display + 1
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        mat[:n, :n] * 100,
        annot=True, fmt=".1f", cmap="YlOrRd",
        xticklabels=range(n), yticklabels=range(n),
        ax=ax, linewidths=0.4,
        cbar_kws={"label": "%"},
    )
    ax.set_xlabel(f"Buts {away}", fontsize=11)
    ax.set_ylabel(f"Buts {home}", fontsize=11)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# Top scores
with col_scores:
    st.markdown("#### Top scores")
    top_df = model.top_scores(home, away, n=top_n)
    st.dataframe(top_df, hide_index=True, use_container_width=True)

# Forces relatives
with col_strength:
    st.markdown("#### Forces comparées")
    strengths = model.team_strengths()
    h_row = strengths[strengths["team"] == home].iloc[0]
    a_row = strengths[strengths["team"] == away].iloc[0]

    comp_df = pd.DataFrame({
        "": ["Attaque", "Défense", "Force globale"],
        home: [
            f"{h_row['attack']:.2f}",
            f"{h_row['defense']:.2f}",
            f"{h_row['overall']:.2f}",
        ],
        away: [
            f"{a_row['attack']:.2f}",
            f"{a_row['defense']:.2f}",
            f"{a_row['overall']:.2f}",
        ],
    })
    st.dataframe(comp_df, hide_index=True, use_container_width=True)

    rank_h = strengths[strengths["team"] == home].index[0] + 1
    rank_a = strengths[strengths["team"] == away].index[0] + 1
    st.caption(f"Classement modèle : {home} #{rank_h} · {away} #{rank_a}")

# ---------------------------------------------------------------------------
# Classement général (expandable)
# ---------------------------------------------------------------------------

st.markdown("---")
with st.expander("🏆 Classement complet des équipes par force"):
    st.dataframe(
        strengths.assign(
            rank=range(1, len(strengths)+1)
        )[["rank", "team", "attack", "defense", "overall"]]
        .rename(columns={"rank": "#", "team": "Équipe", "attack": "Attaque",
                         "defense": "Défense", "overall": "Force globale"})
        .style.format({"Attaque": "{:.3f}", "Défense": "{:.3f}", "Force globale": "{:.3f}"}),
        hide_index=True,
        use_container_width=True,
        height=400,
    )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.caption(
    f"Données : {len(df)} matchs ({df['date'].dt.year.min()}–{df['date'].dt.year.max()}) · "
    "Source : martj42/international_results · "
    "Modèle : Dixon-Coles (1997) avec pondération temporelle (demi-vie 10 ans)"
)

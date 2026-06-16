"""
Interface Streamlit — Modèle Dixon-Coles Coupe du Monde 2026
Lance avec : streamlit run app.py
"""

import os
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.graph_objects as go
from itertools import product

from data_loader import load_all
from model import DixonColesModel
from tournament import build_groups, run_monte_carlo

# ---------------------------------------------------------------------------
# Config page
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="⚽ Prédicteur CdM 2026",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ Prédicteur Coupe du Monde 2026")
st.caption("Modèle de Poisson Dixon-Coles • Pondération temporelle")

csv_mtime_top = datetime.fromtimestamp(os.path.getmtime("results.csv")).strftime("%d/%m/%Y à %H:%M")
st.info(f"🔄 **Données mises à jour le {csv_mtime_top}**", icon="📅")

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

strengths = model.team_strengths()

tab_match, tab_tournoi = st.tabs(["🎯 Match", "🏆 Tournoi"])

# =============================================================================
# ONGLET MATCH
# =============================================================================

with tab_match:
    result = model.predict(home, away)
    mat = result["score_matrix"]

    st.subheader(f"{home}  vs  {away}")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(f"Victoire {home}", f"{result['prob_home_win']:.1%}")
    c2.metric("Match nul", f"{result['prob_draw']:.1%}")
    c3.metric(f"Victoire {away}", f"{result['prob_away_win']:.1%}")
    c4.metric("Score le + probable", result["most_likely_score"])
    c5.metric("Buts attendus", f"{result['expected_home_goals']} – {result['expected_away_goals']}")

    st.markdown("---")

    col_heat, col_scores, col_strength = st.columns([2, 1, 1])

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

    with col_scores:
        st.markdown("#### Top scores")
        top_df = model.top_scores(home, away, n=top_n)
        st.dataframe(top_df, hide_index=True, use_container_width=True)

    with col_strength:
        st.markdown("#### Forces comparées")
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

# =============================================================================
# ONGLET TOURNOI
# =============================================================================

with tab_tournoi:

    # --- Top 10 favoris (force brute) ---
    st.markdown("#### 🌟 Top 10 des favoris (force du modèle)")

    top10 = strengths.head(10).iloc[::-1]
    max_val = strengths["overall"].max()

    colors = ["#26215C" if i == 9 else "#3C3489" if i >= 7 else "#7F77DD" if i >= 4 else "#AFA9EC"
              for i in range(len(top10))]

    fig_fav = go.Figure(go.Bar(
        x=top10["overall"],
        y=top10["team"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:.2f}" for v in top10["overall"]],
        textposition="outside",
        hovertemplate="%{y} : %{x:.3f}<extra></extra>",
    ))
    fig_fav.update_layout(
        height=380,
        margin=dict(l=10, r=40, t=10, b=10),
        xaxis=dict(title="Force globale (attaque / défense)", range=[0, max_val * 1.15]),
        yaxis=dict(title=""),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_fav, use_container_width=True)

    st.markdown("---")

    # --- Simulation Monte Carlo du tournoi complet ---
    st.markdown("#### 🎲 Simulation Monte Carlo du tournoi (10 000 simulations)")
    st.caption(
        "Phase de groupes (12×4) → 32èmes → 16èmes → 1/4 → 1/2 → finale, "
        "simulés via le modèle Dixon-Coles à chaque match."
    )

    n_sims = st.select_slider("Nombre de simulations", options=[1000, 2000, 5000, 10000], value=10000)

    @st.cache_resource(show_spinner=f"Simulation en cours… (~{n_sims//150} secondes)")
    def get_tournament_results(_model, n_sims: int):
        groups = build_groups()
        return run_monte_carlo(_model, groups, n_sims=n_sims, seed=42), groups

    tourney_results, groups = get_tournament_results(model, n_sims)

    top_tourney = tourney_results.head(12).iloc[::-1]
    fig_tourney = go.Figure(go.Bar(
        x=top_tourney["prob_winner"] * 100,
        y=top_tourney["team"],
        orientation="h",
        marker_color=["#26215C" if i == 11 else "#3C3489" if i >= 9 else "#7F77DD" if i >= 5 else "#AFA9EC"
                      for i in range(len(top_tourney))],
        text=[f"{v:.1%}" for v in top_tourney["prob_winner"]],
        textposition="outside",
        hovertemplate="%{y} : %{x:.1f}%<extra></extra>",
    ))
    fig_tourney.update_layout(
        height=420,
        margin=dict(l=10, r=40, t=10, b=10),
        xaxis=dict(title="Probabilité de gagner le tournoi (%)"),
        yaxis=dict(title=""),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.markdown("**Probabilité de remporter la Coupe du Monde**")
    st.plotly_chart(fig_tourney, use_container_width=True)

    with st.expander("📊 Détail par étape (vainqueur, finale, demies, quarts, 16èmes)"):
        st.dataframe(
            tourney_results.head(30).rename(columns={
                "team": "Équipe",
                "prob_winner": "Vainqueur",
                "prob_finalist": "Finale",
                "prob_semifinalist": "1/2 finale",
                "prob_quarterfinalist": "1/4 finale",
                "prob_round16": "16èmes",
            }).style.format({
                "Vainqueur": "{:.1%}", "Finale": "{:.1%}", "1/2 finale": "{:.1%}",
                "1/4 finale": "{:.1%}", "16èmes": "{:.1%}",
            }),
            hide_index=True,
            use_container_width=True,
            height=400,
        )

    with st.expander("🗂️ Composition des 12 groupes"):
        n_cols = 3
        group_items = list(groups.items())
        for row_start in range(0, len(group_items), n_cols):
            cols = st.columns(n_cols)
            for col, (gname, gteams) in zip(cols, group_items[row_start:row_start + n_cols]):
                col.markdown(f"**{gname}**")
                for t in gteams:
                    col.write(f"• {t}")

    st.markdown("---")

    # --- Classement général (expandable) ---
    with st.expander("📋 Classement complet des équipes par force"):
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
last_match_date = df["date"].max().strftime("%d/%m/%Y")
csv_mtime = datetime.fromtimestamp(os.path.getmtime("results.csv")).strftime("%d/%m/%Y %H:%M")

st.caption(
    f"📅 Dernier match dans les données : **{last_match_date}** · "
    f"🔄 Fichier results.csv mis à jour le : **{csv_mtime}** · "
    f"📊 {len(df)} matchs ({df['date'].dt.year.min()}–{df['date'].dt.year.max()})"
)
st.caption(
    "Source : martj42/international_results · "
    "Modèle : Dixon-Coles (1997) avec pondération temporelle (demi-vie 10 ans)"
)

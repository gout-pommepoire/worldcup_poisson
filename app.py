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
from backtest import summary_stats
from tournament import (
    build_groups, load_wc2026_fixtures, group_standings,
    qualification_probabilities, build_bracket_labels,
    predict_round32_qualifiers, most_likely_winner,
)


def render_bracket_tree(rounds: list[list[tuple[str, str]]], champion: str) -> str:
    """
    Tableau à élimination directe complet en arbre, à la manière des tableaux
    CdM classiques (32èmes → finale), avec lignes de connexion calculées en
    pixels (et non en CSS approximatif) pour un alignement parfait.

    rounds : liste de tours, chacun étant une liste de matchs (team_a, team_b),
             du premier tour (32èmes, 16 matchs) à la finale (1 match).
    """
    U = 50            # unité verticale = hauteur de slot d'un match du 1er tour
    BOX_W = 150
    BOX_H = 40
    COL_GAP = 56
    n_first_round = len(rounds[0]) * 2   # nb d'équipes au 1er tour

    def center_y(k: int, i: int) -> float:
        """Centre vertical (en U) du match i du tour k (k=0 → 1er tour)."""
        return (i + 0.5) * (2 ** k) * U

    total_height = n_first_round * U
    total_width = (len(rounds) + 1) * (BOX_W + COL_GAP)

    elems = []

    def add_box(x, y, label_a, label_b, gold=False):
        bg = "linear-gradient(135deg,#FFD700,#B8860B)" if gold else "linear-gradient(135deg,#1a1a2e,#26215C)"
        color = "#1a1a2e" if gold else "#fff"
        elems.append(f'''
        <div style="position:absolute; left:{x}px; top:{y - BOX_H/2}px; width:{BOX_W}px; height:{BOX_H}px;
                    background:{bg}; border-radius:7px; color:{color}; font-size:12.5px; font-weight:600;
                    display:flex; flex-direction:column; justify-content:center; padding:0 8px;
                    box-shadow:0 2px 5px rgba(0,0,0,0.3); overflow:hidden; white-space:nowrap;
                    text-overflow:ellipsis; line-height:1.5;">
            <div>{label_a}</div>
            <div style="opacity:0.55; font-size:9px; font-weight:400;">vs</div>
            <div>{label_b}</div>
        </div>''')

    def add_hline(x, y, w):
        elems.append(f'<div style="position:absolute; left:{x}px; top:{y-1}px; width:{w}px; height:2px; background:#9994d9;"></div>')

    def add_vline(x, y0, y1):
        elems.append(f'<div style="position:absolute; left:{x-1}px; top:{min(y0,y1)}px; width:2px; height:{abs(y1-y0)}px; background:#9994d9;"></div>')

    for k, matches in enumerate(rounds):
        x = k * (BOX_W + COL_GAP)
        for i, (a, b) in enumerate(matches):
            y = center_y(k, i)
            add_box(x, y, a, b)
        if k < len(rounds) - 1:
            for p in range(len(matches) // 2):
                y0, y1 = center_y(k, 2*p), center_y(k, 2*p + 1)
                mid_x = x + BOX_W + COL_GAP / 2
                add_hline(x + BOX_W, y0, COL_GAP / 2)
                add_hline(x + BOX_W, y1, COL_GAP / 2)
                add_vline(mid_x, y0, y1)
                add_hline(mid_x, center_y(k + 1, p), COL_GAP / 2)

    # Champion
    last_k = len(rounds) - 1
    x_final = last_k * (BOX_W + COL_GAP)
    y_final = center_y(last_k, 0)
    x_champ = x_final + BOX_W + COL_GAP
    add_hline(x_final + BOX_W, y_final, COL_GAP)
    elems.append(f'''
    <div style="position:absolute; left:{x_champ}px; top:{y_final - BOX_H/2 - 4}px; width:{BOX_W}px; height:{BOX_H+8}px;
                background:linear-gradient(135deg,#FFD700,#B8860B); border-radius:8px; color:#1a1a2e;
                font-size:14px; font-weight:800; display:flex; align-items:center; justify-content:center;
                box-shadow:0 3px 8px rgba(0,0,0,0.35); text-align:center; padding:0 6px;">
        🏆 {champion}
    </div>''')

    inner_html = "".join(elems)
    return f'''
    <div style="overflow-x:auto; width:100%; padding-bottom:10px;">
      <div style="position:relative; width:{total_width}px; height:{total_height}px; min-width:{total_width}px;">
        {inner_html}
      </div>
    </div>
    '''

HOST_NATIONS = {"USA", "Mexico", "Canada"}

WC2026_TEAMS = sorted([
    'Algeria', 'Argentina', 'Australia', 'Austria', 'Belgium',
    'Bosnia and Herzegovina', 'Brazil', 'Canada', 'Cape Verde', 'Colombia',
    'Croatia', 'Curaçao', 'Czech Republic', 'DR Congo', 'Ecuador', 'Egypt',
    'England', 'France', 'Germany', 'Ghana', 'Haiti', 'Iran', 'Iraq',
    'Ivory Coast', 'Japan', 'Jordan', 'Mexico', 'Morocco', 'Netherlands',
    'New Zealand', 'Norway', 'Panama', 'Paraguay', 'Portugal', 'Qatar',
    'Saudi Arabia', 'Scotland', 'Senegal', 'South Africa', 'South Korea',
    'Spain', 'Sweden', 'Switzerland', 'Tunisia', 'Turkey', 'USA', 'Uruguay',
    'Uzbekistan',
])

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
wc_teams = [t for t in WC2026_TEAMS if t in model.att]

# ---------------------------------------------------------------------------
# Sidebar — sélection des équipes
# ---------------------------------------------------------------------------

st.sidebar.header("🎯 Choisir le match")

default_home = wc_teams.index("France") if "France" in wc_teams else 0
default_away = wc_teams.index("Brazil") if "Brazil" in wc_teams else 1

home = st.sidebar.selectbox("Équipe domicile", wc_teams, index=default_home)
away = st.sidebar.selectbox("Équipe extérieure", wc_teams, index=default_away)

if home == away:
    st.sidebar.warning("Choisir deux équipes différentes.")
    st.stop()

st.sidebar.markdown("---")
max_goals_display = st.sidebar.slider("Buts max affichés (heatmap)", 4, 8, 6)
top_n = st.sidebar.slider("Top N scores", 5, 15, 10)

strengths = model.team_strengths()

tab_match, tab_groupes, tab_bilan = st.tabs(["🎯 Match", "🏟️ Groupes & Tableau", "📈 Bilan"])

# =============================================================================
# ONGLET MATCH
# =============================================================================

with tab_match:
    is_neutral = home not in HOST_NATIONS
    result = model.predict(home, away, neutral=is_neutral)
    mat = result["score_matrix"]

    st.subheader(f"{home}  vs  {away}")
    if not is_neutral:
        st.caption(f"🏟️ {home} joue à domicile en tant que pays hôte — avantage du terrain appliqué.")
    else:
        st.caption("🌍 Match joué en terrain neutre — pas d'avantage du terrain appliqué.")

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
        top_df = model.top_scores(home, away, n=top_n, neutral=is_neutral)
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
# ONGLET GROUPES & TABLEAU
# =============================================================================

with tab_groupes:

    @st.cache_resource(show_spinner="Calcul des classements et probabilités de qualification…")
    def get_group_data(_model):
        groups = build_groups()
        fixtures = load_wc2026_fixtures()
        qualif = qualification_probabilities(_model, groups, fixtures, n_sims=3000, seed=42)
        standings = {g: group_standings(teams, fixtures) for g, teams in groups.items()}
        matches_played = {
            g: int(fixtures[
                fixtures["home_team"].isin(teams) & fixtures["away_team"].isin(teams) & fixtures["played"]
            ].shape[0])
            for g, teams in groups.items()
        }
        return groups, fixtures, qualif, standings, matches_played

    groups, fixtures, qualif_df, standings_map, matches_played = get_group_data(model)

    st.markdown("#### 🗂️ Classement des groupes")
    st.caption(
        "Classement calculé sur les matchs déjà joués · probabilité de qualification "
        "(1er/2e + 8 meilleurs 3èmes) estimée par simulation Monte Carlo des matchs restants."
    )

    group_choice = st.selectbox("Groupe", list(groups.keys()))

    table = standings_map[group_choice].merge(
        qualif_df[qualif_df["group"] == group_choice][["team", "prob_qualif"]],
        on="team",
    )
    table = table.assign(rank=range(1, len(table) + 1))[
        ["rank", "team", "j", "v", "n", "d", "gf", "ga", "gd", "pts", "prob_qualif"]
    ].rename(columns={
        "rank": "#", "team": "Équipe", "j": "J", "v": "V", "n": "N", "d": "D",
        "gf": "BP", "ga": "BC", "gd": "Diff", "pts": "Pts", "prob_qualif": "Proba qualif.",
    })

    st.dataframe(
        table.style.format({"Proba qualif.": "{:.0%}"}),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(f"{matches_played[group_choice]}/6 matchs joués dans ce groupe.")

    st.markdown("---")

    st.markdown("#### 🏆 Tableau à élimination directe — pronostic du modèle")
    st.caption(
        "Qualifiés des 32èmes projetés à partir du classement actuel de chaque groupe "
        "(1er/2e réel + meilleurs 3èmes selon leur proba de qualification), puis vainqueur "
        "le plus probable à chaque tour jusqu'à la finale. C'est un pronostic basé sur "
        "l'état actuel — pas une certitude, ça évoluera avec les résultats."
    )

    group_names = list(groups.keys())
    round32_labels = build_bracket_labels(group_names)
    round32_pred = predict_round32_qualifiers(round32_labels, standings_map, qualif_df)
    round32_winners = [most_likely_winner(model, a, b) for a, b in round32_pred]

    round16_matchups = [(round32_winners[2*i], round32_winners[2*i+1]) for i in range(8)]
    round16_winners = [most_likely_winner(model, a, b) for a, b in round16_matchups]

    quarts_matchups = [(round16_winners[2*i], round16_winners[2*i+1]) for i in range(4)]
    quarts_winners = [most_likely_winner(model, a, b) for a, b in quarts_matchups]

    demi_matchups = [(quarts_winners[2*i], quarts_winners[2*i+1]) for i in range(2)]
    demi_winners = [most_likely_winner(model, a, b) for a, b in demi_matchups]

    finale_matchup = (demi_winners[0], demi_winners[1])
    champion = most_likely_winner(model, *finale_matchup)

    st.caption("📱 Sur mobile, fais glisser le tableau horizontalement pour voir tous les tours.")
    st.markdown(
        render_bracket_tree(
            [round32_pred, round16_matchups, quarts_matchups, demi_matchups, [finale_matchup]],
            champion,
        ),
        unsafe_allow_html=True,
    )

# =============================================================================
# ONGLET BILAN — prédictions vs résultats réels
# =============================================================================

with tab_bilan:
    st.markdown("#### 📈 Bilan des pronostics — matchs CdM 2026 déjà joués")
    st.caption(
        "Pour chaque match déjà joué, le modèle est ré-entraîné **sans ce match** "
        "(leave-one-out) puis utilisé pour prédire — comparaison honnête, sans triche."
    )
    st.warning(
        "⚠️ **\"xG\" ici = buts attendus *pré-match* du modèle**, pas un xG calculé sur les tirs/occasions "
        "réels du match (pas de données de tirs disponibles). Un match dominé sans but marqué "
        "(ex : beaucoup de tirs, 0 but) sera donc vu comme une erreur du modèle, alors qu'un vrai xG "
        "post-match (basé sur les occasions créées) confirmerait souvent que le modèle avait juste sur le fond.",
        icon="⚠️",
    )

    BACKTEST_CSV = "backtest_results.csv"

    if not os.path.exists(BACKTEST_CSV):
        backtest_df = pd.DataFrame()
        st.info(
            "Bilan pas encore calculé. Lance `python precompute_backtest.py` en local, "
            "puis commit/push `backtest_results.csv`."
        )
    else:
        backtest_df = pd.read_csv(BACKTEST_CSV)
        bt_mtime = datetime.fromtimestamp(os.path.getmtime(BACKTEST_CSV)).strftime("%d/%m/%Y à %H:%M")
        st.caption(f"📌 Bilan précalculé le {bt_mtime} (calcul lourd fait en local, pas sur le serveur)")

    if backtest_df.empty:
        pass
    else:
        stats = summary_stats(backtest_df)

        c1, c2, c3, c4 = st.columns(4)
        n_with_real_xg = int(backtest_df["has_real_xg"].sum())
        c1.metric("Matchs analysés", stats["n_matches"])
        c2.metric("Score exact", f"{stats['pct_score_exact']:.0%}")
        c3.metric("Bon résultat (1N2)", f"{stats['pct_resultat_correct']:.0%}")
        c4.metric("xG cohérent", f"{stats['pct_xg_coherent']:.0%}",
                   help=f"{n_with_real_xg}/{stats['n_matches']} matchs avec un vrai xG relevé manuellement")

        st.markdown("---")

        display_df = backtest_df.copy()
        display_df["xG dom."] = display_df.apply(
            lambda r: f"{r['xG_reel_dom']:.2f}" if r["has_real_xg"] else f"{r['xG_predit_dom']:.2f}",
            axis=1,
        )
        display_df["xG ext."] = display_df.apply(
            lambda r: f"{r['xG_reel_ext']:.2f}" if r["has_real_xg"] else f"{r['xG_predit_ext']:.2f}",
            axis=1,
        )

        display_df = display_df[[
            "date", "home_team", "away_team", "score_reel", "score_predit",
            "xG dom.", "xG ext.", "erreur_xG",
            "resultat_predit", "resultat_reel", "verdict",
        ]].rename(columns={
            "date": "Date", "home_team": "Domicile", "away_team": "Extérieur",
            "score_reel": "Score réel", "score_predit": "Score prédit",
            "erreur_xG": "Erreur xG",
            "resultat_predit": "1N2 prédit", "resultat_reel": "1N2 réel",
            "verdict": "Verdict",
        })

        st.dataframe(
            display_df.style.format({"Erreur xG": "{:.2f}"}),
            hide_index=True,
            use_container_width=True,
            height=460,
        )

        st.caption(
            "🟢 Bon résultat : le modèle avait le bon vainqueur/nul, même si le score exact diffère · "
            "🟡 Résultat raté, xG cohérent : mauvais 1N2 mais le xG collait à la performance réelle "
            "(ex: domination sans concrétisation) · 🔴 Raté : ni le résultat ni le xG ne correspondaient."
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

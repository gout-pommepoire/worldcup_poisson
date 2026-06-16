"""
Simulation Monte Carlo du tournoi complet (format CdM 2026 : 48 équipes).

Format :
  - 12 groupes de 4 équipes (round-robin, 6 matchs/groupe)
  - Qualifiés : les 2 premiers de chaque groupe (24) + les 8 meilleurs 3èmes
  - 32 équipes en phase à élimination directe : 32èmes, 16èmes, 1/4, 1/2, finale

On utilise le modèle Dixon-Coles déjà entraîné pour tirer un score aléatoire
à chaque match simulé (via sa matrice de probabilités de score).
"""

import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict

from model import DixonColesModel
from data_loader import normalize_team


# ---------------------------------------------------------------------------
# Reconstruction des groupes depuis le calendrier 2026
# ---------------------------------------------------------------------------

def build_groups(csv_path: str = "results.csv") -> dict[str, list[str]]:
    df = pd.read_csv(csv_path, parse_dates=["date"])
    wc26 = df[(df["tournament"] == "FIFA World Cup") & (df["date"].dt.year == 2026)]

    G = nx.Graph()
    for _, row in wc26.iterrows():
        G.add_edge(normalize_team(row["home_team"]), normalize_team(row["away_team"]))

    components = sorted(nx.connected_components(G), key=lambda x: sorted(x)[0])
    return {
        f"Groupe {i+1}": sorted(group)
        for i, group in enumerate(components)
    }


# ---------------------------------------------------------------------------
# Tirage d'un score de match selon le modèle
# ---------------------------------------------------------------------------

_MATRIX_CACHE: dict[tuple[str, str], np.ndarray] = {}


def get_cached_matrix(model: DixonColesModel, team_a: str, team_b: str,
                       max_goals: int = 6) -> np.ndarray:
    """Les paramètres du modèle sont fixes pendant tout le Monte Carlo :
    une paire d'équipes a toujours la même matrice de score."""
    key = (team_a, team_b)
    if key not in _MATRIX_CACHE:
        _MATRIX_CACHE[key] = model.score_matrix(team_a, team_b, max_goals=max_goals)
    return _MATRIX_CACHE[key]


def draw_score(model: DixonColesModel, team_a: str, team_b: str,
                rng: np.random.Generator, max_goals: int = 6) -> tuple[int, int]:
    """Tire un score (buts_a, buts_b) selon la matrice de probabilités."""
    mat = get_cached_matrix(model, team_a, team_b, max_goals=max_goals)
    flat = mat.flatten()
    idx = rng.choice(len(flat), p=flat)
    return idx // mat.shape[1], idx % mat.shape[1]


def draw_knockout_winner(model: DixonColesModel, team_a: str, team_b: str,
                          rng: np.random.Generator) -> str:
    """
    Match à élimination directe : en cas de nul après 90 min,
    on retire un vainqueur via tirs au but (50/50 légèrement pondéré par la force).
    """
    ga, gb = draw_score(model, team_a, team_b, rng)
    if ga > gb:
        return team_a
    if gb > ga:
        return team_b
    # Nul → prolongation/TAB, légère pondération par force relative
    sa = model.att[team_a] / model.defe[team_a]
    sb = model.att[team_b] / model.defe[team_b]
    p_a = 0.5 + 0.05 * np.tanh(np.log(sa / sb))   # avantage marginal pour le plus fort
    return team_a if rng.random() < p_a else team_b


# ---------------------------------------------------------------------------
# Phase de groupes
# ---------------------------------------------------------------------------

def simulate_group(model: DixonColesModel, teams: list[str],
                    rng: np.random.Generator) -> pd.DataFrame:
    """Simule les 6 matchs round-robin d'un groupe, retourne le classement."""
    stats = {t: {"pts": 0, "gf": 0, "ga": 0, "gd": 0} for t in teams}

    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            a, b = teams[i], teams[j]
            ga, gb = draw_score(model, a, b, rng)
            stats[a]["gf"] += ga; stats[a]["ga"] += gb
            stats[b]["gf"] += gb; stats[b]["ga"] += ga
            if ga > gb:
                stats[a]["pts"] += 3
            elif gb > ga:
                stats[b]["pts"] += 3
            else:
                stats[a]["pts"] += 1; stats[b]["pts"] += 1

    for t in teams:
        stats[t]["gd"] = stats[t]["gf"] - stats[t]["ga"]

    df = pd.DataFrame.from_dict(stats, orient="index").reset_index()
    df = df.rename(columns={"index": "team"})
    # Tri : points, diff buts, buts marqués (départage simplifié, sans face-à-face)
    df = df.sort_values(["pts", "gd", "gf"], ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Phase à élimination directe
# ---------------------------------------------------------------------------

def build_bracket_32(qualified_groups: dict[str, pd.DataFrame],
                      best_thirds: list[str]) -> list[tuple[str, str]]:
    """
    Construit les 16 affiches des 32èmes de finale.
    Simplification : positions standard CdM (1er groupe A vs 2ème groupe B etc.)
    On utilise un tirage simplifié mais respectant la contrainte
    "pas deux équipes du même groupe avant les 1/4" autant que possible.
    """
    firsts  = {g: df.iloc[0]["team"] for g, df in qualified_groups.items()}
    seconds = {g: df.iloc[1]["team"] for g, df in qualified_groups.items()}

    group_names = list(qualified_groups.keys())  # Groupe 1..12
    pairs = []

    # 8 affiches 1er vs 2ème de groupes différents (rotation standard)
    for i in range(8):
        g_first  = group_names[i % 12]
        g_second = group_names[(i + 4) % 12]
        pairs.append((firsts[g_first], seconds[g_second]))

    # 4 meilleurs 3èmes affrontent les 4 vainqueurs de groupes restants
    remaining_firsts = [firsts[g] for g in group_names[8:12]]
    for i in range(4):
        pairs.append((remaining_firsts[i], best_thirds[i]))

    # 4 affiches restantes : seconds restants vs 4 autres meilleurs 3èmes
    remaining_seconds = [seconds[g] for g in group_names[4:8]]
    for i in range(4):
        pairs.append((remaining_seconds[i], best_thirds[4 + i] if i + 4 < len(best_thirds) else remaining_seconds[i]))

    return pairs[:16]


def simulate_knockout_round(model: DixonColesModel, matchups: list[tuple[str, str]],
                             rng: np.random.Generator) -> list[str]:
    return [draw_knockout_winner(model, a, b, rng) for a, b in matchups]


def pair_up(teams: list[str]) -> list[tuple[str, str]]:
    return [(teams[i], teams[i+1]) for i in range(0, len(teams), 2)]


# ---------------------------------------------------------------------------
# Simulation complète d'un tournoi
# ---------------------------------------------------------------------------

def simulate_tournament(model: DixonColesModel, groups: dict[str, list[str]],
                         rng: np.random.Generator) -> dict:
    """Retourne {'winner': ..., 'finalist': ..., 'semifinalists': [...], 'quarterfinalists':[...]}"""
    group_results = {g: simulate_group(model, teams, rng) for g, teams in groups.items()}

    thirds = []
    for g, df in group_results.items():
        row = df.iloc[2].to_dict()
        row["group"] = g
        thirds.append(row)
    thirds_df = pd.DataFrame(thirds).sort_values(["pts", "gd", "gf"], ascending=False)
    best_8_thirds = thirds_df.head(8)["team"].tolist()

    round32 = build_bracket_32(group_results, best_8_thirds)
    round16_teams = simulate_knockout_round(model, round32, rng)

    round16 = pair_up(round16_teams)
    quarter_teams = simulate_knockout_round(model, round16, rng)

    quarters = pair_up(quarter_teams)
    semi_teams = simulate_knockout_round(model, quarters, rng)

    semis = pair_up(semi_teams)
    final_teams = simulate_knockout_round(model, semis, rng)

    final = pair_up(final_teams)
    winner = simulate_knockout_round(model, final, rng)[0]

    return {
        "winner": winner,
        "finalists": final_teams,
        "semifinalists": semi_teams,
        "quarterfinalists": quarter_teams,
        "round16": round16_teams,
    }


# ---------------------------------------------------------------------------
# Monte Carlo — N simulations
# ---------------------------------------------------------------------------

def run_monte_carlo(model: DixonColesModel, groups: dict[str, list[str]],
                     n_sims: int = 10_000, seed: int = 42) -> pd.DataFrame:
    _MATRIX_CACHE.clear()
    rng = np.random.default_rng(seed)

    counters = {
        "winner": defaultdict(int),
        "finalist": defaultdict(int),
        "semifinalist": defaultdict(int),
        "quarterfinalist": defaultdict(int),
        "round16": defaultdict(int),
    }

    for _ in range(n_sims):
        result = simulate_tournament(model, groups, rng)
        counters["winner"][result["winner"]] += 1
        for t in result["finalists"]:
            counters["finalist"][t] += 1
        for t in result["semifinalists"]:
            counters["semifinalist"][t] += 1
        for t in result["quarterfinalists"]:
            counters["quarterfinalist"][t] += 1
        for t in result["round16"]:
            counters["round16"][t] += 1

    all_teams = sorted({t for g in groups.values() for t in g})
    rows = []
    for t in all_teams:
        rows.append({
            "team": t,
            "prob_winner": counters["winner"][t] / n_sims,
            "prob_finalist": counters["finalist"][t] / n_sims,
            "prob_semifinalist": counters["semifinalist"][t] / n_sims,
            "prob_quarterfinalist": counters["quarterfinalist"][t] / n_sims,
            "prob_round16": counters["round16"][t] / n_sims,
        })

    return pd.DataFrame(rows).sort_values("prob_winner", ascending=False).reset_index(drop=True)

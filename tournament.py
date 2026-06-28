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
from collections import defaultdict, Counter

from model import DixonColesModel
from data_loader import normalize_team


# ---------------------------------------------------------------------------
# Groupes officiels CdM 2026 (tirage au sort FIFA, lettres A à L)
# ---------------------------------------------------------------------------

GROUPS_2026: dict[str, list[str]] = {
    "Groupe A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "Groupe B": ["Canada", "Switzerland", "Bosnia and Herzegovina", "Qatar"],
    "Groupe C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "Groupe D": ["USA", "Paraguay", "Australia", "Turkey"],
    "Groupe E": ["Germany", "Ivory Coast", "Ecuador", "Curaçao"],
    "Groupe F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "Groupe G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "Groupe H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "Groupe I": ["France", "Senegal", "Iraq", "Norway"],
    "Groupe J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "Groupe K": ["Portugal", "DR Congo", "Uzbekistan", "Colombia"],
    "Groupe L": ["England", "Croatia", "Ghana", "Panama"],
}


def build_groups(csv_path: str = "results.csv") -> dict[str, list[str]]:
    """Groupes officiels du tirage au sort FIFA (fixes, ne dépendent pas du calendrier)."""
    return {g: sorted(teams) for g, teams in GROUPS_2026.items()}


# ---------------------------------------------------------------------------
# Structure officielle du tableau à élimination directe (FIFA, format 48 équipes)
# ---------------------------------------------------------------------------
# Les 16 affiches des 32èmes, dans l'ordre officiel des matchs 73 à 88.
# "1er"/"2e" réfère au classement du groupe ; "3e" est rempli par un des 8
# meilleurs 3èmes (assignation simplifiée par ordre de probabilité de
# qualification, la table officielle complète FIFA gérant ~495 combinaisons
# possibles selon quels groupes se qualifient comme 3èmes).
ROUND32_SLOTS: list[tuple[str, str]] = [
    ("2e Groupe A", "2e Groupe B"),    # match 73
    ("1er Groupe E", "3e"),            # match 74
    ("1er Groupe F", "2e Groupe C"),   # match 75
    ("1er Groupe C", "2e Groupe F"),   # match 76
    ("1er Groupe I", "3e"),            # match 77
    ("2e Groupe E", "2e Groupe I"),    # match 78
    ("1er Groupe A", "3e"),            # match 79
    ("1er Groupe L", "3e"),            # match 80
    ("1er Groupe D", "3e"),            # match 81
    ("1er Groupe G", "3e"),            # match 82
    ("2e Groupe K", "2e Groupe L"),    # match 83
    ("1er Groupe H", "2e Groupe J"),   # match 84
    ("1er Groupe B", "3e"),            # match 85
    ("1er Groupe J", "2e Groupe H"),   # match 86
    ("1er Groupe K", "3e"),            # match 87
    ("2e Groupe D", "2e Groupe G"),    # match 88
]

# Groupes (dans l'ordre officiel) auxquels les 8 meilleurs 3èmes sont assignés
# (slots "3e" des matchs 74, 77, 79, 80, 81, 82, 85, 87 ci-dessus)
THIRD_PLACE_SLOT_GROUPS = ["E", "I", "A", "L", "D", "G", "B", "K"]

# 16èmes (indices dans la liste des 16 vainqueurs des 32èmes, ordre matchs 73-88)
ROUND16_PAIRS = [(0, 2), (1, 4), (3, 5), (6, 7), (10, 11), (8, 9), (13, 15), (12, 14)]
# Quarts (indices dans la liste des 8 vainqueurs des 16èmes, ordre ci-dessus)
QUARTERS_PAIRS = [(0, 1), (4, 5), (2, 3), (6, 7)]
# Demies (indices dans la liste des 4 vainqueurs des quarts, ordre ci-dessus)
SEMIS_PAIRS = [(0, 1), (2, 3)]


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
    Construit les 16 affiches des 32èmes de finale selon la structure officielle
    FIFA (ROUND32_SLOTS) : 1er/2e de groupes spécifiques se croisent volontairement
    entre les deux moitiés du tableau, et 8 des 16 matchs opposent un 1er de groupe
    à un des 8 meilleurs 3èmes.

    qualified_groups : dict {"Groupe A": DataFrame trié, ...}
    best_thirds : les 8 équipes 3èmes qualifiées, dans l'ordre des groupes
                  THIRD_PLACE_SLOT_GROUPS (E, I, A, L, D, G, B, K)
    """
    def team(label: str) -> str:
        # label ex: "1er Groupe E" / "2e Groupe C"
        pos, group = label.split(" ", 1)
        idx = 0 if pos == "1er" else 1
        return qualified_groups[group].iloc[idx]["team"]

    third_by_group = dict(zip(THIRD_PLACE_SLOT_GROUPS, best_thirds))
    pairs = []
    third_iter = iter(THIRD_PLACE_SLOT_GROUPS)
    for a_label, b_label in ROUND32_SLOTS:
        a = team(a_label)
        b = third_by_group[next(third_iter)] if b_label == "3e" else team(b_label)
        pairs.append((a, b))

    return pairs


def simulate_knockout_round(model: DixonColesModel, matchups: list[tuple[str, str]],
                             rng: np.random.Generator) -> list[str]:
    return [draw_knockout_winner(model, a, b, rng) for a, b in matchups]


# ---------------------------------------------------------------------------
# Monte Carlo — N simulations
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Classement en cours + probas de qualification (état réel des groupes)
# ---------------------------------------------------------------------------

def load_wc2026_fixtures(csv_path: str = "results.csv") -> pd.DataFrame:
    """Calendrier complet des 72 matchs de phase de groupes (joués + à venir)."""
    df = pd.read_csv(csv_path, parse_dates=["date"])
    wc26 = df[(df["tournament"] == "FIFA World Cup") & (df["date"].dt.year == 2026)].copy()
    wc26["home_team"] = wc26["home_team"].map(normalize_team)
    wc26["away_team"] = wc26["away_team"].map(normalize_team)
    wc26["played"] = wc26["home_score"].notna() & (wc26["home_score"].astype(str) != "NA")
    wc26.loc[wc26["played"], "home_goals"] = wc26.loc[wc26["played"], "home_score"].astype(float).astype(int)
    wc26.loc[wc26["played"], "away_goals"] = wc26.loc[wc26["played"], "away_score"].astype(float).astype(int)
    return wc26[["date", "home_team", "away_team", "home_goals", "away_goals", "played"]]


def group_standings(teams: list[str], fixtures: pd.DataFrame) -> pd.DataFrame:
    """Classement réel actuel d'un groupe, calculé uniquement sur les matchs déjà joués."""
    teams_set = set(teams)
    played = fixtures[
        fixtures["played"]
        & fixtures["home_team"].isin(teams_set)
        & fixtures["away_team"].isin(teams_set)
    ]

    stats = {t: {"j": 0, "v": 0, "n": 0, "d": 0, "gf": 0, "ga": 0, "pts": 0} for t in teams}
    for _, row in played.iterrows():
        a, b = row["home_team"], row["away_team"]
        ga, gb = int(row["home_goals"]), int(row["away_goals"])
        stats[a]["j"] += 1; stats[b]["j"] += 1
        stats[a]["gf"] += ga; stats[a]["ga"] += gb
        stats[b]["gf"] += gb; stats[b]["ga"] += ga
        if ga > gb:
            stats[a]["v"] += 1; stats[a]["pts"] += 3; stats[b]["d"] += 1
        elif gb > ga:
            stats[b]["v"] += 1; stats[b]["pts"] += 3; stats[a]["d"] += 1
        else:
            stats[a]["n"] += 1; stats[b]["n"] += 1
            stats[a]["pts"] += 1; stats[b]["pts"] += 1

    df = pd.DataFrame.from_dict(stats, orient="index").reset_index().rename(columns={"index": "team"})
    df["gd"] = df["gf"] - df["ga"]
    return df.sort_values(["pts", "gd", "gf"], ascending=False).reset_index(drop=True)


def simulate_group_remaining(model: DixonColesModel, teams: list[str],
                              fixtures: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Classement d'un groupe pour UNE simulation : matchs déjà joués gardés tels quels,
    matchs restants tirés selon le modèle."""
    teams_set = set(teams)
    group_fixtures = fixtures[
        fixtures["home_team"].isin(teams_set) & fixtures["away_team"].isin(teams_set)
    ]

    stats = {t: {"pts": 0, "gf": 0, "ga": 0} for t in teams}

    for _, row in group_fixtures.iterrows():
        a, b = row["home_team"], row["away_team"]
        if row["played"]:
            ga, gb = int(row["home_goals"]), int(row["away_goals"])
        else:
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

    df = pd.DataFrame.from_dict(stats, orient="index").reset_index().rename(columns={"index": "team"})
    return df.sort_values(["pts", "gd", "gf"], ascending=False).reset_index(drop=True)


def qualification_probabilities(model: DixonColesModel, groups: dict[str, list[str]],
                                 fixtures: pd.DataFrame, n_sims: int = 3000,
                                 seed: int = 42) -> pd.DataFrame:
    """
    Probabilité de qualification pour les 32èmes (1er + 2ème de groupe + 8 meilleurs 3èmes),
    estimée par Monte Carlo sur les matchs de groupe restants (les matchs déjà joués sont fixés).
    """
    _MATRIX_CACHE.clear()
    rng = np.random.default_rng(seed)

    qualif_count = defaultdict(int)
    top2_count = defaultdict(int)
    third_count = defaultdict(int)

    for _ in range(n_sims):
        group_results = {g: simulate_group_remaining(model, teams, fixtures, rng) for g, teams in groups.items()}

        thirds = []
        for g, df in group_results.items():
            top2_count[df.iloc[0]["team"]] += 1
            top2_count[df.iloc[1]["team"]] += 1
            qualif_count[df.iloc[0]["team"]] += 1
            qualif_count[df.iloc[1]["team"]] += 1
            row = df.iloc[2].to_dict()
            row["group"] = g
            thirds.append(row)

        thirds_df = pd.DataFrame(thirds).sort_values(["pts", "gd", "gf"], ascending=False)
        best_8 = thirds_df.head(8)["team"].tolist()
        for t in best_8:
            qualif_count[t] += 1
            third_count[t] += 1

    rows = []
    for g, teams in groups.items():
        for t in teams:
            rows.append({
                "group": g,
                "team": t,
                "prob_qualif": qualif_count[t] / n_sims,
                "prob_top2": top2_count[t] / n_sims,
                "prob_best_third": third_count[t] / n_sims,
            })

    return pd.DataFrame(rows)


def predicted_round32(standings_map: dict[str, pd.DataFrame], qualif_df: pd.DataFrame) -> list[tuple[str, str]]:
    """
    Tableau des 32èmes déterministe à partir de l'état actuel : 1er/2e réel de
    chaque groupe (même si pas terminé) + les 8 meilleurs 3èmes selon leur proba
    de qualification, assignés aux 8 emplacements officiels (THIRD_PLACE_SLOT_GROUPS)
    en évitant qu'une équipe affronte le 1er de SON PROPRE groupe. Comme chaque
    équipe n'appartient qu'à UN groupe, ce tableau ne peut structurellement pas
    contenir de doublon.
    """
    thirds = []  # (team, own_group_letter, prob)
    for g, df in standings_map.items():
        if len(df) >= 3:
            team = df.iloc[2]["team"]
            row = qualif_df[(qualif_df["group"] == g) & (qualif_df["team"] == team)]
            prob = float(row["prob_qualif"].iloc[0]) if not row.empty else 0.0
            own_letter = g.split(" ")[-1]  # "Groupe E" -> "E"
            thirds.append((team, own_letter, prob))
    thirds.sort(key=lambda x: x[2], reverse=True)
    pool = thirds[:8]

    # Assignation greedy aux 8 slots, en évitant qu'un 3e affronte le 1er de son propre groupe
    best_8 = []
    for slot_group in THIRD_PLACE_SLOT_GROUPS:
        idx = next((i for i, (_, own, _) in enumerate(pool) if own != slot_group), 0)
        team, _, _ = pool.pop(idx)
        best_8.append(team)

    return build_bracket_32(standings_map, best_8)


def knockout_monte_carlo(model: DixonColesModel, round32_pairs: list[tuple[str, str]],
                          n_sims: int = 3000, seed: int = 42) -> dict:
    """
    Monte Carlo sur la seule phase à élimination directe, à partir d'un tableau
    de 32èmes FIXE (32 équipes réelles distinctes). Comme l'ensemble des équipes
    pouvant occuper chaque position d'un tour est un sous-ensemble disjoint des
    32 équipes de départ, aucun doublon n'est possible à aucun tour.
    """
    _MATRIX_CACHE.clear()
    rng = np.random.default_rng(seed)

    slot_counts = {
        "round16": [Counter() for _ in range(16)],
        "quarts":  [Counter() for _ in range(8)],
        "demi":    [Counter() for _ in range(4)],
        "finale":  [Counter() for _ in range(2)],
        "champion": Counter(),
    }

    for _ in range(n_sims):
        round32_winners = simulate_knockout_round(model, round32_pairs, rng)
        for i, t in enumerate(round32_winners):
            slot_counts["round16"][i][t] += 1

        round16_matchups = [(round32_winners[i], round32_winners[j]) for i, j in ROUND16_PAIRS]
        round16_winners = simulate_knockout_round(model, round16_matchups, rng)
        for i, t in enumerate(round16_winners):
            slot_counts["quarts"][i][t] += 1

        quarts_matchups = [(round16_winners[i], round16_winners[j]) for i, j in QUARTERS_PAIRS]
        quarts_winners = simulate_knockout_round(model, quarts_matchups, rng)
        for i, t in enumerate(quarts_winners):
            slot_counts["demi"][i][t] += 1

        demi_matchups = [(quarts_winners[i], quarts_winners[j]) for i, j in SEMIS_PAIRS]
        demi_winners = simulate_knockout_round(model, demi_matchups, rng)
        for i, t in enumerate(demi_winners):
            slot_counts["finale"][i][t] += 1

        champion = simulate_knockout_round(model, [(demi_winners[0], demi_winners[1])], rng)[0]
        slot_counts["champion"][champion] += 1

    def top(counter: Counter) -> tuple[str, float]:
        team, count = counter.most_common(1)[0]
        return team, count / n_sims

    bracket = {k: [top(c) for c in slot_counts[k]] for k in ("round16", "quarts", "demi", "finale")}
    bracket["champion"] = top(slot_counts["champion"])
    return bracket

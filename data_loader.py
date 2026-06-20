"""
Chargement des données depuis results.csv (martj42/international_results).
Contient tous les matchs internationaux depuis 1872, incluant la CdM 2026.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

RESULTS_CSV = "results.csv"
TODAY = datetime.today()

# Demi-vie de pondération temporelle : 10 ans → w ≈ 0.5
HALF_LIFE_DAYS = 10 * 365

TEAM_ALIASES = {
    "West Germany":    "Germany",
    "FR Yugoslavia":   "Serbia",
    "Czechoslovakia":  "Czech Republic",
    "Soviet Union":    "Russia",
    "Yugoslavia":      "Serbia",
    "United States":   "USA",
    "IR Iran":         "Iran",
    "Korea Republic":  "South Korea",
    "Korea DPR":       "North Korea",
}

# Coefficient réducteur par compétition : certaines compétitions ne reflètent pas
# le vrai niveau d'une sélection (joueurs locaux uniquement, équipes diminuées).
# Tout ce qui n'est pas listé garde un poids de 1.0 (CdM, qualifs, Euro, Copa
# América, CAN, etc. mobilisent les vrais cadres).
COMPETITION_WEIGHTS = {
    "Arab Cup": 0.3,  # réservé aux joueurs évoluant dans le championnat local
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def time_weight(date: pd.Timestamp) -> float:
    days_ago = (TODAY - date.to_pydatetime().replace(tzinfo=None)).days
    return np.exp(-np.log(2) * max(days_ago, 0) / HALF_LIFE_DAYS)


def normalize_team(name: str) -> str:
    return TEAM_ALIASES.get(name.strip(), name.strip())


# ---------------------------------------------------------------------------
# Chargement
# ---------------------------------------------------------------------------

FRIENDLY_TOURNAMENTS = {"Friendly", "Friendly tournament"}

RECENT_CUTOFF = pd.Timestamp("2020-01-01")


def load_all(csv_path: str = RESULTS_CSV) -> pd.DataFrame:
    """
    Stratégie hybride :
      - Matchs FIFA World Cup depuis 1930 (historique complet)
      - Tous matchs compétitifs depuis 2020 (couvre les équipes sans historique CdM)
    Les amicaux sont toujours exclus.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Fichier '{csv_path}' introuvable.\n"
            "Lance : curl -L https://raw.githubusercontent.com/martj42/international_results/master/results.csv -o results.csv"
        )

    df = pd.read_csv(csv_path, parse_dates=["date"])

    # Supprimer matchs sans score (futurs ou manquants)
    df = df.dropna(subset=["home_score", "away_score"])
    df = df[df["home_score"].astype(str) != "NA"]
    df["home_goals"] = df["home_score"].astype(float).astype(int)
    df["away_goals"] = df["away_score"].astype(float).astype(int)

    is_wc       = df["tournament"] == "FIFA World Cup"
    is_recent   = df["date"] >= RECENT_CUTOFF
    is_friendly = df["tournament"].isin(FRIENDLY_TOURNAMENTS)
    is_conifa   = df["tournament"].str.contains("CONIFA", na=False)

    mask = (is_wc | (is_recent & ~is_friendly)) & ~is_conifa
    df = df[mask].copy()

    # Retirer les équipes avec trop peu de matchs (estimations non fiables)
    MIN_MATCHES = 5
    counts = pd.concat([df["home_team"], df["away_team"]]).value_counts()
    valid_teams = counts[counts >= MIN_MATCHES].index
    df = df[df["home_team"].isin(valid_teams) & df["away_team"].isin(valid_teams)].copy()

    df["home_team"] = df["home_team"].map(normalize_team)
    df["away_team"] = df["away_team"].map(normalize_team)
    competition_weight = df["tournament"].map(lambda t: COMPETITION_WEIGHTS.get(t, 1.0))
    df["weight"]    = df["date"].apply(time_weight) * competition_weight
    df["stage"]     = df["tournament"]
    df["source"]    = "results_csv"
    df["neutral"]   = df["neutral"].fillna(False).astype(bool)

    df = df.sort_values("date").reset_index(drop=True)

    n_wc     = is_wc[df.index].sum() if False else (df["stage"] == "FIFA World Cup").sum()
    n_recent = len(df) - n_wc
    print(f"✅ {len(df)} matchs chargés")
    print(f"   CdM historique (1930–2026) : {n_wc} matchs")
    print(f"   Compétitifs récents (2020+) : {n_recent} matchs")
    print(f"   Équipes couvertes           : {len(set(df['home_team']) | set(df['away_team']))}")

    return df[["date", "home_team", "away_team", "home_goals", "away_goals", "stage", "weight", "source", "neutral"]]

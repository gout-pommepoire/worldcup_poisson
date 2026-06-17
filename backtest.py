"""
Backtest des prédictions sur les matchs CdM 2026 déjà joués.

Pour chaque match joué, on réentraîne le modèle SANS ce match (leave-one-out)
puis on compare la prédiction "avant-match" au résultat réel.

Deux niveaux de jugement :
  - Résultat (1N2)   : le modèle avait-il le bon vainqueur/nul ?
  - xG (buts attendus) : même si le score loupe, les buts attendus du modèle
                          étaient-ils proches des buts réels marqués ?
"""

import os
import numpy as np
import pandas as pd

from model import DixonColesModel
from data_loader import normalize_team


XG_CLOSE_THRESHOLD = 1.5   # somme des erreurs absolues (home+away) jugée "proche"
XG_REEL_CSV = "xg_reel.csv"


def get_played_2026_matches(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df["date"].dt.year == 2026) & (df["stage"] == "FIFA World Cup")
    ].reset_index(drop=True)


def load_xg_reel(path: str = XG_REEL_CSV) -> pd.DataFrame:
    """Charge les xG réels relevés manuellement (date, équipes, xG dom/ext)."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=["date", "home_team", "away_team", "xg_home_reel", "xg_away_reel"])

    xg = pd.read_csv(path)
    if xg.empty:
        return xg

    xg["date"] = pd.to_datetime(xg["date"])
    xg["home_team"] = xg["home_team"].map(normalize_team)
    xg["away_team"] = xg["away_team"].map(normalize_team)
    return xg


def outcome_label(home_goals: int, away_goals: int) -> str:
    if home_goals > away_goals:
        return "home"
    if away_goals > home_goals:
        return "away"
    return "draw"


def predicted_outcome(result: dict) -> str:
    probs = {
        "home": result["prob_home_win"],
        "draw": result["prob_draw"],
        "away": result["prob_away_win"],
    }
    return max(probs, key=probs.get)


def verdict(result_correct: bool, score_exact: bool, xg_close: bool, has_real_xg: bool) -> str:
    if score_exact:
        return "✅ Score exact"
    if result_correct:
        return "🟢 Bon résultat"
    if xg_close:
        return "🟡 Résultat raté, xG réel cohérent" if has_real_xg else "🟡 Résultat raté, xG (proxy) cohérent"
    return "🔴 Raté"


def run_backtest(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    df_all : DataFrame complet (sortie de load_all()), utilisé comme pool
             pour le ré-entraînement leave-one-out.
    """
    played = get_played_2026_matches(df_all)
    xg_reel = load_xg_reel()
    rows = []

    for i, match in played.iterrows():
        home, away = match["home_team"], match["away_team"]
        actual_h, actual_a = int(match["home_goals"]), int(match["away_goals"])

        xg_match = xg_reel[
            (xg_reel["date"] == match["date"]) &
            (xg_reel["home_team"] == home) &
            (xg_reel["away_team"] == away)
        ] if not xg_reel.empty else pd.DataFrame()
        has_real_xg = not xg_match.empty
        real_xg_h = float(xg_match.iloc[0]["xg_home_reel"]) if has_real_xg else None
        real_xg_a = float(xg_match.iloc[0]["xg_away_reel"]) if has_real_xg else None

        # Leave-one-out : on retire CE match précis du pool d'entraînement
        train_df = df_all.drop(
            df_all[
                (df_all["date"] == match["date"]) &
                (df_all["home_team"] == home) &
                (df_all["away_team"] == away)
            ].index
        )

        model = DixonColesModel()
        model.fit(train_df)

        if home not in model.att or away not in model.att:
            continue

        result = model.predict(home, away)
        lam, mu = result["expected_home_goals"], result["expected_away_goals"]

        actual_out = outcome_label(actual_h, actual_a)
        pred_out   = predicted_outcome(result)
        # Bon résultat si la proba 1N2 est correcte OU si le score le plus probable donne le bon 1N2
        pred_score_parts = result["most_likely_score"].split("-")
        pred_score_outcome = outcome_label(int(pred_score_parts[0]), int(pred_score_parts[1]))
        result_correct = (actual_out == pred_out) or (actual_out == pred_score_outcome)

        pred_score = result["most_likely_score"]
        actual_score = f"{actual_h}-{actual_a}"
        score_exact = (pred_score == actual_score)

        # Comparaison contre le xG réel si disponible, sinon contre le score final (proxy)
        if has_real_xg:
            xg_error = abs(lam - real_xg_h) + abs(mu - real_xg_a)
        else:
            xg_error = abs(lam - actual_h) + abs(mu - actual_a)
        xg_close = xg_error <= XG_CLOSE_THRESHOLD

        rows.append({
            "date": match["date"].strftime("%d/%m"),
            "home_team": home,
            "away_team": away,
            "score_reel": actual_score,
            "score_predit": pred_score,
            "xG_predit_dom": round(lam, 2),
            "xG_predit_ext": round(mu, 2),
            "xG_reel_dom": round(real_xg_h, 2) if has_real_xg else None,
            "xG_reel_ext": round(real_xg_a, 2) if has_real_xg else None,
            "has_real_xg": has_real_xg,
            "buts_reels_dom": actual_h,
            "buts_reels_ext": actual_a,
            "erreur_xG": round(xg_error, 2),
            "prob_home_win": result["prob_home_win"],
            "prob_draw": result["prob_draw"],
            "prob_away_win": result["prob_away_win"],
            "resultat_predit": {"home": "Dom.", "draw": "Nul", "away": "Ext."}[pred_out],
            "resultat_reel": {"home": "Dom.", "draw": "Nul", "away": "Ext."}[actual_out],
            "resultat_correct": result_correct,
            "score_exact": score_exact,
            "xG_coherent": xg_close,
            "verdict": verdict(result_correct, score_exact, xg_close, has_real_xg),
        })

    return pd.DataFrame(rows)


def summary_stats(backtest_df: pd.DataFrame) -> dict:
    n = len(backtest_df)
    if n == 0:
        return {}
    return {
        "n_matches": n,
        "pct_score_exact": backtest_df["score_exact"].mean(),
        "pct_resultat_correct": backtest_df["resultat_correct"].mean(),
        "pct_xg_coherent": backtest_df["xG_coherent"].mean(),
        "erreur_xg_moyenne": backtest_df["erreur_xG"].mean(),
    }

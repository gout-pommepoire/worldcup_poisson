"""
Backtest des prédictions sur les matchs CdM 2026 déjà joués.

Pour chaque match joué, on réentraîne le modèle SANS ce match (leave-one-out)
puis on compare la prédiction "avant-match" au résultat réel.

Deux niveaux de jugement :
  - Résultat (1N2)   : le modèle avait-il le bon vainqueur/nul ?
  - xG (buts attendus) : même si le score loupe, les buts attendus du modèle
                          étaient-ils proches des buts réels marqués ?
"""

import numpy as np
import pandas as pd

from model import DixonColesModel


XG_CLOSE_THRESHOLD = 1.5   # somme des erreurs absolues (home+away) jugée "proche"


def get_played_2026_matches(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df["date"].dt.year == 2026) & (df["stage"] == "FIFA World Cup")
    ].reset_index(drop=True)


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


def verdict(result_correct: bool, score_exact: bool, xg_close: bool) -> str:
    if score_exact:
        return "✅ Score exact"
    if result_correct:
        return "🟢 Bon résultat"
    if xg_close:
        return "🟡 Résultat raté, xG cohérent"
    return "🔴 Raté"


def run_backtest(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    df_all : DataFrame complet (sortie de load_all()), utilisé comme pool
             pour le ré-entraînement leave-one-out.
    """
    played = get_played_2026_matches(df_all)
    rows = []

    for i, match in played.iterrows():
        home, away = match["home_team"], match["away_team"]
        actual_h, actual_a = int(match["home_goals"]), int(match["away_goals"])

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
        result_correct = (actual_out == pred_out)

        pred_score = result["most_likely_score"]
        actual_score = f"{actual_h}-{actual_a}"
        score_exact = (pred_score == actual_score)

        xg_error = abs(lam - actual_h) + abs(mu - actual_a)
        xg_close = xg_error <= XG_CLOSE_THRESHOLD

        rows.append({
            "date": match["date"].strftime("%d/%m"),
            "home_team": home,
            "away_team": away,
            "score_reel": actual_score,
            "score_predit": pred_score,
            "xG_domicile": round(lam, 2),
            "xG_exterieur": round(mu, 2),
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
            "verdict": verdict(result_correct, score_exact, xg_close),
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

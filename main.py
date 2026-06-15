"""
Point d'entrée principal.

Usage :
    python main.py
    python main.py --match "France" "Brazil"
    python main.py --all-matches          # utilise tous les matchs compétitifs
    python main.py --strengths            # graphique des forces d'équipes
"""

import argparse
import sys

from data_loader import load_all
from model import DixonColesModel
from viz import plot_match, plot_strengths


def parse_args():
    p = argparse.ArgumentParser(description="Modèle Dixon-Coles — Coupe du Monde 2026")
    p.add_argument("--csv",        default="results.csv", help="Chemin vers results.csv")
    p.add_argument("--match",   nargs=2, metavar=("HOME", "AWAY"),
                   help='Équipes à prédire, ex: "France" "Brazil"')
    p.add_argument("--top-scores", type=int, default=10)
    p.add_argument("--strengths", action="store_true",
                   help="Afficher le graphique des forces d'équipes")
    return p.parse_args()


def main():
    args = parse_args()

    print("\n=== CHARGEMENT DES DONNÉES ===")
    df = load_all(csv_path=args.csv)

    print("\n=== ENTRAÎNEMENT DU MODÈLE ===")
    model = DixonColesModel()
    model.fit(df)

    strengths = model.team_strengths()
    print("\n=== TOP 20 ÉQUIPES (force globale) ===")
    print(strengths.head(20).to_string(index=False))

    if args.strengths:
        plot_strengths(strengths, top_n=20)

    # Prédiction
    if args.match:
        home, away = args.match
    else:
        top2 = strengths["team"].head(2).tolist()
        home, away = top2[0], top2[1]
        print(f"\n(Exemple automatique : {home} vs {away})")

    print(f"\n=== PRÉDICTION : {home} vs {away} ===")
    try:
        result = model.predict(home, away)
        print(f"  Victoire {home:25s} : {result['prob_home_win']:.1%}")
        print(f"  Match nul                       : {result['prob_draw']:.1%}")
        print(f"  Victoire {away:25s} : {result['prob_away_win']:.1%}")
        print(f"  Score le + probable             : {result['most_likely_score']}")
        print(f"  Buts attendus  {home} {result['expected_home_goals']} — {result['expected_away_goals']} {away}")

        print(f"\n  Top {args.top_scores} scores les plus probables :")
        print(model.top_scores(home, away, n=args.top_scores).to_string(index=False))

        plot_match(result)

    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

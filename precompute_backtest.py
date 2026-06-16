"""
Précalcule le backtest (leave-one-out) en local et sauvegarde le résultat.

À lancer à la main après chaque mise à jour de results.csv ou xg_reel.csv :
    python precompute_backtest.py

L'app Streamlit lit ensuite directement backtest_results.csv (rapide,
pas de calcul lourd sur le serveur web — évite les timeouts/boucles
sur Streamlit Cloud).
"""

import time

from data_loader import load_all
from backtest import run_backtest, summary_stats

OUTPUT_CSV = "backtest_results.csv"


def main():
    print("Chargement des données…")
    df = load_all()

    print("Calcul du backtest (leave-one-out, ~30s par match joué)…")
    t0 = time.time()
    results = run_backtest(df)
    print(f"Terminé en {time.time()-t0:.0f}s")

    results.to_csv(OUTPUT_CSV, index=False)
    print(f"Sauvegardé dans {OUTPUT_CSV}")

    stats = summary_stats(results)
    print()
    print("=== Résumé ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

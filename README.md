# ⚽ Prédicteur Coupe du Monde 2026 — Modèle Dixon-Coles

App Streamlit : prédiction de matchs + bilan des pronostics, basée sur un modèle
de Poisson Dixon-Coles (1997) entraîné sur l'historique des matchs internationaux.

## Installation

```bash
pip install -r requirements.txt
```

## Lancer l'app

```bash
streamlit run app.py
```

Onglets :
- **🎯 Match** : choisis deux des 48 équipes qualifiées pour la CdM 2026, le modèle
  donne les probabilités 1N2, la distribution complète des scores et le top des
  scores les plus probables.
- **📈 Bilan** : compare les pronostics du modèle (calculés en *leave-one-out*,
  sans tricher) aux résultats réels des matchs déjà joués.

## Données

### `results.csv`
Source : [martj42/international_results](https://github.com/martj42/international_results)
— tous les matchs internationaux depuis 1872, mis à jour régulièrement, incluant
les matchs CdM 2026 au fur et à mesure qu'ils sont joués.

```bash
curl -L https://raw.githubusercontent.com/martj42/international_results/master/results.csv -o results.csv
```

Le fichier contient une colonne `neutral` (terrain neutre ou non) — essentielle
pour la CdM 2026 où seuls les matchs des 3 pays hôtes (USA, Mexique, Canada) sont
joués à domicile au sens propre, tous les autres sont en terrain neutre.

**Workflow de mise à jour quotidien (pendant le tournoi) :**
1. Remplir les scores manquants (`NA`) dans `results.csv` une fois les matchs joués
2. Relever le xG réel sur [Sofascore](https://www.sofascore.com) après chaque
   match et l'ajouter dans `xg_reel.csv`
3. Lancer en local :
   ```bash
   python precompute_backtest.py
   ```
4. Commit/push les 3 fichiers ensemble :
   ```bash
   git add results.csv xg_reel.csv backtest_results.csv
   git commit -m "MAJ données"
   git push
   ```

### `xg_reel.csv`
xG réels relevés manuellement sur Sofascore après chaque match (format :
`date,home_team,away_team,xg_home_reel,xg_away_reel`, noms d'équipes en anglais).
Utilisé dans l'onglet Bilan pour juger un pronostic "raté au score mais cohérent
sur le fond" (ex : domination sans concrétisation).

### `backtest_results.csv`
Précalculé en local par `precompute_backtest.py` (jamais sur le serveur Streamlit
Cloud — le leave-one-out sur tous les matchs joués est trop lourd pour le tier
gratuit et provoquait des boucles de redémarrage).

## Modèle

**Dixon-Coles (1997)** avec pondération temporelle et correction des scores faibles :

```
λ_home = α × att_home × def_away × home_advantage   (home_advantage = 1 si terrain neutre)
λ_away = α × att_away × def_home

P(x, y) = τ(x,y) × Poisson(x; λ_home) × Poisson(y; λ_away)
```

- **Pondération temporelle** : poids exponentiel, demi-vie 10 ans — un match
  d'il y a 10 ans pèse moitié moins qu'un match d'aujourd'hui.
- **Terrain neutre** : `home_advantage` n'est appliqué que pour les matchs des
  pays hôtes (USA/Mexique/Canada) jouant chez eux. Pour tous les autres matchs
  de Coupe du Monde (joués en terrain neutre), `home_advantage = 1`.
- **Correction τ** : ajustement sur les scores faibles (0-0, 1-0, 0-1, 1-1),
  où la corrélation entre buts domicile/extérieur est plus marquée.
- **Pool d'entraînement** : tous les matchs de Coupe du Monde depuis 1930 +
  tous les matchs compétitifs (hors amicaux) depuis 2020, pour couvrir les
  équipes sans historique CdM (ex : Cap-Vert).
- **Estimation** : maximum de vraisemblance pondérée (L-BFGS-B), un paramètre
  attaque/défense par équipe + avantage du terrain (ρ) + corrélation τ (ρ) +
  facteur d'échelle global (α).

## Fichiers

| Fichier | Rôle |
|---|---|
| `data_loader.py` | Chargement et nettoyage de `results.csv` |
| `model.py` | Modèle Dixon-Coles (fit, predict, score_matrix) |
| `backtest.py` | Backtest leave-one-out vs résultats réels / xG Sofascore |
| `precompute_backtest.py` | Script à lancer en local pour générer `backtest_results.csv` |
| `app.py` | Interface Streamlit |

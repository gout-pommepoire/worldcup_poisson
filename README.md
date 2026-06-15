# Modèle Dixon-Coles — Coupe du Monde 2026

## Installation

```bash
pip install -r requirements.txt
```

## Données

### 1. Historique 1930–2022 (obligatoire)
1. Va sur https://www.kaggle.com/datasets/abecklas/fifa-world-cup
2. Télécharge **WorldCupMatches.csv**
3. Place-le dans ce dossier

### 2. Matchs 2026 en cours (optionnel)
1. Crée un compte gratuit sur https://www.football-data.org/client/register
2. Récupère ta clé API (email de confirmation)
3. Passe-la avec `--api-key`

## Utilisation

```bash
# Avec données historiques seulement
python main.py

# Avec données 2026 en temps réel
python main.py --api-key TON_API_KEY

# Prédire un match spécifique
python main.py --match "France" "Brazil"

# Avec données live + match spécifique
python main.py --api-key TON_API_KEY --match "Argentina" "Germany"

# Afficher le graphique des forces d'équipes
python main.py --strengths
```

## Modèle

**Dixon-Coles (1997)** avec :
- Pondération temporelle exponentielle (demi-vie 10 ans)
- Correction τ sur les scores faibles (0-0, 1-0, 0-1, 1-1)
- Paramètres : force d'attaque, force de défense, avantage domicile, ρ

```
λ_home = α × att_home × def_away × home_advantage
λ_away = α × att_away × def_home

P(x, y) = τ(x,y) × Poisson(x; λ_home) × Poisson(y; λ_away)
```

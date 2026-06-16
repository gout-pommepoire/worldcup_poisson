"""
Modèle de Poisson Dixon-Coles avec pondération temporelle.

Principe :
  λ_home = α * att_home * def_away * home_advantage
  λ_away = α * att_away * def_home

  P(x, y) = τ(x,y) * Poisson(x; λ_home) * Poisson(y; λ_away)

  τ : correction Dixon-Coles sur les scores faibles (0-0, 1-0, 0-1, 1-1)
  pondération : chaque match contribue avec son poids temporel.
"""

import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from scipy.special import gammaln
from itertools import product

warnings.filterwarnings("ignore")

MAX_GOALS = 8   # grille de scores simulés


# ---------------------------------------------------------------------------
# Correction Dixon-Coles τ
# ---------------------------------------------------------------------------

def tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    if x == 0 and y == 0:
        return 1 - lam * mu * rho
    if x == 0 and y == 1:
        return 1 + lam * rho
    if x == 1 and y == 0:
        return 1 + mu * rho
    if x == 1 and y == 1:
        return 1 - rho
    return 1.0


# ---------------------------------------------------------------------------
# Log-vraisemblance négative pondérée — version vectorisée
# ---------------------------------------------------------------------------

def neg_log_likelihood(params: np.ndarray, hg: np.ndarray, ag: np.ndarray,
                        hi: np.ndarray, ai: np.ndarray, w: np.ndarray,
                        n: int) -> float:
    """
    params[:n-1]   = att des équipes 1..n (att[0] = 1 fixé pour identifiabilité)
    params[n-1:2n-1] = def
    params[2n-1]   = home_adv
    params[2n]     = rho
    params[2n+1]   = alpha
    """
    att_free = params[:n-1]
    att_full = np.concatenate([[1.0], att_free])   # att[0] = 1 fixé

    defe     = params[n-1:2*n-1]
    home_adv = params[2*n-1]
    rho      = params[2*n]
    alpha    = params[2*n+1]

    lam = np.maximum(alpha * att_full[hi] * defe[ai] * home_adv, 1e-6)
    mu  = np.maximum(alpha * att_full[ai] * defe[hi],            1e-6)

    ll = w * (
        hg * np.log(lam) - lam - gammaln(hg + 1)
      + ag * np.log(mu)  - mu  - gammaln(ag + 1)
    )

    # Correction Dixon-Coles τ
    tau_arr = np.ones(len(hg))
    m00 = (hg == 0) & (ag == 0);  tau_arr[m00] = np.maximum(1 - lam[m00]*mu[m00]*rho, 1e-10)
    m10 = (hg == 1) & (ag == 0);  tau_arr[m10] = 1 + mu[m10]  * rho
    m01 = (hg == 0) & (ag == 1);  tau_arr[m01] = 1 + lam[m01] * rho
    m11 = (hg == 1) & (ag == 1);  tau_arr[m11] = 1 - rho

    ll += w * np.log(np.maximum(tau_arr, 1e-10))
    return -ll.sum()


# ---------------------------------------------------------------------------
# Entraînement
# ---------------------------------------------------------------------------

class DixonColesModel:
    def __init__(self):
        self.teams: list[str] = []
        self.att:   dict[str, float] = {}
        self.defe:  dict[str, float] = {}
        self.home_adv: float = 1.0
        self.rho:      float = 0.0
        self.alpha:    float = 1.0
        self.fitted:   bool  = False

    def fit(self, df: pd.DataFrame) -> "DixonColesModel":
        teams = sorted(set(df["home_team"]) | set(df["away_team"]))
        self.teams = teams
        n = len(teams)
        idx = {t: i for i, t in enumerate(teams)}

        # Pré-calcul des arrays numpy (vectorisation)
        valid = df["home_team"].isin(idx) & df["away_team"].isin(idx)
        df = df[valid].reset_index(drop=True)
        hi = np.array([idx[t] for t in df["home_team"]])
        ai = np.array([idx[t] for t in df["away_team"]])
        hg = df["home_goals"].to_numpy(dtype=int)
        ag = df["away_goals"].to_numpy(dtype=int)
        w  = df["weight"].to_numpy(dtype=float)

        # att[0] fixé à 1 → (n-1) att libres + n def + home_adv + rho + alpha = 2n+2 params
        x0 = np.ones(2*n + 2)
        x0[-3] = 1.1    # home_adv
        x0[-2] = -0.1   # rho
        x0[-1] = 0.3    # alpha

        bounds = (
            [(0.05, 15.0)] * (n-1) +   # att libres
            [(0.02, 15.0)] * n +        # def
            [(1.0, 2.0), (-0.99, 0.99), (0.01, 5.0)]
        )

        print("⏳ Optimisation en cours…")
        result = minimize(
            neg_log_likelihood,
            x0,
            args=(hg, ag, hi, ai, w, n),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 3000, "maxfun": 60000, "ftol": 1e-9, "gtol": 1e-6},
        )

        if not result.success:
            print(f"⚠️  Convergence partielle : {result.message}")

        att_full = np.concatenate([[1.0], result.x[:n-1]])
        self.att      = dict(zip(teams, att_full))
        self.defe     = dict(zip(teams, result.x[n-1:2*n-1]))
        self.home_adv = result.x[2*n-1]
        self.rho      = result.x[2*n]
        self.alpha    = result.x[2*n+1]
        self.fitted   = True

        print(f"✅ Modèle entraîné sur {len(teams)} équipes")
        print(f"   home_adv={self.home_adv:.3f} | rho={self.rho:.3f} | alpha={self.alpha:.3f}")
        return self

    # -----------------------------------------------------------------------
    # Prédiction
    # -----------------------------------------------------------------------

    def _lambdas(self, home: str, away: str) -> tuple[float, float]:
        lam = self.alpha * self.att[home] * self.defe[away] * self.home_adv
        mu  = self.alpha * self.att[away] * self.defe[home]
        return max(lam, 1e-6), max(mu, 1e-6)

    def score_matrix(self, home: str, away: str, max_goals: int = MAX_GOALS) -> np.ndarray:
        """Matrice (max_goals+1, max_goals+1) des probabilités de score — vectorisée."""
        lam, mu = self._lambdas(home, away)
        goals = np.arange(max_goals + 1)

        log_pmf_h = goals * np.log(lam) - lam - gammaln(goals + 1)
        log_pmf_a = goals * np.log(mu)  - mu  - gammaln(goals + 1)
        matrix = np.exp(log_pmf_h[:, None] + log_pmf_a[None, :])

        rho = self.rho
        matrix[0, 0] *= max(1 - lam * mu * rho, 1e-10)
        if max_goals >= 1:
            matrix[1, 0] *= 1 + mu * rho
            matrix[0, 1] *= 1 + lam * rho
            matrix[1, 1] *= 1 - rho

        matrix /= matrix.sum()
        return matrix

    def predict(self, home: str, away: str) -> dict:
        """
        Retourne :
          - score_matrix  : probabilités de chaque score
          - prob_home_win : P(domicile gagne)
          - prob_draw     : P(nul)
          - prob_away_win : P(extérieur gagne)
          - most_likely   : score le plus probable
          - expected      : (buts_dom attendus, buts_ext attendus)
        """
        self._check_teams(home, away)
        mat = self.score_matrix(home, away)

        p_home = float(np.tril(mat, -1).sum())   # hg > ag
        p_draw = float(np.trace(mat))
        p_away = float(np.triu(mat, 1).sum())    # ag > hg

        idx = np.unravel_index(mat.argmax(), mat.shape)
        most_likely = (int(idx[0]), int(idx[1]))

        lam, mu = self._lambdas(home, away)

        return {
            "home_team":     home,
            "away_team":     away,
            "prob_home_win": round(p_home, 4),
            "prob_draw":     round(p_draw, 4),
            "prob_away_win": round(p_away, 4),
            "most_likely_score": f"{most_likely[0]}-{most_likely[1]}",
            "expected_home_goals": round(lam, 2),
            "expected_away_goals": round(mu, 2),
            "score_matrix":  mat,
        }

    def top_scores(self, home: str, away: str, n: int = 10) -> pd.DataFrame:
        """Top N scores les plus probables."""
        self._check_teams(home, away)
        mat = self.score_matrix(home, away)
        rows = [
            {"score": f"{hg}-{ag}", "prob": mat[hg, ag]}
            for hg, ag in product(range(MAX_GOALS + 1), range(MAX_GOALS + 1))
        ]
        return (
            pd.DataFrame(rows)
            .sort_values("prob", ascending=False)
            .head(n)
            .reset_index(drop=True)
            .assign(prob=lambda d: d["prob"].map("{:.2%}".format))
        )

    def team_strengths(self) -> pd.DataFrame:
        """Tableau des forces attaque/défense de toutes les équipes."""
        return (
            pd.DataFrame({
                "team":   self.teams,
                "attack": [self.att[t] for t in self.teams],
                "defense":[self.defe[t] for t in self.teams],
            })
            .assign(overall=lambda d: d["attack"] / d["defense"])
            .sort_values("overall", ascending=False)
            .reset_index(drop=True)
        )

    def _check_teams(self, *teams):
        for t in teams:
            if t not in self.att:
                known = [k for k in self.att if t.lower() in k.lower()]
                hint = f" Peut-être : {known}" if known else ""
                raise ValueError(f"Équipe inconnue : '{t}'.{hint}")

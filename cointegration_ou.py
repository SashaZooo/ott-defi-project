import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
"""
Module 2 : Cointégration, Processus d'Ornstein-Uhlenbeck (OU) et Signaux Dynamiques EWMA.
"""

import numpy as np
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import pandas as pd
from scipy.stats import linregress
from statsmodels.tsa.stattools import adfuller
from config import DEX_PAIRS, EWMA_SPAN, ADF_PVALUE_MAX


def fit_ornstein_uhlenbeck(spread_series: np.ndarray) -> dict:
    """
    Calibre un processus d'Ornstein-Uhlenbeck discret : S_t - S_{t-1} = a + b * S_{t-1} + e_t
    Retourne la vitesse de retour à la moyenne (theta), la demi-vie (half_life) en heures et sigma_ou.
    """
    s_prev = spread_series[:-1]
    ds = np.diff(spread_series)
    slope, intercept, _, _, _ = linregress(s_prev, ds)

    # b = slope = -(1 - exp(-theta * dt)) avec dt = 1 heure
    if slope >= -1e-5:
        theta = 0.01  # Borne minimale de sécurité si retour lent
    else:
        theta = -np.log(max(1.0 + slope, 1e-4))

    half_life = np.clip(np.log(2.0) / theta, 2.0, 240.0)
    residuals = ds - (intercept + slope * s_prev)
    sigma_ou = np.std(residuals)

    return {
        "theta": float(theta),
        "half_life": float(half_life),
        "sigma_ou": float(sigma_ou)
    }


def analyze_and_build_signals(df: pd.DataFrame, formation_end_idx: int) -> dict:
    """
    1. Teste la corrélation et la cointégration sur la période de formation (In-Sample).
    2. Calibre le processus d'Ornstein-Uhlenbeck (demi-vie mr) et les covariances.
    3. Génère les spreads et les Z-scores dynamiques EWMA sur toute la période sans look-ahead bias.
    """
    log_df = np.log(df.drop(columns=["Date"]))
    formation_log = log_df.iloc[:formation_end_idx]
    returns_formation = formation_log.diff().dropna()

    spreads_df = pd.DataFrame(index=df.index)
    zscores_df = pd.DataFrame(index=df.index)
    pair_metrics = {}

    for pair_name, cfg in DEX_PAIRS.items():
        leg1, leg2 = cfg["leg1"], cfg["leg2"]

        # 1. Corrélation de Pearson sur la période de formation
        corr = formation_log[leg1].corr(formation_log[leg2])

        # 2. Régression linéaire (Engle-Granger étape 1) UNIQUEMENT sur la période de formation
        res = linregress(formation_log[leg2], formation_log[leg1])
        hedge_ratio = res.slope
        intercept = res.intercept

        # Construction du spread sur l'ensemble des dates avec les paramètres de formation
        full_spread = log_df[leg1] - (hedge_ratio * log_df[leg2] + intercept)
        form_spread = full_spread.iloc[:formation_end_idx].to_numpy()

        # 3. Test ADF de stationnarité sur le spread de formation
        _, pvalue, _, _, _, _ = adfuller(form_spread, regression="c")

        # 4. Calibration Ornstein-Uhlenbeck sur la période de formation
        ou_params = fit_ornstein_uhlenbeck(form_spread)
        mr_hours = ou_params["half_life"]

        # 5. Calcul du rendement attendu (EP) et de la matrice de covariance 2x2 (Risk)
        # Correction par rapport au papier : on utilise la demi-vie OU et l'amplitude d'écart-type
        ret_leg1 = returns_formation[leg1].abs().mean()
        ret_leg2 = returns_formation[leg2].abs().mean()
        cov_2x2 = returns_formation[[leg1, leg2]].cov().to_numpy() * mr_hours

        # 6. Z-Score dynamique par moyenne mobile exponentielle (EWMA) - causal (sans futur)
        ewma_mean = full_spread.ewm(span=EWMA_SPAN, min_periods=24).mean()
        ewma_std = full_spread.ewm(span=EWMA_SPAN, min_periods=24).std().replace(0, np.nan).bfill()
        dynamic_zscore = ((full_spread - ewma_mean) / ewma_std).fillna(0.0)

        spreads_df[pair_name] = full_spread
        zscores_df[pair_name] = dynamic_zscore

        pair_metrics[pair_name] = {
            "universe": cfg["universe"],
            "leg1": leg1,
            "leg2": leg2,
            "corr": float(corr),
            "adf_pvalue": float(pvalue),
            "cointegrated": bool(pvalue <= ADF_PVALUE_MAX),
            "hedge_ratio": float(hedge_ratio),
            "theta_ou": ou_params["theta"],
            "half_life_hrs": mr_hours,
            "expected_profit_vec": np.array([ret_leg1 * mr_hours, ret_leg2 * mr_hours]),
            "cov_matrix": cov_2x2,
            "swap_fee": cfg["swap_fee"],
            "pool_liquidity": cfg["pool_liquidity"]
        }

    return {
        "spreads": spreads_df,
        "zscores": zscores_df,
        "pair_metrics": pair_metrics
    }
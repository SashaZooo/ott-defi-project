"""
Module 2 : Cointégration (Engle-Granger / MacKinnon), Processus OU et Signaux EWMA causaux.
- Stablecoins : stationnarité directe I(0) autour de la parité (ADF).
- LST (wstETH/WETH) : cointégration avec dérive déterministe (trend='ct') capturant le staking yield.
- DeFi (UNI, AAVE) : cointégration standard d'Engle-Granger (trend='c').
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import linregress
from statsmodels.tsa.stattools import adfuller, coint
from config import DEX_PAIRS, EWMA_SPAN, ADF_PVALUE_MAX


def fit_ornstein_uhlenbeck(spread_series: np.ndarray) -> dict:
    """Calibre un processus d'Ornstein-Uhlenbeck discret par régression AR(1) sur ΔS_t."""
    s_prev = spread_series[:-1]
    ds = np.diff(spread_series)
    slope, intercept, _, _, _ = linregress(s_prev, ds)

    if slope >= -1e-5:
        theta = 0.01
    else:
        theta = -np.log(max(1.0 + slope, 1e-4))

    half_life = np.clip(np.log(2.0) / theta, 2.0, 240.0)
    residuals = ds - (intercept + slope * s_prev)
    sigma_ou = np.std(residuals)

    return {"theta": float(theta), "half_life": float(half_life), "sigma_ou": float(sigma_ou)}


def analyze_and_build_signals(df: pd.DataFrame, formation_end_idx: int) -> dict:
    """Calibre la cointégration et l'OU sur l'In-Sample uniquement, puis construit le Z-score EWMA."""
    log_df = np.log(df.drop(columns=["Date"]))
    formation_log = log_df.iloc[:formation_end_idx]
    returns_formation = formation_log.diff().dropna()

    spreads_df = pd.DataFrame(index=df.index)
    zscores_df = pd.DataFrame(index=df.index)
    pair_metrics = {}

    t_full = np.arange(len(df), dtype=float)
    t_form = t_full[:formation_end_idx]

    for pair_name, cfg in DEX_PAIRS.items():
        leg1, leg2 = cfg["leg1"], cfg["leg2"]
        corr = formation_log[leg1].corr(formation_log[leg2])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if cfg["universe"] == "Stablecoins":
                hedge_ratio = 1.0
                intercept = 0.0
                drift = 0.0
                full_spread = log_df[leg1] - log_df[leg2]
                form_spread = full_spread.iloc[:formation_end_idx].to_numpy()
                res_adf = adfuller(form_spread, regression="c")
                pvalue = float(res_adf[1])

            elif cfg["universe"] == "LST":
                X_form = np.column_stack([np.ones(len(t_form)), formation_log[leg2].to_numpy(), t_form])
                coeffs, _, _, _ = np.linalg.lstsq(X_form, formation_log[leg1].to_numpy(), rcond=None)
                intercept, hedge_ratio, drift = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])
                full_spread = log_df[leg1] - (intercept + hedge_ratio * log_df[leg2] + drift * t_full)
                form_spread = full_spread.iloc[:formation_end_idx].to_numpy()
                _, pvalue, _ = coint(formation_log[leg1], formation_log[leg2], trend="ct")

            else:
                res = linregress(formation_log[leg2], formation_log[leg1])
                hedge_ratio = float(res.slope)
                intercept = float(res.intercept)
                drift = 0.0
                full_spread = log_df[leg1] - (hedge_ratio * log_df[leg2] + intercept)
                form_spread = full_spread.iloc[:formation_end_idx].to_numpy()
                _, pvalue, _ = coint(formation_log[leg1], formation_log[leg2], trend="c")

        ou_params = fit_ornstein_uhlenbeck(form_spread)
        mr_hours = ou_params["half_life"]

        expected_spread_profit = ou_params["sigma_ou"] * np.sqrt(mr_hours)
        cov_2x2 = returns_formation[[leg1, leg2]].cov().to_numpy() * mr_hours

        ewma_mean = full_spread.ewm(span=EWMA_SPAN, min_periods=24).mean()
        ewma_std = full_spread.ewm(span=EWMA_SPAN, min_periods=24).std()
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
            "intercept": float(intercept),
            "drift": float(drift),
            "theta_ou": ou_params["theta"],
            "sigma_ou": ou_params["sigma_ou"],
            "half_life_hrs": mr_hours,
            "expected_profit": float(expected_spread_profit),
            "cov_matrix": cov_2x2,
            "swap_fee": cfg["swap_fee"],
            "pool_liquidity": cfg["pool_liquidity"],
        }

    return {"spreads": spreads_df, "zscores": zscores_df, "pair_metrics": pair_metrics}

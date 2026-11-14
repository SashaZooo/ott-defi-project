"""
Module 3 : Solveur d'optimisation bi-objectif (Markowitz Market-Neutral avec frictions DeFi).
"""

import numpy as np
from scipy.optimize import minimize


def solve_optimal_weights(
    active_signals: list,
    pair_metrics: dict,
    committed_weights: dict,
    risk_aversion_lambda: float,
    capital_per_asset: float
) -> dict:
    """
    Résout le problème bi-objectif (Équation 6) pour toutes les paires déclenchant un signal.
    active_signals : liste de dicts [{"pair": str, "long_asset": str, "short_asset": str, "z_abs": float}]
    Retourne un dictionnaire {pair_name: (w_long, w_short)}
    """
    n_signals = len(active_signals)
    if n_signals == 0:
        return {}

    # Pour chaque paire n, la contrainte de neutralité au marché (Éq. 5) lie directement
    # le poids short au poids long : w_short = - w_long * (1 - f_n) / (1 + f_n)
    # On optimise donc sur le vecteur x = [w_long_1, w_long_2, ..., w_long_k] dans [0, 1]^k
    friction_ratios = []
    ep_scalars = []
    quad_risk_scalars = []

    for sig in active_signals:
        p_name = sig["pair"]
        pm = pair_metrics[p_name]

        # Friction proportionnelle totale estimée (Frais de swap AMM + Slippage moyen)
        est_slippage = (capital_per_asset * 0.5) / (2.0 * pm["pool_liquidity"])
        total_f = pm["swap_fee"] + est_slippage
        ratio = (1.0 - total_f) / (1.0 + total_f)
        friction_ratios.append(ratio)

        # Profit espéré net pour 1 unité de poids long (Éq. 1 ajustée de la force du signal Z)
        signal_multiplier = min(sig["z_abs"] / 2.0, 2.5)
        raw_ep = np.sum(pm["expected_profit_vec"]) * signal_multiplier
        net_ep = max(raw_ep - 2.0 * total_f, 0.0005)
        ep_scalars.append(net_ep)

        # Risque de la paire (Éq. 2) : W_n * (COV_n (x) [[1, -1], [-1, 1]]) * W_n^T
        # Sans modifier COV_n en place (correction du bug de optimal_trading_technique.py)
        sign_mat = np.array([[1.0, -1.0], [-1.0, 1.0]])
        signed_cov = pm["cov_matrix"] * sign_mat
        w_unit = np.array([1.0, -ratio])
        pair_variance = float(w_unit @ signed_cov @ w_unit.T)
        quad_risk_scalars.append(max(pair_variance, 1e-6))

    ep_vec = np.array(ep_scalars)
    risk_vec = np.array(quad_risk_scalars)
    ratios_vec = np.array(friction_ratios)

    # Fonction objectif à minimiser : -(Profit_Espéré - lambda * Variance)
    def objective(w_longs):
        expected_return = np.dot(w_longs, ep_vec)
        # Pénalité quadratique de risque + régularisation L2 pour diversification
        portfolio_risk = np.dot(w_longs ** 2, risk_vec * 100.0 + 0.002)
        return -(expected_return - risk_aversion_lambda * portfolio_risk)

    # Contrainte de budget par actif (Éq. 4) :
    # tw_c + somme(w_long sur c) + somme(|w_short| sur c) <= 1.0
    involved_assets = set()
    for sig in active_signals:
        involved_assets.add(sig["long_asset"])
        involved_assets.add(sig["short_asset"])

    constraints = []
    for asset in involved_assets:
        avail_budget = max(1.0 - committed_weights.get(asset, 0.0), 0.0)

        def asset_Constraint(w_longs, target_asset=asset, budget=avail_budget):
            usage = 0.0
            for idx, sig in enumerate(active_signals):
                if sig["long_asset"] == target_asset:
                    usage += w_longs[idx]
                if sig["short_asset"] == target_asset:
                    usage += w_longs[idx] * ratios_vec[idx]
            return budget - usage

        constraints.append({"type": "ineq", "fun": asset_Constraint})

    bounds = [(0.05, 0.85) for _ in range(n_signals)]
    x0 = np.full(n_signals, min(0.35, 0.85 / n_signals))

    res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)

    optimal_allocations = {}
    w_sol = res.x if res.success else x0
    for idx, sig in enumerate(active_signals):
        w_l = float(np.clip(w_sol[idx], 0.0, 1.0))
        w_s = -w_l * ratios_vec[idx]
        if w_l >= 0.02:
            optimal_allocations[sig["pair"]] = (w_l, w_s)

    return optimal_allocations
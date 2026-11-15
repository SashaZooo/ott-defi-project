"""
Module 3 : Solveur bi-objectif SLSQP.
- Intègre les frais de swap et le slippage concentré Uniswap V3 dans la fonction objectif.
- Normalise le compromis Rendement/Variance pour rendre l'aversion au risque lambda pleinement effective.
"""
import numpy as np
from scipy.optimize import minimize

# Facteur de concentration de liquidité Uniswap V3 par tier de frais (liquidité virtuelle active autour du tick courant)
V3_CONCENTRATION_FACTOR = {
    0.0001: 25.0,  # Pools 1bp (Stablecoins / wstETH) : liquidité concentrée à ±0.2% du peg
    0.0005: 10.0,  # Pools 5bp
    0.0030: 4.0,   # Pools 30bp (DeFi)
}


def effective_v3_liquidity(tvl_usd: float, swap_fee: float) -> float:
    mult = V3_CONCENTRATION_FACTOR.get(round(swap_fee, 4), 5.0)
    return max(tvl_usd * mult, 100_000.0)


def solve_optimal_weights(
    active_signals: list,
    pair_metrics: dict,
    committed_weights: dict,
    risk_aversion_lambda: float,
    capital_per_asset: float,
) -> dict:
    if not active_signals:
        return {}

    tradable_signals = []
    ep_scalars, risk_scalars, slip_Scalars, ratios = [], [], [], []

    for sig in active_signals:
        pm = pair_metrics[sig["pair"]]
        beta = float(np.clip(abs(pm["hedge_ratio"]), 0.1, 2.0))

        # Frais de swap aller-retour (1 swap direct dans le pool à l'entrée + 1 à la sortie)
        roundtrip_swap_fee = 2.0 * pm["swap_fee"]
        # Espérance de retour à la moyenne d'Ornstein-Uhlenbeck : E[|ΔS|] = sigma_ou * |Z| * (1 - e^{-theta * mr})
        gross_ep = pm["expected_profit"] * sig["z_abs"] * (1.0 - np.exp(-pm["theta_ou"] * pm["half_life_hrs"]))
        net_ep_zero_size = gross_ep - roundtrip_swap_fee

        if net_ep_zero_size <= 0.0:
            continue

        w_unit = np.array([1.0, -beta])
        pair_var = float(w_unit @ pm["cov_matrix"] @ w_unit.T)
        # Écart-type du spread sur l'horizon de demi-vie (même échelle que gross_ep)
        pair_std = max(np.sqrt(max(pair_var, 1e-10)), 1e-4)

        eff_liq = effective_v3_liquidity(pm["pool_liquidity"], pm["swap_fee"])
        # Coût de slippage aller-retour pour w=1.0 : 2 * (capital / (2 * L_eff)) = capital / L_eff
        roundtrip_slip_at_full = capital_per_asset / eff_liq

        tradable_signals.append(sig)
        ep_scalars.append(net_ep_zero_size)
        # Pénalité quadratique mise à l'échelle de l'écart-type + impact de marché quadratique
        risk_scalars.append(pair_std)
        slip_Scalars.append(roundtrip_slip_at_full)
        ratios.append(beta)

    n_signals = len(tradable_signals)
    if n_signals == 0:
        return {}

    ep_vec = np.array(ep_scalars)
    risk_vec = np.array(risk_scalars)
    slip_vec = np.array(slip_Scalars)
    r_vec = np.array(ratios)

    def objective(w_longs):
        # Profit net attendu (linéaire en w) moins slippage quadratique (w^2) moins pénalité de risque (lambda * w^2)
        expected_net_gain = np.dot(w_longs, ep_vec) - np.dot(w_longs ** 2, slip_vec)
        risk_penalty = risk_aversion_lambda * np.dot(w_longs ** 2, risk_vec)
        return -(expected_net_gain - risk_penalty)

    involved = {s["long_asset"] for s in tradable_signals} | {s["short_asset"] for s in tradable_signals}
    constraints = []
    for asset in involved:
        avail = max(1.0 - committed_weights.get(asset, 0.0), 0.0)

        def constr(w_longs, target=asset, budget=avail):
            usage = 0.0
            for i, sig in enumerate(tradable_signals):
                if sig["long_asset"] == target:
                    usage += w_longs[i]
                if sig["short_asset"] == target:
                    usage += w_longs[i] * r_vec[i]
            return budget - usage

        constraints.append({"type": "ineq", "fun": constr})

    bounds = [(0.0, 1.0) for _ in range(n_signals)]
    x0 = np.full(n_signals, 0.25)
    res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)

    optimal = {}
    for idx, sig in enumerate(tradable_signals):
        w_l = float(np.clip(res.x[idx], 0.0, 1.0))
        w_s = -float(np.clip(w_l * r_vec[idx], 0.0, 1.0))
        if w_l > 0.01:
            optimal[sig["pair"]] = (w_l, w_s)

    return optimal

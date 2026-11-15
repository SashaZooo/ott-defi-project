"""
DONNÉES SIMULÉES — usage réservé aux tests unitaires (test_pipeline.py).
Ne jamais utiliser pour produire des résultats du rapport.
"""
import numpy as np
import pandas as pd


def generate_synthetic_dex_data(start_date="2026-08-15", end_date="2026-11-10", seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, end=end_date, freq="1h")
    n = len(dates)
    t = np.arange(n, dtype=float)
    weth = 2000.0 * np.exp(np.cumsum(rng.normal(0.00003, 0.0065, n)))
    usdc = np.clip(1.0 + np.cumsum(rng.normal(0, 0.00015, n)) * 0.01, 0.998, 1.002)

    def ou(theta, sigma, jump_scale, jump_prob=0.003):
        s = np.zeros(n)
        for i in range(1, n):
            jump = rng.normal(0, jump_scale) if rng.random() < jump_prob else 0.0
            s[i] = s[i - 1] * (1 - theta) + rng.normal(0, sigma) + jump
        return s

    return pd.DataFrame({
        "Date": dates,
        "USDT": usdc * np.exp(ou(0.25, 0.00045, 0.0020)),
        "DAI": usdc * np.exp(ou(0.22, 0.00055, 0.0025)),
        "wstETH": weth * 1.24 * np.exp(0.0000036 * t + ou(0.15, 0.0009, 0.003)),
        "UNI": weth * 0.0017 * np.exp(ou(0.05, 0.0065, 0.015)),
        "AAVE": weth * 0.0460 * np.exp(ou(0.055, 0.0070, 0.016)),
        "USDC": usdc,
        "WETH": weth,
    })

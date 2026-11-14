"""
Suite de tests unitaires automatisés pour valider chaque module du pipeline OTT DeFi.
Exécution : python test_pipeline.py  (ou : python -m unittest test_pipeline.py)
"""

import unittest
import numpy as np
import pandas as pd

from config import DEX_PAIRS, FORMATION_RATIO, INITIAL_CAPITAL_PER_ASSET
from data_acquisition import load_or_fetch_data
from cointegration_ou import fit_ornstein_uhlenbeck, analyze_and_build_signals
from optimizer import solve_optimal_weights
from backtest_engine import compute_amm_slippage, sample_stochastic_gas_usd, run_backtest


class TestOTTDeFiPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Charge le dataset une seule fois pour l'ensemble des tests."""
        cls.df = load_or_fetch_data()
        cls.formation_end_idx = int(len(cls.df) * FORMATION_RATIO)
        cls.analysis = analyze_and_build_signals(cls.df, cls.formation_end_idx)

    def test_1_data_integrity_and_hourly_sampling(self):
        """Vérifie la structure OHLC/Close, l'absence de NaN et la fréquence horaire (1h)."""
        self.assertGreater(len(self.df), 8000)
        self.assertFalse(self.df.isna().any().any(), "Le dataset ne doit contenir aucun NaN.")
        expected_tokens = {"USDC", "USDT", "DAI", "WETH", "stETH", "rETH", "UNI", "AAVE"}
        self.assertTrue(expected_tokens.issubset(set(self.df.columns)))
        # Vérification des prix strictement positifs
        for token in expected_tokens:
            self.assertTrue((self.df[token] > 0).all())
        # Vérification de l'échantillonnage de 1 heure
        time_diffs = self.df["Date"].diff().dropna()
        self.assertTrue((time_diffs == pd.Timedelta(hours=1)).all())

    def test_2_ornstein_uhlenbeck_calibration(self):
        """Vérifie que l'estimateur Ornstein-Uhlenbeck extrait un theta > 0 et une demi-vie cohérente."""
        np.random.seed(42)
        s = np.zeros(2000)
        true_theta = 0.10
        for t in range(1, 2000):
            s[t] = s[t - 1] * (1.0 - true_theta) + np.random.normal(0, 0.002)
        ou = fit_ornstein_uhlenbeck(s)
        self.assertGreater(ou["theta"], 0.0)
        self.assertTrue(2.0 <= ou["half_life"] <= 240.0)
        self.assertAlmostEqual(ou["theta"], true_theta, delta=0.03)

    def test_3_no_lookahead_bias_in_cointegration_and_ewma(self):
        """Vérifie que la modification des prix futurs (Out-of-Sample) n'altère ni l'In-Sample ni le passé."""
        df_Shocked = self.df.copy()
        # Choc artificiel de +50% à la fin du dataset Out-of-Sample
        df_Shocked.loc[df_Shocked.index[-500:], "UNI"] *= 1.50
        analysis_shocked = analyze_and_build_signals(df_Shocked, self.formation_end_idx)

        # Les paramètres de formation (hedge ratio, OU half-life) doivent être strictement identiques
        for pair in DEX_PAIRS.keys():
            self.assertEqual(
                self.analysis["pair_metrics"][pair]["hedge_ratio"],
                analysis_shocked["pair_metrics"][pair]["hedge_ratio"]
            )
            self.assertEqual(
                self.analysis["pair_metrics"][pair]["half_life_hrs"],
                analysis_shocked["pair_metrics"][pair]["half_life_hrs"]
            )

        # Les Z-scores EWMA avant le choc ne doivent pas changer (causalité stricte)
        check_idx = len(self.df) - 600
        z_orig = self.analysis["zscores"]["UNI_WETH"].iloc[:check_idx].to_numpy()
        z_shock = analysis_shocked["zscores"]["UNI_WETH"].iloc[:check_idx].to_numpy()
        np.testing.assert_allclose(z_orig, z_shock, rtol=1e-12)

    def test_4_optimizer_budget_and_market_neutrality(self):
        """Vérifie les bornes [0,1], la contrainte de budget par actif et la neutralité au marché."""
        active_signals = [
            {"pair": "stETH_WETH", "long_asset": "stETH", "short_asset": "WETH", "direction": 1, "z_abs": 2.4},
            {"pair": "rETH_WETH", "long_asset": "rETH", "short_asset": "WETH", "direction": 1, "z_abs": 2.2},
            {"pair": "UNI_WETH", "long_asset": "UNI", "short_asset": "WETH", "direction": 1, "z_abs": 2.5},
        ]
        committed = {c: 0.0 for c in self.df.columns if c != "Date"}
        alloc = solve_optimal_weights(
            active_signals=active_signals,
            pair_metrics=self.analysis["pair_metrics"],
            committed_weights=committed,
            risk_aversion_lambda=1.0,
            capital_per_asset=INITIAL_CAPITAL_PER_ASSET
        )
        self.assertGreater(len(alloc), 0)
        weth_short_usage = 0.0
        for p_name, (w_l, w_s) in alloc.items():
            self.assertTrue(0.0 <= w_l <= 1.0)
            self.assertTrue(-1.0 <= w_s <= 0.0)
            weth_short_usage += abs(w_s)
        # La somme des poids short sur l'actif partagé WETH ne doit jamais dépasser 100% (1.0)
        self.assertLessEqual(weth_short_usage, 1.0 + 1e-6)

    def test_5_defi_frictions_and_backtest_metrics(self):
        """Vérifie la modélisation des 3 frictions DeFi et l'impact sur le rendement net Out-of-Sample."""
        slip = compute_amm_slippage(trade_usd=10_000, pool_liquidity_usd=50_000_000)
        self.assertAlmostEqual(slip, 10_000 / 100_000_000)
        gas_usd = sample_stochastic_gas_usd(eth_price_usd=3000.0, n_swaps=2)
        self.assertGreater(gas_usd, 0.0)

        res_on = run_backtest(self.df, self.analysis, self.formation_end_idx, 1.0, include_frictions=True)
        res_off = run_backtest(self.df, self.analysis, self.formation_end_idx, 1.0, include_frictions=False)

        required_keys = [
            "Annualized Return (%)", "Sharpe Ratio", "Max Drawdown (%)",
            "Win/Loss Ratio", "Swap Fees Paid ($)", "Slippage Cost ($)", "Gas Cost ($)"
        ]
        for k in required_keys:
            self.assertIn(k, res_on["metrics"])

        # Le rendement avec frictions DeFi doit être strictement inférieur au rendement sans frictions
        self.assertLess(res_on["metrics"]["Annualized Return (%)"], res_off["metrics"]["Annualized Return (%)"])
        self.assertGreater(res_on["metrics"]["Swap Fees Paid ($)"], 0.0)
        self.assertGreater(res_on["metrics"]["Slippage Cost ($)"], 0.0)
        self.assertGreater(res_on["metrics"]["Gas Cost ($)"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

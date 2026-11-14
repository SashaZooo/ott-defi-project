"""
Configuration centrale du projet OTT - Extension DeFi.
"""

# 1. Fenêtre temporelle et échantillonnage (Q3 2025 - Q3 2026, fréquence 1 heure)
START_DATE = "2025-07-01"
END_DATE = "2026-07-01"
TIMEFRAME = "1h"
FORMATION_RATIO = 0.35  # 35% In-Sample (Formation), 65% Out-of-Sample (Backtest)

# 2. Univers d'actifs DEX (Uniswap V3) et paramètres de friction par pool
# Chaque paire définit : (Actif_1, Actif_2, Frais_Swap_AMM, Profondeur_Liquidite_USD)
DEX_PAIRS = {
    # Univers 1 : Stablecoins (Frais Uniswap V3 0.01%)
    "USDT_USDC": {
        "universe": "Stablecoins",
        "leg1": "USDT",
        "leg2": "USDC",
        "swap_fee": 0.0001,        # 0.01%
        "pool_liquidity": 150_000_000,
        "gecko_pool": "0x3416cf6c708da44db2624d63ea0aaef7113527c6"
    },
    "DAI_USDC": {
        "universe": "Stablecoins",
        "leg1": "DAI",
        "leg2": "USDC",
        "swap_fee": 0.0001,        # 0.01%
        "pool_liquidity": 80_000_000,
        "gecko_pool": "0x5777d92f208679db4b9778590fa3cab3ac9e2168"
    },
    # Univers 2 : Liquid Staking Derivatives - LST (Frais 0.01% et 0.05%)
    "stETH_WETH": {
        "universe": "LST",
        "leg1": "stETH",
        "leg2": "WETH",
        "swap_fee": 0.0001,        # 0.01%
        "pool_liquidity": 200_000_000,
        "gecko_pool": "0x109830a1aaad605bbf02a9dfa7b0b92ec2fb7daa"
    },
    "rETH_WETH": {
        "universe": "LST",
        "leg1": "rETH",
        "leg2": "WETH",
        "swap_fee": 0.0005,        # 0.05%
        "pool_liquidity": 65_000_000,
        "gecko_pool": "0xa4e0faa58465a2d369aa21b3e42d43374c6f9613"
    },
    # Univers 3 : Écosystème DeFi (Frais 0.30%)
    "UNI_WETH": {
        "universe": "DeFi",
        "leg1": "UNI",
        "leg2": "WETH",
        "swap_fee": 0.0030,        # 0.30%
        "pool_liquidity": 45_000_000,
        "gecko_pool": "0x1d42064fc4beb5f8aaf85f4617ae8b3b5b8bd801"
    },
    "AAVE_WETH": {
        "universe": "DeFi",
        "leg1": "AAVE",
        "leg2": "WETH",
        "swap_fee": 0.0030,        # 0.30%
        "pool_liquidity": 35_000_000,
        "gecko_pool": "0x5ab53ee1d50eef2c1dd3d5402789cd27bb52c1bb"
    }
}

# 3. Paramètres de modélisation et signaux (Extension OU + EWMA)
EWMA_SPAN = 72             # Fenêtre glissante de 72 heures (3 jours) pour le Z-score
OPEN_THRESHOLD = 2.0       # Ouverture à +/- 2.0 écarts-types (corrigé du bug d'échelle)
CLOSE_THRESHOLD = 0.4      # Fermeture à +/- 0.4 écart-type
ADF_PVALUE_MAX = 0.10      # Seuil de stationnarité du test Augmented Dickey-Fuller

# 4. Paramètres du portefeuille et d'optimisation
INITIAL_CAPITAL_PER_ASSET = 25_000.0  # 25 000 USD équivalent par actif (150k$ au total)
RISK_FREE_RATE = 0.04                 # Taux sans risque annuel (4%)
LAMBDA_LIST = [0.5, 1.0, 2.0]         # Aversion au risque à tester
DEFAULT_LAMBDA = 1.0

# 5. Paramètres des coûts de gaz stochastiques (Réseau Ethereum / L2)
GAS_UNITS_PER_SWAP = 135_000          # Unités de gaz consommées par swap Uniswap V3
MEAN_GWEI = 3.5                       # Moyenne Gwei (environnement post-Dencun / L2)
STD_GWEI = 0.45                       # Volatilité log-normale du gaz
ETH_REFERENCE_PRICE_USD = 3100.0      # Prix de référence ETH pour convertir le gaz en USD
"""
Configuration centrale du projet OTT - Extension DeFi (Données Réelles Uniswap V3).
"""

# 1. Fenêtre temporelle réelle (88 jours / ~2 089 heures, fréquence 1 heure)
START_DATE = "2026-08-15"
END_DATE = "2026-11-10"
TIMEFRAME = "1h"
FORMATION_RATIO = 0.35  # 35% In-Sample (Formation), 65% Out-of-Sample (Backtest)

# 2. Source de données : GeckoTerminal (agrégateur DEX), réseau Ethereum mainnet
GECKO_NETWORK = "eth"

TOKENS = {
    "USDC":   "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
    "USDT":   "0xdac17f958d2ee523a2206206994597c13d831ec7",
    "DAI":    "0x6b175474e89094c44da98b954eedeac495271d0f",
    "WETH":   "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
    "wstETH": "0x7f39c581f595b53c5cb19bd0b3f8da6c935e2ca0",
    "UNI":    "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984",
    "AAVE":   "0x7fc66500c84a76ad7e9c93437bfc5ac33e2ddae9",
}

# 5 paires couvrant les 3 univers exigés par l'énoncé (Section 2.A)
DEX_PAIRS = {
    "USDT_USDC": {
        "universe": "Stablecoins", "leg1": "USDT", "leg2": "USDC",
        "swap_fee": 0.0001, "pool_liquidity": 33_500_000
    },
    "DAI_USDC": {
        "universe": "Stablecoins", "leg1": "DAI", "leg2": "USDC",
        "swap_fee": 0.0001, "pool_liquidity": 1_000_000
    },
    "wstETH_WETH": {
        "universe": "LST", "leg1": "wstETH", "leg2": "WETH",
        "swap_fee": 0.0001, "pool_liquidity": 3_700_000
    },
    "UNI_WETH": {
        "universe": "DeFi", "leg1": "UNI", "leg2": "WETH",
        "swap_fee": 0.0030, "pool_liquidity": 11_400_000
    },
    "AAVE_WETH": {
        "universe": "DeFi", "leg1": "AAVE", "leg2": "WETH",
        "swap_fee": 0.0030, "pool_liquidity": 5_500_000
    },
}

# 3. Paramètres de modélisation et signaux (Extension OU + EWMA)
EWMA_SPAN = 72
OPEN_THRESHOLD = 2.0
CLOSE_THRESHOLD = 0.4
ADF_PVALUE_MAX = 0.10

# 4. Paramètres du portefeuille et d'optimisation
INITIAL_CAPITAL_PER_ASSET = 25_000.0  # 25 000 USD x 7 actifs = 175 000 USD
RISK_FREE_RATE = 0.04
LAMBDA_LIST = [0.5, 1.0, 2.0]
DEFAULT_LAMBDA = 1.0

# 5. Paramètres des coûts de gaz stochastiques
GAS_UNITS_PER_SWAP = 135_000
MEAN_GWEI = 3.5
STD_GWEI = 0.45
ETH_REFERENCE_PRICE_USD = 3100.0

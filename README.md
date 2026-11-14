# Extensions of the Optimal Trading Technique (OTT) in DeFi

Modélisation, backtest Out-of-Sample et extension du modèle OTT de Yang & Malik (2024) aux pools AMM Uniswap V3.

## Structure des Modules
- config.py : Paramètres des pools DEX (Stablecoins, LST, DeFi), frais AMM (0.01%, 0.05%, 0.30%), liquidité et gaz en Gwei.
- data_acquisition.py : Extraction et génération des séries horaires DEX (Q3 2025 - Q3 2026).
- cointegration_ou.py : Cointégration sans look-ahead bias, processus Ornstein-Uhlenbeck (demi-vie) et Z-score dynamique EWMA.
- optimizer.py : Solveur convexe bi-objectif (SLSQP) sous contraintes de neutralité au marché et de frictions.
- backtest_engine.py : Simulation Out-of-Sample intégrant frais de swap variables, slippage AMM et coûts de gaz stochastiques.
- main.py : Exécution du pipeline complet et génération des tables et figures.
- generate_report.py : Génération automatique du rapport LaTeX (report.tex).

## Exécution
pip install -r requirements.txt
python main.py
python generate_report.py

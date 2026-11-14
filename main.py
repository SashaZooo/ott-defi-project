"""
Point d'entrée principal : Exécute le pipeline complet OTT DeFi et génère les tableaux et figures.
"""

import pandas as pd
import matplotlib.pyplot as plt
from config import FORMATION_RATIO, LAMBDA_LIST
from data_acquisition import load_or_fetch_data
from cointegration_ou import analyze_and_build_signals
from backtest_engine import run_backtest


def main():
    print("=" * 75)
    print(" ALGORITHMIC TRADING PROJECT : EXTENSIONS OF OTT IN DEFI (UNISWAP V3)")
    print("=" * 75)

    # 1. Chargement des données
    df = load_or_fetch_data()
    formation_end_idx = int(len(df) * FORMATION_RATIO)
    print(f"Période totale     : {df['Date'].iloc[0]} -> {df['Date'].iloc[-1]} ({len(df)}h)")
    print(f"Période Formation  : {formation_end_idx} heures (In-Sample)")
    print(f"Période Backtest   : {len(df) - formation_end_idx} heures (Out-of-Sample)\n")

    # 2. Cointégration, Ornstein-Uhlenbeck et Signaux EWMA
    analysis = analyze_and_build_signals(df, formation_end_idx)
    coint_rows = []
    for pair_name, pm in analysis["pair_metrics"].items():
        coint_rows.append({
            "Pair": pair_name,
            "Universe": pm["universe"],
            "Corr": round(pm["corr"], 4),
            "ADF p-val": round(pm["adf_pvalue"], 4),
            "Coint": "Yes" if pm["cointegrated"] else "No",
            "OU Theta": round(pm["theta_ou"], 4),
            "Half-Life (h)": round(pm["half_life_hrs"], 2),
            "AMM Fee (%)": round(pm["swap_fee"] * 100, 2)
        })
    coint_table = pd.DataFrame(coint_rows)
    print("--- TABLEAU 1 : SCREENING COINTÉGRATION & ORNSTEIN-UHLENBECK (IN-SAMPLE) ---")
    print(coint_table.to_string(index=False))
    print()

    # 3. Backtest Out-of-Sample pour différents niveaux d'aversion au risque (Lambda)
    perf_rows = []
    curves = {}

    for lam in LAMBDA_LIST:
        res_fric = run_backtest(
            df, analysis, start_idx=formation_end_idx, risk_aversion_lambda=lam, include_frictions=True
        )
        m_f = res_fric["metrics"]
        m_f["Scenario"] = f"OTT DeFi (lambda={lam}, Frictions ON)"
        perf_rows.append(m_f)
        curves[f"OTT DeFi (λ={lam})"] = res_fric["equity_curve"]

    # Comparaison sans frictions pour mesurer l'impact des coûts DeFi (lambda = 1.0)
    res_no_fric = run_backtest(
        df, analysis, start_idx=formation_end_idx, risk_aversion_lambda=1.0, include_frictions=False
    )
    m_nf = res_no_fric["metrics"]
    m_nf["Scenario"] = "OTT Ideal (lambda=1.0, Frictions OFF)"
    perf_rows.append(m_nf)
    curves["OTT Sans Frictions (λ=1.0)"] = res_no_fric["equity_curve"]

    perf_df = pd.DataFrame(perf_rows)
    cols_order = ["Scenario"] + [c for c in perf_df.columns if c != "Scenario"]
    perf_df = perf_df[cols_order]

    print("--- TABLEAU 2 : RÉSULTATS DU BACKTEST OUT-OF-SAMPLE & IMPACT DES FRICTIONS ---")
    print(perf_df.to_string(index=False))
    print()

    # Sauvegarde des tableaux en CSV pour le rapport
    coint_table.to_csv("data/table1_cointegration_ou.csv", index=False)
    perf_df.to_csv("data/table2_backtest_metrics.csv", index=False)

    # 4. Génération de la figure de performance pour le rapport LaTeX
    plt.figure(figsize=(11, 5.5))
    for label, eq_df in curves.items():
        linestyle = "--" if "Sans Frictions" in label else "-"
        plt.plot(eq_df["Date"], eq_df["Equity"], label=label, linestyle=linestyle, linewidth=1.8)

    plt.title("Out-of-Sample Portfolio Equity Curve - OTT DeFi Extension (Q4 2025 - Q3 2026)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (USD Equivalent)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results_ott_defi.png", dpi=300)
    print("[Succès] Graphique sauvegardé sous 'results_ott_defi.png' et tables exportées dans 'data/'.")


if __name__ == "__main__":
    main()
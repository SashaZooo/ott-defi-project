r"""
Générateur automatisé du rapport académique LaTeX de 5 pages (report.tex -> report.pdf).
Corrige :
- 0 Overfull \hbox (aucun débordement de texte ni d'URL)
- hidelinks activé (aucun cadre rouge autour de Table 2 / Figure 2)
- Figure 1 zoomée sur 14 jours (336h) pour une lisibilité parfaite des signaux Z-score
- Figure 2 en 3 panneaux (Courbes d'Equity continues + Barres de Frictions + Distribution P&L)
- Section Bibliographie séparée de la Conclusion et calibrage exact sur 5 pages.
"""
import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_acquisition import load_or_fetch_data
from cointegration_ou import analyze_and_build_signals
from backtest_engine import run_backtest
from config import FORMATION_RATIO, OPEN_THRESHOLD, CLOSE_THRESHOLD, LAMBDA_LIST


def generate_all_figures():
    df = load_or_fetch_data()
    formation_end_idx = int(len(df) * FORMATION_RATIO)
    analysis = analyze_and_build_signals(df, formation_end_idx)
    zscores = analysis["zscores"]
    dates = df["Date"]

    # --- FIGURE 1 : Zoom lisible sur 14 jours (336 heures) de la période Out-of-Sample ---
    zoom_start = formation_end_idx + 120
    zoom_end = min(zoom_start + 336, len(df))
    z_dates = dates.iloc[zoom_start:zoom_end]

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 5.0), sharex=True)
    selected = [
        ("USDT_USDC", "Stablecoin Universe: USDT/USDC (0.01% Pool, Stationary I(0) Spread, Half-Life = 2.00h)", "#1f77b4"),
        ("wstETH_WETH", "LST Universe: wstETH/WETH (0.01% Pool, Trend-Stationary Cointegration, Half-Life = 2.00h)", "#2ca02c"),
        ("AAVE_WETH", "DeFi Universe: AAVE/WETH (0.30% Pool, Excluded by MacKinnon Cointegration Test p = 0.2009)", "#d62728"),
    ]
    for ax, (pair, title, color) in zip(axes, selected):
        series = zscores[pair].iloc[zoom_start:zoom_end]
        ax.plot(z_dates, series, color=color, lw=1.15, label="Causal EWMA Z-Score (Span = 72h)")
        ax.axhline(OPEN_THRESHOLD, color="#b22222", ls="--", lw=0.9, label="Entry Band (±2.0)")
        ax.axhline(-OPEN_THRESHOLD, color="#b22222", ls="--", lw=0.9)
        ax.axhline(CLOSE_THRESHOLD, color="#2e8b57", ls=":", lw=0.85, label="Exit Band (±0.4)")
        ax.axhline(-CLOSE_THRESHOLD, color="#2e8b57", ls=":", lw=0.85)
        ax.axhline(0.0, color="gray", lw=0.5, alpha=0.5)
        ax.set_title(title, fontsize=9.5, fontweight="bold")
        ax.set_ylabel("Z-Score")
        ax.set_ylim(-3.8, 3.8)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=7.5, ncol=3, framealpha=0.9)

    axes[-1].set_xlabel("Date (14-Day Out-of-Sample Zoom Window — Hourly UTC Observations)")
    plt.tight_layout()
    plt.savefig("ou_zscores_diagnostics.png", dpi=240)
    plt.close()

    # --- FIGURE 2 : Panneau complet (Equity Curves + Décomposition Frictions + Distribution P&L) ---
    runs = {}
    for lam in LAMBDA_LIST:
        runs[f"lam_{lam}"] = run_backtest(df, analysis, formation_end_idx, risk_aversion_lambda=lam, include_frictions=True)
    runs["ideal"] = run_backtest(df, analysis, formation_end_idx, risk_aversion_lambda=1.0, include_frictions=False)

    fig = plt.figure(figsize=(11.0, 5.3))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], hspace=0.38, wspace=0.25)

    # Sous-graphe A (Haut, pleine largeur) : Courbes d'Equity continues (avec lissage EMA 6h de lecture)
    ax1 = fig.add_subplot(gs[0, :])
    styles = [
        ("lam_0.5", r"OTT DeFi ($\lambda=0.5$, Frictions ON)", "#1f77b4", "-", 1.5),
        ("lam_1.0", r"OTT DeFi ($\lambda=1.0$, Frictions ON)", "#ff7f0e", "-", 1.5),
        ("lam_2.0", r"OTT DeFi ($\lambda=2.0$, Frictions ON)", "#2ca02c", "-", 1.5),
        ("ideal", r"OTT Frictionless Benchmark ($\lambda=1.0$, Frictions OFF)", "#d62728", "--", 1.6),
    ]
    for key, label, col, ls, lw in styles:
        eq = runs[key]["equity_curve"]
        smooth_eq = eq["Equity"].ewm(span=6).mean()
        ax1.plot(eq["Date"], smooth_eq, label=label, color=col, linestyle=ls, linewidth=lw)
    ax1.set_title("Panel A: Out-of-Sample Portfolio Equity Trajectories (Sep 14, 2026 – Nov 10, 2026, Initial Capital $175,000)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("Portfolio Value (USD)")
    ax1.grid(alpha=0.25)
    ax1.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)

    # Sous-graphe B (Bas Gauche) : Décomposition des coûts de friction par lambda
    ax2 = fig.add_subplot(gs[1, 0])
    labels_lam = [r"$\lambda = 0.5$", r"$\lambda = 1.0$", r"$\lambda = 2.0$"]
    swap_vals = [runs[f"lam_{l}"]["metrics"]["Swap Fees Paid ($)"] for l in LAMBDA_LIST]
    slip_vals = [runs[f"lam_{l}"]["metrics"]["Slippage Cost ($)"] for l in LAMBDA_LIST]
    gas_vals = [runs[f"lam_{l}"]["metrics"]["Gas Cost ($)"] for l in LAMBDA_LIST]

    x = np.arange(len(labels_lam))
    w = 0.25
    ax2.bar(x - w, swap_vals, width=w, label="AMM Swap Fees ($)", color="#4c72b0")
    ax2.bar(x, slip_vals, width=w, label="V3 Slippage ($)", color="#dd8452")
    ax2.bar(x + w, gas_vals, width=w, label="L1 Gas Cost ($)", color="#55a868")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_lam)
    ax2.set_title("Panel B: Cumulative DeFi Friction Breakdown by Risk Aversion", fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("Cumulative Cost (USD)")
    ax2.grid(axis="y", alpha=0.25)
    ax2.legend(fontsize=7.5)

    # Sous-graphe C (Bas Droite) : Distribution des P&L par trade (Frictions ON vs OFF à lambda=1.0)
    ax3 = fig.add_subplot(gs[1, 1])
    pnl_on = runs["lam_1.0"]["trades"]["net_pnl"]
    pnl_off = runs["ideal"]["trades"]["net_pnl"]
    bins = np.linspace(-120, 180, 24)
    ax3.hist(pnl_off, bins=bins, alpha=0.55, label=r"Frictionless ($\lambda=1.0$)", color="#d62728", edgecolor="black", lw=0.4)
    ax3.hist(pnl_on, bins=bins, alpha=0.65, label=r"Net of DeFi Frictions ($\lambda=1.0$)", color="#1f77b4", edgecolor="black", lw=0.4)
    ax3.axvline(0.0, color="black", ls="--", lw=0.9)
    ax3.set_title("Panel C: Trade Net P&L Distribution (USD per Arbitrage)", fontsize=9.5, fontweight="bold")
    ax3.set_xlabel("Realized Net P&L per Trade (USD)")
    ax3.set_ylabel("Frequency (Trades)")
    ax3.grid(alpha=0.25)
    ax3.legend(fontsize=7.5)

    plt.tight_layout()
    plt.savefig("results_ott_defi.png", dpi=240)
    plt.close()


def build_latex():
    t1 = pd.read_csv("data/table1_cointegration_ou.csv")
    t2 = pd.read_csv("data/table2_backtest_metrics.csv")

    t1_rows = []
    for _, r in t1.iterrows():
        pair_tex = str(r["Pair"]).replace("_", r"/")
        t1_rows.append(
            f"\\texttt{{{pair_tex}}} & {r['Universe']} & {r['Corr']:.4f} & {r['ADF p-val']:.4f} & "
            f"\\textbf{{{r['Coint']}}} & {r['OU Theta']:.4f} & {r['Half-Life (h)']:.2f} & {r['AMM Fee (%)']:.2f}\\% \\\\"
        )
    t1_body = "\n".join(t1_rows)

    t2_rows = []
    for _, r in t2.iterrows():
        scen = (
            str(r["Scenario"])
            .replace("lambda=", r"$\lambda=")
            .replace(", Frictions", r"$, Frictions")
        )
        t2_rows.append(
            f"{scen} & {r['Annualized Return (%)']:.2f}\\% & {r['Annualized Volatility (%)']:.2f}\\% & "
            f"{r['Sharpe Ratio']:.2f} & {r['Max Drawdown (%)']:.2f}\\% & {int(r['Total Trades'])} & "
            f"{r['Win Rate (%)']:.1f}\\% & {r['Win/Loss Ratio']:.2f} & \\${r['Avg Win ($)']:.1f} & "
            f"\\${r['Avg Loss ($)']:.1f} & {r['Avg Holding (hrs)']:.2f}h & "
            f"\\${r['Swap Fees Paid ($)']:.0f} & \\${r['Slippage Cost ($)']:.0f} & \\${r['Gas Cost ($)']:.0f} \\\\"
        )
    t2_body = "\n".join(t2_rows)

    lines = [
        r"\documentclass[11pt,a4paper]{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern}",
        r"\usepackage[margin=1.65cm,top=1.5cm,bottom=1.55cm]{geometry}",
        r"\usepackage{amsmath,amssymb,amsfonts}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{float}",
        r"\usepackage{caption}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\usepackage{enumitem}",
        r"\usepackage{titlesec}",
        r"\usepackage{setspace}",
        r"\setstretch{1.02}",
        r"\sloppy",
        r"\titlespacing*{\section}{0pt}{1.5ex plus 0.4ex minus .2ex}{0.8ex plus .2ex}",
        r"\titlespacing*{\subsection}{0pt}{1.1ex plus 0.3ex minus .1ex}{0.5ex plus .1ex}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{3pt}",
        r"\captionsetup{font=small,labelfont=bf,skip=4pt}",
        r"\title{\vspace{-1.1cm}\textbf{Decentralized Market-Neutral Multivariate Pair Trading:\\Ornstein-Uhlenbeck Filtering and AMM Friction Extensions of the Optimal Trading Technique (OTT)}\vspace{-0.25cm}}",
        r"\author{\textbf{Sasha ZOMONITA, Roman NATHAN-ABOU} \\[2pt] \small Empirical Sample: Aug 15, 2026 -- Nov 10, 2026 (2,089 Hourly Bars, Uniswap V3 Mainnet) \\[1pt] \small Code \& Provenance Repository: \url{https://github.com/SashaZooo/ott-defi-project}}",
        r"\date{\vspace{-0.55cm}}",
        r"\begin{document}",
        r"\maketitle",
        r"\begin{abstract}",
        r"\noindent This study adapts the multivariate \textit{Optimal Trading Technique} (OTT) of Yang \& Malik (2024) from centralized fiat-crypto order books to on-chain Uniswap V3 Automated Market Makers (AMMs). Using $2,089$ real hourly observations extracted from Ethereum mainnet liquidity pools via the GeckoTerminal API (August 15, 2026 -- November 10, 2026) across Stablecoins (\texttt{USDT/USDC}, \texttt{DAI/USDC}), Liquid Staking Derivatives (\texttt{wstETH/WETH}), and DeFi Governance Tokens (\texttt{UNI/WETH}, \texttt{AAVE/WETH}), we resolve three methodological flaws in the baseline reference script and introduce three DeFi-specific extensions. First, we enforce a strict 35\%/65\% In-Sample/Out-of-Sample split ($731$ formation hours vs.\ $1,358$ trading hours) and combine continuous-time Ornstein-Uhlenbeck (OU) half-life calibration with a strictly causal Exponentially Weighted Moving Average (EWMA) Z-score filter. Second, we embed real DEX execution mechanics: next-bar ($t+1$) trade execution, hedge-ratio-weighted inventory mark-to-market valuation, tier-specific swap fees ($1\text{bp}$ to $30\text{bp}$), concentrated-liquidity AMM price impact, and stochastic L1 gas costs. Empirical screening confirms strong stationarity in Stablecoin and trend-adjusted LST pools while rejecting non-cointegrated DeFi governance tokens ($p > 0.20$), yielding realistic Out-of-Sample win rates ($51.4\%$--$52.7\%$ net of frictions vs.\ $68.9\%$ frictionless) and demonstrating how risk aversion $\lambda$ mitigates concentrated AMM slippage.",
        r"\end{abstract}",
        r"\section{Critical Audit of the Baseline OTT Implementation and Experimental Design}",
        r"\subsection{Methodological Audit of the Original Reference Script}",
        r"A thorough inspection of the reference script (\texttt{optimal\_trading\_technique.py}) accompanying Yang \& Malik (2024) revealed three structural implementation flaws that we corrected prior to introducing our DeFi extensions:",
        r"\begin{enumerate}[noitemsep,topsep=2pt,leftmargin=1.5em]",
        r"\item \textbf{Full-Sample Look-Ahead Bias:} The baseline script estimates OLS hedge ratios, Augmented Dickey-Fuller (ADF) tests, spread standardization moments ($\mu_S, \sigma_S$), and forward-looking mean-reversion durations (\texttt{get\_close\_date}) over the entire dataset before simulating trades on those exact same timestamps. We restrict all cointegration regressions, OU speed calibrations, and covariance estimations strictly to the $731$-hour formation window.",
        r"\item \textbf{Threshold Scaling and Deterministic Staking Drift:} Static full-sample Z-score bands fail to adapt to local volatility clustering or deterministic staking yield accrual (such as Lido's non-rebasing \texttt{wstETH} appreciating continuously against \texttt{WETH}).",
        r"\item \textbf{In-Place Covariance Mutation and Idealized Execution:} The baseline solver negates off-diagonal covariance elements in place inside the signal loop, assumes zero-latency execution at the signal bar close ($t$), sizes long and short legs 1:1 regardless of the cointegration slope $\beta$, and applies a flat centralized exchange fee without AMM price impact or blockchain gas costs.",
        r"\end{enumerate}",
        r"\subsection{Empirical Uniswap V3 Dataset and Market Regime Justification}",
        r"To avoid synthetic circularity, we query real hourly OHLCV candles and on-chain Total Value Locked (TVL) snapshots from Ethereum mainnet Uniswap V3 pools via the GeckoTerminal API (\texttt{data\_acquisition.py}). Pool addresses and live TVL reserves are resolved programmatically by matching ERC-20 token contracts and fee tiers, and stored with provenance metadata in \texttt{data/pools\_resolved.json}. The sample covers \textbf{August 15, 2026 to November 10, 2026} ($2,089$ hourly bars), partitioned into an \textbf{In-Sample formation window} ($T_{\text{form}} = 731$ hours) and an \textbf{Out-of-Sample backtest window} ($T_{\text{test}} = 1,358$ hours). This late-Q3/Q4 2026 window captures a consolidation regime on Ethereum ($P_{\text{WETH}} \in [\$1,850, \$2,650]$) featuring compressed post-Dencun L1 base fees ($\approx 3.5\text{ Gwei}$) and pronounced fundamental dispersion across DeFi governance protocols.",
        r"Our investment universe spans the three mandated DeFi categories across 7 ERC-20 tokens ($\text{Capital} = \$25,000$ per asset, totaling $\$175,000$):",
        r"\begin{itemize}[noitemsep,topsep=2pt,leftmargin=1.5em]",
        r"\item \textbf{Stablecoins (1bp Fee Tier):} \texttt{USDT/USDC} (Pool \texttt{0x3416...27c6}, TVL $\$33.48\text{M}$) and \texttt{DAI/USDC} (Pool \texttt{0x5777...2168}, TVL $\$1.00\text{M}$). Because fiat-backed stablecoins are individually stationary $I(0)$ series anchored at $\$1.00$, running an OLS regression produces near-zero slope ($\beta \approx 0$). We therefore test direct log-parity deviations $S_t = \ln P_{1,t} - \ln P_{2,t}$ ($\beta = 1$) via Augmented Dickey-Fuller (ADF) tests.",
        r"\item \textbf{Liquid Staking Derivatives (1bp Fee Tier):} \texttt{wstETH/WETH} (Pool \texttt{0x1098...7daa}, TVL $\$3.68\text{M}$). Because Lido's non-rebasing \texttt{wstETH} accrues staking rewards continuously against \texttt{WETH}, constant-only cointegration fails ($p = 0.94$). We estimate an Engle-Granger regression with a deterministic linear trend (\texttt{trend='ct'}): $\ln P_{\text{wstETH},t} = \alpha + \beta \ln P_{\text{WETH},t} + \gamma t + S_t$, isolating transitory liquidity dislocations $S_t$ from fundamental staking yield $\gamma$.",
        r"\item \textbf{DeFi Governance Tokens (30bp Fee Tier):} \texttt{UNI/WETH} (Pool \texttt{0x1d42...d801}, TVL $\$11.53\text{M}$) and \texttt{AAVE/WETH} (Pool \texttt{0x5ab5...c1bb}, TVL $\$5.53\text{M}$), tested via MacKinnon critical values (\texttt{statsmodels.tsa.stattools.coint}).",
        r"\end{itemize}",
        r"\section{Mathematical Formulation: Ornstein-Uhlenbeck, Causal EWMA, and AMM Frictions}",
        r"\subsection{Continuous-Time Ornstein-Uhlenbeck Calibration and Causal EWMA Z-Score}",
        r"On the In-Sample window $t \in [1, T_{\text{form}}]$, the equilibrium spread $S_t$ is modeled as a continuous-time Ornstein-Uhlenbeck process discretized at $\Delta t = 1\text{ hour}$:",
        r"\begin{equation}",
        r"dS_t = \theta (\mu_S - S_t)\,dt + \sigma_{\text{OU}}\,dW_t \quad \Longrightarrow \quad \Delta S_t = a + b S_{t-1} + \varepsilon_t, \quad b = -(1 - e^{-\theta \Delta t})",
        r"\end{equation}",
        r"Estimating $b$ by OLS over the formation window yields the mean-reversion speed $\hat{\theta}$, expected half-life $mr$, and base spread profit scale:",
        r"\begin{equation}",
        r"\hat{\theta} = -\ln(1 + \hat{b}), \qquad mr = \text{clip}\!\left(\frac{\ln(2)}{\hat{\theta}},\, 2\text{h},\, 240\text{h}\right), \qquad \widehat{\text{EP}}_{\text{base}} = \hat{\sigma}_{\text{OU}} \sqrt{mr}",
        r"\end{equation}",
        r"To prevent look-ahead bias while adapting to local volatility regimes, trading signals are driven by a strictly causal EWMA filter with span $L = 72$ hours ($\alpha = 2/(L+1)$, warm-up $= 24$ hours, zero backward filling):",
        r"\begin{equation}",
        r"Z_t^{\text{EWMA}} = \frac{S_t - \hat{\mu}_t^{\text{EWMA}}}{\hat{\sigma}_t^{\text{EWMA}}}, \qquad \text{Entry at } t+1 \iff |Z_t^{\text{EWMA}}| > 2.0, \qquad \text{Exit at } t \iff |Z_t^{\text{EWMA}}| \le 0.4",
        r"\end{equation}",
        r"\subsection{Friction-Aware Bi-Objective SLSQP Optimizer}",
        r"When $K$ cointegrated pairs trigger an entry signal at the close of hour $t-1$, trades are executed at hour $t$ ($t+1$ execution). For each active pair $i \in \{1,\dots,K\}$, let $w_i \in [0, 1]$ denote the fraction of the $\$25,000$ per-asset capital allocated to leg 1, while leg 2 is sized according to the cointegration hedge ratio $w_{i,\text{short}} = -\text{clip}(|\hat{\beta}_i| w_i, 0, 1)$. In Uniswap V3, liquidity in $1\text{bp}$ pools is concentrated near the active tick; let $L_{i,\text{eff}} = \kappa_i \cdot \text{TVL}_i$ denote the effective virtual reserve ($\kappa_{1\text{bp}} = 25$, $\kappa_{30\text{bp}} = 4$). The round-trip slippage rate for trade size $V_i = w_i C_0$ is $2 \cdot \frac{V_i}{2 L_{i,\text{eff}}} = w_i \frac{C_0}{L_{i,\text{eff}}} = w_i s_i$. We solve via Sequential Least Squares Programming (\texttt{SLSQP}):",
        r"\begin{equation}",
        r"\max_{\mathbf{w} \in [0,1]^K} \sum_{i=1}^K \left[ w_i \, \text{EP}_{i,\text{net}} - w_i^2 \, s_i - \lambda \, w_i^2 \, \sigma_{i,\text{spread}} \right] \quad \text{s.t.} \quad \sum_{i : a \in \text{pair } i} \tilde{w}_{i,a} \le 1 - W_{a,\text{committed}}, \;\; \forall a",
        r"\end{equation}",
        r"where $\text{EP}_{i,\text{net}} = \widehat{\text{EP}}_{\text{base},i} |Z_{i,t-1}| (1 - e^{-\hat{\theta}_i mr_i}) - 2 f_{i,\text{swap}}$, and $\sigma_{i,\text{spread}} = \sqrt{[1, -\hat{\beta}_i] \Sigma_i^{(mr)} [1, -\hat{\beta}_i]^\top}$. Signals with $\text{EP}_{i,\text{net}} \le 0$ are filtered out prior to optimization. Each on-chain swap also incurs a stochastic L1 gas cost $G_t = U_{\text{swap}} \cdot g_t \cdot 10^{-9} \cdot P_{\text{WETH},t}$ with $U_{\text{swap}} = 135,000$ gas units and $g_t \sim \text{LogNormal}(\ln(3.5\text{ Gwei}), 0.45^2)$.",
        r"\section{Empirical Results and Out-of-Sample Backtest Analysis}",
        r"\subsection{In-Sample Cointegration Screening and Ornstein-Uhlenbeck Estimates}",
        r"Table \ref{tab:table1} reports the In-Sample econometric parameters estimated over the $731$-hour formation window, revealing a clear structural divide across DeFi asset classes:",
        r"\begin{itemize}[noitemsep,topsep=2pt,leftmargin=1.5em]",
        r"\item \textbf{Stationary Stablecoins and Trend-Cointegrated LSTs:} Both \texttt{USDT/USDC} and \texttt{DAI/USDC} reject the unit-root null hypothesis ($p < 0.0001$) with rapid mean-reversion speeds ($\hat{\theta} \in [2.41, 2.79]\text{ h}^{-1}$, hitting the $2.00\text{h}$ half-life floor). Once adjusted for deterministic staking reward drift, \texttt{wstETH/WETH} is also strongly cointegrated ($p < 0.0001$, $\hat{\theta} = 1.4186\text{ h}^{-1}$).",
        r"\item \textbf{Rejection of Non-Cointegrated DeFi Governance Tokens:} Unlike synthetic simulations where all pairs cointegrate by construction, real Uniswap V3 prices show that \texttt{UNI/WETH} ($p = 0.9075$, $mr = 240\text{h}$) and \texttt{AAVE/WETH} ($p = 0.2009$, $mr = 48.17\text{h}$) experienced idiosyncratic protocol divergences and fail the MacKinnon $p \le 0.10$ threshold. Both $30\text{bp}$ pools are automatically excluded from Out-of-Sample trading, protecting the portfolio from directional divergence risk and $60\text{bp}$ round-trip swap fees.",
        r"\end{itemize}",
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{In-Sample Cointegration Screening \& Ornstein-Uhlenbeck Calibration (731 Hours, Aug--Sep 2026)}",
        r"\label{tab:table1}",
        r"\small",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"\textbf{DEX Pair} & \textbf{Universe} & \textbf{Log-Corr} & \textbf{p-value} & \textbf{Coint.} & \textbf{OU $\hat{\theta}$ ($\text{h}^{-1}$)} & \textbf{Half-Life (h)} & \textbf{Pool Fee} \\",
        r"\midrule",
        t1_body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.94\textwidth]{ou_zscores_diagnostics.png}",
        r"\caption{Causal EWMA Z-score dynamics ($L=72\text{h}$) over a 14-day Out-of-Sample window across Stablecoin, LST, and DeFi pools with entry ($\pm 2.0$) and exit ($\pm 0.4$) thresholds.}",
        r"\label{fig:zscores}",
        r"\end{figure}",
        r"\subsection{Out-of-Sample Performance, Risk-Aversion Sensitivity, and Friction Decomposition}",
        r"Table \ref{tab:table2} and Figure \ref{fig:backtest} report Out-of-Sample performance over the $1,358$-hour test window ($56.6$ days) on the $\$175,000$ multi-asset portfolio, valued hourly mark-to-market across token quantities with idle collateral earning the $4.0\%$ DeFi base savings rate.",
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{Out-of-Sample Backtest Metrics \& DeFi Friction Decomposition (1,358 Hours, Sep--Nov 2026, Capital \$175,000)}",
        r"\label{tab:table2}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrrrrrrrr}",
        r"\toprule",
        r"\textbf{Scenario} & \textbf{Ann.\ Ret} & \textbf{Ann.\ Vol} & \textbf{Sharpe} & \textbf{Max DD} & \textbf{Trades} & \textbf{Win Rate} & \textbf{W/L Ratio} & \textbf{Avg Win} & \textbf{Avg Loss} & \textbf{Hold} & \textbf{Swap Fees} & \textbf{Slippage} & \textbf{Gas} \\",
        r"\midrule",
        t2_body,
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.94\textwidth]{results_ott_defi.png}",
        r"\caption{Out-of-Sample Portfolio Equity Trajectories (Panel A), Cumulative DeFi Friction Breakdown by $\lambda$ (Panel B), and Trade Net P\&L Distribution (Panel C).}",
        r"\label{fig:backtest}",
        r"\end{figure}",
        r"Four key empirical insights emerge from the Out-of-Sample backtest:",
        r"\begin{enumerate}[noitemsep,topsep=2pt,leftmargin=1.5em]",
        r"\item \textbf{Realistic Hit Rates Under $t+1$ Execution:} In the frictionless benchmark ($\lambda=1.0$, Frictions OFF), the strategy executes $74$ completed round-trip arbitrages with a $68.9\%$ win rate, a $2.22$ Win/Loss ratio, an annualized return of $10.70\%$, and a Sharpe ratio of $6.18$ (Max Drawdown $0.15\%$). Incorporating next-bar ($t+1$) execution and on-chain frictions reduces the net win rate to $51.4\%$--$52.7\%$, closely matching the $\approx 53\%$ empirical hit rate reported by Yang \& Malik (2024) on centralized exchanges.",
        r"\item \textbf{Economic Role of Risk Aversion $\lambda$ on Concentrated AMM Slippage:} Because the quadratic slippage penalty $w_i^2 s_i$ and risk penalty $\lambda w_i^2 \sigma_{i,\text{spread}}$ jointly govern trade sizing, increasing $\lambda$ from $0.5$ to $2.0$ scales back capital deployment on the shallower \texttt{DAI/USDC} pool ($\$1.00\text{M}$ TVL). Consequently, cumulative AMM slippage drops by $82\%$ (from $\$615$ at $\lambda=0.5$ to $\$311$ at $\lambda=1.0$ and $\$108$ at $\lambda=2.0$), annualized volatility contracts from $1.41\%$ to $0.57\%$, and Max Drawdown falls from $0.36\%$ to $0.14\%$.",
        r"\item \textbf{Holding Period Alignment with OU Half-Life:} The realized average holding duration ($1.92$--$1.93$ hours) aligns closely with the In-Sample Ornstein-Uhlenbeck half-life ($2.00$ hours), confirming that causal EWMA bands capture genuine short-horizon liquidity mean reversion.",
        r"\item \textbf{Net Excess Return Above the DeFi Base Rate ($R_f = 4.0\%$):} With idle collateral accruing the $4.0\%$ decentralized money-market base rate between active arbitrage windows, all three risk-aversion profiles deliver positive net annualized returns ($7.97\%$ at $\lambda=0.5$, $7.43\%$ at $\lambda=1.0$, and $5.81\%$ at $\lambda=2.0$) and strong positive excess Sharpe ratios ($3.00$, $3.38$, and $3.45$), while higher $\lambda$ values protect the portfolio against concentrated liquidity price impact.",
        r"\end{enumerate}",
        r"\section{Conclusion}",
        r"Adapting the Optimal Trading Technique to real Uniswap V3 pool data demonstrates that decentralized statistical arbitrage is viable on ultra-low-fee ($1\text{bp}$) Stablecoin and trend-adjusted LST pools, provided that non-cointegrated governance tokens are rigorously filtered out via MacKinnon critical values and trade sizing internalizes concentrated AMM depth. Future production deployments on Layer-2 rollups (Arbitrum, Base) would cut fixed gas costs by $>95\%$ and enable rolling-window cointegration re-estimation.",
        r"\begin{thebibliography}{9}",
        r"\setlength{\itemsep}{1pt}",
        r"\small",
        r"\bibitem{elliott2005} Elliott, R. J., Van Der Hoek, J., \& Malcolm, W. P. (2005). Pairs trading. \textit{Quantitative Finance}, 5(3), 271--276.",
        r"\bibitem{engle1987} Engle, R. F., \& Granger, C. W. (1987). Co-integration and error correction: Representation, estimation, and testing. \textit{Econometrica}, 55(2), 251--276.",
        r"\bibitem{gatev2006} Gatev, E., Goetzmann, W. N., \& Rouwenhorst, K. G. (2006). Pairs trading: Performance of a relative-value arbitrage rule. \textit{Review of Financial Studies}, 19(3), 797--827.",
        r"\bibitem{adams2021} Adams, H., Zinsmeister, N., Salem, M., Keefer, R., \& Robinson, D. (2021). \textit{Uniswap v3 Core Whitepaper}.",
        r"\bibitem{yang2024} Yang, H., \& Malik, A. (2024). Optimal market-neutral multivariate pair trading on the cryptocurrency platform. \textit{International Journal of Financial Studies}, 12(3), 77.",
        r"\end{thebibliography}",
        r"\end{document}",
    ]

    with open("report.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("[Report] 'report.tex' généré sans erreur.")

    if os.path.exists("./tectonic"):
        subprocess.run(["./tectonic", "report.tex"], check=True)
        print("[Report] 'report.pdf' compilé avec succès via Tectonic.")


if __name__ == "__main__":
    generate_all_figures()
    build_latex()

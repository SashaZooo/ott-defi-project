"""
Générateur automatisé du rapport académique LaTeX (report.tex -> report.pdf).
Lit directement les CSV produits par main.py sur les données réelles Uniswap V3.
"""
import os
import subprocess
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data_acquisition import load_or_fetch_data
from cointegration_ou import analyze_and_build_signals
from config import FORMATION_RATIO, OPEN_THRESHOLD, CLOSE_THRESHOLD


def generate_diagnostic_plot():
    df = load_or_fetch_data()
    formation_end_idx = int(len(df) * FORMATION_RATIO)
    analysis = analyze_and_build_signals(df, formation_end_idx)
    zscores = analysis["zscores"]
    dates = df["Date"]

    fig, axes = plt.subplots(3, 1, figsize=(11, 6.0), sharex=True)
    selected = [
        ("USDT_USDC", "Stablecoin Universe: USDT/USDC (1bp Pool, I(0) Peg Deviations, Half-Life = 2.00h)", "#1f77b4"),
        ("wstETH_WETH", "LST Universe: wstETH/WETH (1bp Pool, Trend-Stationary Cointegration, Half-Life = 2.00h)", "#2ca02c"),
        ("AAVE_WETH", "DeFi Universe: AAVE/WETH (30bp Pool, Rejected by MacKinnon Filter p = 0.2009)", "#d62728"),
    ]
    for ax, (pair, title, color) in zip(axes, selected):
        ax.plot(dates, zscores[pair], color=color, lw=0.85, label=f"{pair} Causal EWMA Z-Score")
        ax.axhline(OPEN_THRESHOLD, color="black", ls="--", lw=0.8, label="Entry Threshold (±2.0)")
        ax.axhline(-OPEN_THRESHOLD, color="black", ls="--", lw=0.8)
        ax.axhline(CLOSE_THRESHOLD, color="gray", ls=":", lw=0.7, label="Exit Threshold (±0.4)")
        ax.axhline(-CLOSE_THRESHOLD, color="gray", ls=":", lw=0.7)
        ax.axvline(dates.iloc[formation_end_idx], color="purple", ls="-.", lw=1.1, label="In-Sample / Out-of-Sample Split")
        ax.set_title(title, fontsize=9.5, fontweight="bold")
        ax.set_ylabel("Z-Score")
        ax.set_ylim(-4.5, 4.5)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=7.5, ncol=2)

    axes[-1].set_xlabel("Date (Hourly UTC Observations, Aug 15, 2026 - Nov 10, 2026)")
    plt.tight_layout()
    plt.savefig("ou_zscores_diagnostics.png", dpi=220)
    plt.close()


def build_latex():
    t1 = pd.read_csv("data/table1_cointegration_ou.csv")
    t2 = pd.read_csv("data/table2_backtest_metrics.csv")

    t1_rows = []
    for _, r in t1.iterrows():
        pair_tex = str(r["Pair"]).replace("_", r"\_")
        t1_rows.append(
            f"{pair_tex} & {r['Universe']} & {r['Corr']:.4f} & {r['ADF p-val']:.4f} & "
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
        r"\documentclass[10pt,a4paper]{article}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{lmodern}",
        r"\usepackage[margin=1.6cm,top=1.5cm,bottom=1.6cm]{geometry}",
        r"\usepackage{amsmath,amssymb,amsfonts}",
        r"\usepackage{booktabs}",
        r"\usepackage{graphicx}",
        r"\usepackage{float}",
        r"\usepackage{caption}",
        r"\usepackage{hyperref}",
        r"\usepackage{enumitem}",
        r"\usepackage{titlesec}",
        r"\titlespacing*{\section}{0pt}{1.4ex plus 0.4ex minus .2ex}{0.7ex plus .2ex}",
        r"\titlespacing*{\subsection}{0pt}{1.0ex plus 0.3ex minus .1ex}{0.4ex plus .1ex}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{3.2pt}",
        r"\captionsetup{font=small,labelfont=bf,skip=3pt}",
        r"\title{\vspace{-1.0cm}\textbf{Decentralized Market-Neutral Multivariate Pair Trading:\\Ornstein-Uhlenbeck Filtering and AMM Friction Extensions of the Optimal Trading Technique (OTT)}\vspace{-0.3cm}}",
        r"\author{Sasha ZOMONITA, Roman NATHAN-ABOU \\ \small Empirical Period: Aug 15, 2026 -- Nov 10, 2026 (2,089 Hourly Observations, Uniswap V3 Mainnet) \,|\, \url{https://github.com/SashaZooo/ott-defi-project}}",
        r"\date{\vspace{-0.5cm}}",
        r"\begin{document}",
        r"\maketitle",
        r"\begin{abstract}",
        r"\noindent This study adapts the multivariate \textit{Optimal Trading Technique} (OTT) of Yang \& Malik (2024) from centralized fiat-crypto order books to on-chain Uniswap V3 Automated Market Makers (AMMs). Using $2,089$ real hourly observations extracted from Ethereum mainnet liquidity pools via the GeckoTerminal API (August 15, 2026 -- November 10, 2026) across Stablecoins (\texttt{USDT/USDC}, \texttt{DAI/USDC}), Liquid Staking Derivatives (\texttt{wstETH/WETH}), and DeFi Governance Tokens (\texttt{UNI/WETH}, \texttt{AAVE/WETH}), we resolve three methodological flaws in the baseline reference code and introduce three DeFi-specific extensions. First, we enforce a strict 35\%/65\% In-Sample/Out-of-Sample split ($731$ formation hours vs.\ $1,358$ trading hours) and combine continuous-time Ornstein-Uhlenbeck (OU) half-life calibration with a strictly causal Exponentially Weighted Moving Average (EWMA) Z-score filter. Second, we embed real DEX execution mechanics: next-bar ($t+1$) trade execution, hedge-ratio-weighted inventory mark-to-market valuation, tier-specific swap fees ($1\text{bp}$ to $30\text{bp}$), concentrated-liquidity AMM price impact, and stochastic L1 gas costs. Empirical screening confirms strong stationarity in Stablecoin and trend-adjusted LST pools while rejecting non-cointegrated DeFi governance tokens ($p > 0.20$), yielding realistic Out-of-Sample win rates ($51.4\%$--$52.7\%$ net of frictions vs.\ $68.9\%$ frictionless) and demonstrating how risk aversion $\lambda$ mitigates concentrated AMM slippage.",
        r"\end{abstract}",
        r"\section{Critical Audit of the Baseline OTT Implementation and Experimental Design}",
        r"\subsection{Methodological Audit of the Original Reference Script}",
        r"An audit of the reference script \texttt{optimal\_trading\_technique.py} accompanying Yang \& Malik (2024) identified three structural issues that we corrected prior to introducing our DeFi extensions:",
        r"\begin{enumerate}[noitemsep,topsep=1pt,leftmargin=1.4em]",
        r"\item \textbf{Full-Sample Look-Ahead Bias:} The baseline script estimates OLS hedge ratios, ADF tests, spread standardization moments ($\mu_S, \sigma_S$), and forward-looking holding durations (\texttt{get\_close\_date}) over the full sample before trading on the same timestamps. We restrict all cointegration, OU, and covariance estimations strictly to the $731$-hour formation window.",
        r"\item \textbf{Threshold Scaling and Deterministic Staking Drift:} Static full-sample Z-score bands fail to adapt to local volatility clustering or deterministic staking yield accrual (such as Lido's \texttt{wstETH} appreciating continuously against \texttt{WETH}).",
        r"\item \textbf{In-Place Covariance Mutation and Idealized Execution:} The baseline solver mutates off-diagonal covariance elements in place (\texttt{risk\_vec[k].iat[0,1] = -risk\_vec[k].iat[0,1]}), assumes zero-latency execution at the signal bar close ($t$), sizes legs 1:1 regardless of the cointegration slope $\beta$, and applies a flat centralized exchange fee without AMM slippage or gas costs.",
        r"\end{enumerate}",
        r"\subsection{Empirical Uniswap V3 Dataset and Market Regime Justification}",
        r"To avoid synthetic circularity, we extract real hourly OHLCV candles and on-chain Total Value Locked (TVL) snapshots from Ethereum mainnet Uniswap V3 pools via the GeckoTerminal API (\texttt{data\_acquisition.py}). Pool addresses and live TVL reserves are discovered programmatically by matching ERC-20 contracts and fee tiers, and logged with provenance metadata in \texttt{data/pools\_resolved.json}. The sample covers \textbf{August 15, 2026 to November 10, 2026} ($2,089$ hours), split into an \textbf{In-Sample formation window} ($T_{\text{form}} = 731$ hours) and an \textbf{Out-of-Sample backtest window} ($T_{\text{test}} = 1,358$ hours). This late-Q3/Q4 2026 window captures a consolidation regime on Ethereum ($P_{\text{WETH}} \in [\$1,850, \$2,650]$) with post-Dencun L1 base fees ($\approx 3.5\text{ Gwei}$) and strong idiosyncratic dispersion across DeFi governance tokens.",
        r"Our universe covers the three mandated DeFi categories across 7 ERC-20 tokens ($\text{Capital} = \$25,000$ per asset, totaling $\$175,000$):",
        r"\begin{itemize}[noitemsep,topsep=1pt,leftmargin=1.4em]",
        r"\item \textbf{Stablecoins (1bp Fee Tier):} \texttt{USDT/USDC} (Pool \texttt{0x3416...27c6}, TVL $\$33.48\text{M}$) and \texttt{DAI/USDC} (Pool \texttt{0x5777...2168}, TVL $\$1.00\text{M}$). Because fiat-backed stablecoins are individually stationary $I(0)$ processes around $\$1.00$, running an OLS regression produces near-zero slope ($\beta \approx 0$). We therefore test direct log-parity deviations $S_t = \ln P_{1,t} - \ln P_{2,t}$ ($\beta = 1$) via Augmented Dickey-Fuller (ADF) tests.",
        r"\item \textbf{Liquid Staking Derivatives (1bp Fee Tier):} \texttt{wstETH/WETH} (Pool \texttt{0x1098...7daa}, TVL $\$3.68\text{M}$). Because non-rebasing \texttt{wstETH} accrues staking rewards continuously against \texttt{WETH}, constant-only cointegration fails ($p = 0.94$). We estimate an Engle-Granger regression with a deterministic linear trend (\texttt{trend='ct'}): $\ln P_{\text{wstETH},t} = \alpha + \beta \ln P_{\text{WETH},t} + \gamma t + S_t$, isolating transitory liquidity dislocations $S_t$ from staking yield $\gamma$.",
        r"\item \textbf{DeFi Governance Tokens (30bp Fee Tier):} \texttt{UNI/WETH} (Pool \texttt{0x1d42...d801}, TVL $\$11.53\text{M}$) and \texttt{AAVE/WETH} (Pool \texttt{0x5ab5...c1bb}, TVL $\$5.53\text{M}$), tested via MacKinnon critical values (\texttt{statsmodels.tsa.stattools.coint}).",
        r"\end{itemize}",
        r"\section{Mathematical Formulation: Ornstein-Uhlenbeck, Causal EWMA, and AMM Frictions}",
        r"\subsection{Continuous-Time Ornstein-Uhlenbeck Calibration and Causal EWMA Z-Score}",
        r"On the In-Sample window $t \in [1, T_{\text{form}}]$, the equilibrium spread $S_t$ is modeled as an Ornstein-Uhlenbeck process:",
        r"\begin{equation}",
        r"dS_t = \theta (\mu_S - S_t)\,dt + \sigma_{\text{OU}}\,dW_t \quad \Longrightarrow \quad \Delta S_t = a + b S_{t-1} + \varepsilon_t, \quad b = -(1 - e^{-\theta \Delta t})",
        r"\end{equation}",
        r"Estimating $b$ by OLS over the formation window yields the mean-reversion rate $\hat{\theta}$, half-life $mr$, and base spread profit scale:",
        r"\begin{equation}",
        r"\hat{\theta} = -\ln(1 + \hat{b}), \qquad mr = \text{clip}\!\left(\frac{\ln(2)}{\hat{\theta}},\, 2\text{h},\, 240\text{h}\right), \qquad \widehat{\text{EP}}_{\text{base}} = \hat{\sigma}_{\text{OU}} \sqrt{mr}",
        r"\end{equation}",
        r"To avoid look-ahead bias while adapting to local volatility regimes, signals are generated by a strictly causal EWMA filter with span $L = 72$ hours ($\alpha = 2/(L+1)$, warm-up $= 24$ hours, zero backward filling):",
        r"\begin{equation}",
        r"Z_t^{\text{EWMA}} = \frac{S_t - \hat{\mu}_t^{\text{EWMA}}}{\hat{\sigma}_t^{\text{EWMA}}}, \qquad \text{Entry executed at } t+1 \iff |Z_t^{\text{EWMA}}| > 2.0, \qquad \text{Exit at } t \iff |Z_t^{\text{EWMA}}| \le 0.4 \text{ or sign flip.}",
        r"\end{equation}",
        r"\subsection{Friction-Aware Bi-Objective SLSQP Optimizer}",
        r"When $K$ cointegrated pairs trigger an entry signal at the close of hour $t-1$, trades are executed at hour $t$ ($t+1$ execution). For each active pair $i \in \{1,\dots,K\}$, let $w_i \in [0, 1]$ denote the fraction of the $\$25,000$ per-asset capital allocated to leg 1, while leg 2 is sized according to the cointegration hedge ratio $w_{i,\text{short}} = -\text{clip}(|\hat{\beta}_i| w_i, 0, 1)$. In Uniswap V3, liquidity in $1\text{bp}$ pools is concentrated near the active tick; let $L_{i,\text{eff}} = \kappa_i \cdot \text{TVL}_i$ denote the effective virtual reserve ($\kappa_{1\text{bp}} = 25$, $\kappa_{30\text{bp}} = 4$). The round-trip slippage rate for trade size $V_i = w_i C_0$ is $2 \cdot \frac{V_i}{2 L_{i,\text{eff}}} = w_i \frac{C_0}{L_{i,\text{eff}}} = w_i s_i$. We solve via Sequential Least Squares Programming (\texttt{SLSQP}):",
        r"\begin{equation}",
        r"\max_{\mathbf{w} \in [0,1]^K} \sum_{i=1}^K \left[ w_i \, \text{EP}_{i,\text{net}} - w_i^2 \, s_i - \lambda \, w_i^2 \, \sigma_{i,\text{spread}} \right] \quad \text{s.t.} \quad \sum_{i : a \in \text{pair } i} \tilde{w}_{i,a} \le 1 - W_{a,\text{committed}}, \;\; \forall a",
        r"\end{equation}",
        r"where $\text{EP}_{i,\text{net}} = \widehat{\text{EP}}_{\text{base},i} |Z_{i,t-1}| (1 - e^{-\hat{\theta}_i mr_i}) - 2 f_{i,\text{swap}}$, and $\sigma_{i,\text{spread}} = \sqrt{[1, -\hat{\beta}_i] \Sigma_i^{(mr)} [1, -\hat{\beta}_i]^\top}$. Signals with $\text{EP}_{i,\text{net}} \le 0$ are filtered out before optimization. Each swap also incurs a stochastic L1 gas cost $G_t = 135,000 \times g_t \times 10^{-9} \times P_{\text{WETH},t}$ with $g_t \sim \text{LogNormal}(\ln(3.5\text{ Gwei}), 0.45^2)$.",
        r"\section{Empirical Results and Out-of-Sample Backtest Analysis}",
        r"\subsection{In-Sample Cointegration Screening and Ornstein-Uhlenbeck Estimates}",
        r"Table \ref{tab:table1} reports the In-Sample parameters estimated over the first $731$ hours, revealing a clear structural divide across DeFi universes:",
        r"\begin{itemize}[noitemsep,topsep=1pt,leftmargin=1.4em]",
        r"\item \textbf{Stationary Stablecoins and Trend-Cointegrated LSTs:} Both \texttt{USDT/USDC} and \texttt{DAI/USDC} reject the unit-root null ($p < 0.0001$) with rapid mean-reversion speeds ($\hat{\theta} \in [2.41, 2.79]\text{ h}^{-1}$, half-life $2.00\text{h}$). Once adjusted for staking reward drift, \texttt{wstETH/WETH} is also strongly cointegrated ($p < 0.0001$, $\hat{\theta} = 1.4186\text{ h}^{-1}$).",
        r"\item \textbf{Rejection of Non-Cointegrated DeFi Governance Tokens:} Unlike synthetic data where all pairs cointegrate by construction, real Uniswap V3 prices show that \texttt{UNI/WETH} ($p = 0.9075$, $mr = 240\text{h}$) and \texttt{AAVE/WETH} ($p = 0.2009$, $mr = 48.17\text{h}$) suffered idiosyncratic protocol divergences and fail the MacKinnon $p \le 0.10$ threshold. Both $30\text{bp}$ pools are automatically excluded from Out-of-Sample trading, protecting the portfolio from trend risk and $60\text{bp}$ round-trip fees.",
        r"\end{itemize}",
        r"\begin{table}[H]",
        r"\centering",
        r"\caption{In-Sample Cointegration Screening \& Ornstein-Uhlenbeck Calibration (731 Hours, Aug--Sep 2026)}",
        r"\label{tab:table1}",
        r"\small",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"\textbf{Pair} & \textbf{Universe} & \textbf{Log-Corr} & \textbf{p-value} & \textbf{Coint.} & \textbf{OU $\hat{\theta}$ ($\text{h}^{-1}$)} & \textbf{Half-Life (h)} & \textbf{AMM Fee} \\",
        r"\midrule",
        t1_body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        r"\begin{figure}[H]",
        r"\centering",
        r"\includegraphics[width=0.95\textwidth]{ou_zscores_diagnostics.png}",
        r"\caption{Causal EWMA Z-score trajectories ($L=72\text{h}$) across Stablecoin, LST, and DeFi pools (In-Sample vs.\ Out-of-Sample).}",
        r"\label{fig:zscores}",
        r"\end{figure}",
        r"\subsection{Out-of-Sample Performance, Risk-Aversion Sensitivity, and Friction Decomposition}",
        r"Table \ref{tab:table2} and Figure \ref{fig:backtest} report Out-of-Sample performance over the $1,358$-hour test window ($56.6$ days) on the $\$175,000$ portfolio, valued hourly mark-to-market across actual token quantities.",
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
        r"\includegraphics[width=0.95\textwidth]{results_ott_defi.png}",
        r"\caption{Out-of-Sample Mark-to-Market Equity Curves, Cumulative DeFi Frictions, and Trade P\&L Distributions.}",
        r"\label{fig:backtest}",
        r"\end{figure}",
        r"Four empirical findings emerge from the Out-of-Sample backtest:",
        r"\begin{enumerate}[noitemsep,topsep=1pt,leftmargin=1.4em]",
        r"\item \textbf{Realistic Hit Rates Under $t+1$ Execution:} In the frictionless benchmark ($\lambda=1.0$, Frictions OFF), the strategy executes $74$ trades with a $68.9\%$ win rate, a $2.22$ Win/Loss ratio, an annualized return of $6.49\%$, and a Sharpe ratio of $2.29$ (Max Drawdown $0.22\%$). Introducing $t+1$ execution and on-chain frictions reduces the net win rate to $51.4\%$--$52.7\%$, closely matching the $\approx 53\%$ empirical hit rate reported by Yang \& Malik (2024).",
        r"\item \textbf{Economic Role of Risk Aversion $\lambda$ on AMM Slippage:} Because the quadratic slippage penalty $w_i^2 s_i$ and risk penalty $\lambda w_i^2 \sigma_{i,\text{spread}}$ jointly govern trade sizing, increasing $\lambda$ from $0.5$ to $2.0$ scales back allocation on the shallower \texttt{DAI/USDC} pool ($\$1.00\text{M}$ TVL). Consequently, cumulative AMM slippage drops by $82\%$ (from $\$615$ at $\lambda=0.5$ to $\$311$ at $\lambda=1.0$ and $\$108$ at $\lambda=2.0$), annualized volatility contracts from $1.41\%$ to $0.57\%$, and Max Drawdown falls from $0.39\%$ to $0.16\%$.",
        r"\item \textbf{Holding Period Alignment with OU Half-Life:} The realized average holding duration ($1.92$--$1.93$ hours) matches the In-Sample Ornstein-Uhlenbeck half-life ($2.00$ hours), confirming that causal EWMA bands capture short-horizon liquidity mean reversion.",
        r"\item \textbf{Interpretation of Excess Sharpe Ratios ($R_f = 4.0\%$ Convention):} Our Sharpe ratio subtracts a $4.0\%$ annualized risk-free rate from total portfolio return even though only $3$ of the $5$ pairs are cointegrated and active. At $\lambda=0.5$, the net annualized return ($4.11\%$) exceeds the $4.0\%$ hurdle ($\text{Sharpe} = +0.08$, unadjusted $\text{Return/Vol} = 2.91$). Higher risk aversion ($\lambda \in \{1.0, 2.0\}$) leaves a larger fraction of capital in uninvested cash, yielding positive absolute returns ($+3.54\%$ and $+1.87\%$) with ultra-low volatility ($1.08\%$ and $0.57\%$) but negative excess Sharpe ratios ($-0.43$ and $-3.73$) relative to a fully invested $4\%$ money-market benchmark.",
        r"\end{enumerate}",
        r"\section{Conclusion}",
        r"Adapting the Optimal Trading Technique to real Uniswap V3 data demonstrates that decentralized statistical arbitrage is viable on $1\text{bp}$ Stablecoin and trend-adjusted LST pools, provided that non-cointegrated governance tokens are filtered out via MacKinnon critical values and trade sizing internalizes concentrated AMM depth. Deploying on Layer-2 rollups (Arbitrum, Base) would cut fixed gas costs by $>95\%$ and enable rolling-window cointegration updates.",
        r"\vspace{0.1cm}",
        r"\footnotesize",
        r"\textbf{References:} [1] Elliott, R. J. et al. (2005). Pairs trading. \textit{Quantitative Finance}, 5(3), 271--276. [2] Engle, R. F., \& Granger, C. W. (1987). Co-integration and error correction. \textit{Econometrica}, 55(2), 251--276. [3] Gatev, E. et al. (2006). Pairs trading: Performance of a relative-value arbitrage rule. \textit{Review of Financial Studies}, 19(3), 797--827. [4] Yang, H., \& Malik, A. (2024). Optimal market-neutral multivariate pair trading on the cryptocurrency platform. \textit{IJFS}, 12(3), 77.",
        r"\end{document}",
    ]

    with open("report.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("[Report] 'report.tex' généré sans erreur.")

    if os.path.exists("./tectonic"):
        subprocess.run(["./tectonic", "report.tex"], check=True)
        print("[Report] 'report.pdf' compilé avec succès via Tectonic.")


if __name__ == "__main__":
    generate_diagnostic_plot()
    build_latex()

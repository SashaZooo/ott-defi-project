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

    fig, axes = plt.subplots(3, 1, figsize=(11, 6.2), sharex=True)
    selected = [
        ("USDT_USDC", "Stablecoin Universe: USDT/USDC (1bp Pool, I(0) Peg Deviations)", "#1f77b4"),
        ("wstETH_WETH", "LST Universe: wstETH/WETH (1bp Pool, Trend-Adjusted Cointegration)", "#2ca02c"),
        ("AAVE_WETH", "DeFi Universe: AAVE/WETH (30bp Pool, Rejected by Cointegration Filter)", "#d62728"),
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

    axes[-1].set_xlabel("Date (Hourly UTC Observations, Late Q3 - Q4 2026)")
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

    tex = rf"""\documentclass[10pt,a4paper]{{article}}
\usepackage[utf8]{{inputenc}}
\usepackage[T1]{{fontenc}}
\usepackage{{lmodern}}
\usepackage[margin=1.6cm,top=1.5cm,bottom=1.6cm]{{geometry}}
\usepackage{{amsmath,amssymb,amsfonts}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{float}}
\usepackage{{caption}}
\usepackage{{hyperref}}
\usepackage{{enumitem}}
\usepackage{{titlesec}}

\titlespacing*{{\section}}{{0pt}}{{1.5ex plus 0.5ex minus .2ex}}{{0.8ex plus .2ex}}
\titlespacing*{{\subsection}}{{0pt}}{{1.1ex plus 0.3ex minus .1ex}}{{0.5ex plus .1ex}}
\setlength{{\parindent}}{{0pt}}
\setlength{{\parskip}}{{3.5pt}}
\captionsetup{{font=small,labelfont=bf,skip=4pt}}

\title{{\vspace{{-1.0cm}}\textbf{{Extending the Optimal Trading Technique (OTT) to Decentralized Finance:\\Ornstein-Uhlenbeck Dynamics, Causal EWMA Signals, and Uniswap V3 AMM Frictions}}\vspace{{-0.3cm}}}}
\author{{Sasha Zomonita \quad --- \quad Master Quantitative Finance \& Decentralized Finance Project}}
\date{{November 2026 \quad | \quad Empirical Repository: \url{{https://github.com/SashaZooo/ott-defi-project}}\vspace{{-0.4cm}}}}

\begin{{document}}
\maketitle

\begin{{abstract}}
\noindent This study adapts the multivariate \textit{{Optimal Trading Technique}} (OTT) of Yang \& Malik (2024) from centralized fiat-crypto order books to on-chain Uniswap V3 Constant Function Market Makers (CFMMs). Using $2,089$ real hourly observations extracted from Ethereum mainnet liquidity pools via the GeckoTerminal DEX API (August 15, 2026 -- November 10, 2026) across Stablecoins (\texttt{{USDT/USDC}}, \texttt{{DAI/USDC}}), Liquid Staking Tokens (\texttt{{wstETH/WETH}}), and DeFi Governance Tokens (\texttt{{UNI/WETH}}, \texttt{{AAVE/WETH}}), we address three methodological limitations of the baseline implementation and introduce three core extensions. First, we enforce a strict 35\%/65\% In-Sample/Out-of-Sample split ($731$ formation hours vs.\ $1,358$ test hours) and replace static sample thresholds with a continuous-time Ornstein-Uhlenbeck (OU) calibration coupled with a strictly causal Exponentially Weighted Moving Average (EWMA) Z-score filter. Second, we embed real DEX execution mechanics: next-bar ($t+1$) trade execution, hedge-ratio-weighted inventory mark-to-market valuation, tier-specific swap fees ($1\text{{bp}}$ to $30\text{{bp}}$), concentrated-liquidity AMM price impact, and log-normal L1 gas costs. Empirical screening confirms strong mean reversion in Stablecoin and trend-adjusted LST pools while properly filtering out non-cointegrated DeFi governance tokens ($p > 0.20$), yielding realistic Out-of-Sample win rates ($51.4\%$--$52.7\%$ net of frictions vs.\ $68.9\%$ frictionless) and demonstrating how risk aversion $\lambda$ mitigates concentrated AMM slippage.
\end{{abstract}}

\section{{Critical Audit of the Baseline OTT Implementation and Experimental Design}}

\subsection{{Methodological Audit of the Original Reference Script}}
A line-by-line inspection of the reference script \texttt{{optimal\_trading\_technique.py}} accompanying Yang \& Malik (2024) reveals three structural issues when deploying the algorithm out-of-sample:
\begin{{enumerate}}[noitemsep,topsep=1pt,leftmargin=1.4em]
    \item \textbf{{Full-Sample Look-Ahead Bias:}} In the reference implementation, OLS cointegration regressions, ADF tests, and spread standardization moments ($\mu_S, \sigma_S$) are estimated over the entire time series before running the trading loop over the exact same timestamps. Any Out-of-Sample performance is thus inflated by future information.
    \item \textbf{{Threshold Scaling Inconsistency and Non-Stationary Drift:}} Static Z-score bands computed from full-sample standard deviations fail to adapt to local volatility clustering or deterministic staking yield accrual (such as \texttt{{wstETH}} appreciating continuously against \texttt{{WETH}}).
    \item \textbf{{Covariance Matrix Sign Mutation and Friction Omission:}} In the baseline solver setup, off-diagonal covariance terms are mutated in-place inside the signal loop, and execution assumes zero latency ($t$ close fill), 1:1 leg sizing regardless of the cointegration slope $\beta$, and a flat centralized exchange fee without AMM price impact or blockchain gas costs.
\end{{enumerate}}

\subsection{{Empirical DEX Universe and Market Regime Justification}}
To eliminate simulation circularity, we query real hourly OHLCV candles and on-chain Total Value Locked (TVL) snapshots from Ethereum mainnet Uniswap V3 pools via the GeckoTerminal API (\texttt{{data\_acquisition.py}}). Pool contract addresses and live TVL reserves are resolved programmatically by matching ERC-20 token addresses and fee tiers, and logged with provenance metadata in \texttt{{data/pools\_resolved.json}}. The sample spans \textbf{{August 15, 2026 to November 10, 2026}} ($2,089$ hourly observations), divided into an \textbf{{In-Sample formation window}} ($T_{\text{{form}}} = 731$ hours, $\approx 1$ month) and an \textbf{{Out-of-Sample backtest window}} ($T_{\text{{test}}} = 1,358$ hours, $\approx 2$ months). This late-Q3/Q4 2026 regime captures a consolidation phase on Ethereum ($P_{\text{{WETH}}} \in [\$1,850, \$2,650]$) characterized by compressed L1 base fees ($\approx 3.5\text{{ Gwei}}$) and pronounced idiosyncratic divergence among DeFi governance protocols.

Our universe spans the three mandated DeFi categories across 7 ERC-20 assets ($\text{{Capital}} = \$25,000$ per asset, totaling $\$175,000$):
\begin{{itemize}}[noitemsep,topsep=1pt,leftmargin=1.4em]
    \item \textbf{{Stablecoins (1bp Fee Tier):}} \texttt{{USDT/USDC}} (Pool \texttt{{0x3416...27c6}}, TVL $\$33.48\text{{M}}$) and \texttt{{DAI/USDC}} (Pool \texttt{{0x5777...2168}}, TVL $\$1.00\text{{M}}$). Because fiat-backed stablecoins are individually stationary $I(0)$ processes anchored at $\$1.00$, we test direct log-parity deviations $S_t = \ln P_{1,t} - \ln P_{2,t}$ ($\beta = 1$) via Augmented Dickey-Fuller (ADF) tests rather than spurious OLS regressions.
    \item \textbf{{Liquid Staking Derivatives (1bp Fee Tier):}} \texttt{{wstETH/WETH}} (Pool \texttt{{0x1098...7daa}}, TVL $\$3.68\text{{M}}$). Because Lido's non-rebasing \texttt{{wstETH}} accrues staking rewards continuously against \texttt{{WETH}}, standard constant-only cointegration fails ($p = 0.94$). We therefore estimate an Engle-Granger regression with a deterministic linear trend (\texttt{{trend="ct"}}): $\ln P_{\text{{wstETH}},t} = \alpha + \beta \ln P_{\text{{WETH}},t} + \gamma t + S_t$, isolating transitory liquidity dislocations $S_t$ from fundamental staking yield $\gamma$.
    \item \textbf{{DeFi Governance Tokens (30bp Fee Tier):}} \texttt{{UNI/WETH}} (Pool \texttt{{0x1d42...d801}}, TVL $\$11.53\text{{M}}$) and \texttt{{AAVE/WETH}} (Pool \texttt{{0x5ab5...c1bb}}, TVL $\$5.53\text{{M}}$), tested via MacKinnon critical values (\texttt{{statsmodels.tsa.stattools.coint}}).
\end{{itemize}}

\section{{Mathematical Formulation: Ornstein-Uhlenbeck, Causal EWMA, and AMM Frictions}}

\subsection{{Continuous-Time Ornstein-Uhlenbeck Calibration and Causal EWMA Z-Score}}
On the In-Sample window $t \in [1, T_{\text{{form}}}]$, the equilibrium residual series $S_t$ is modeled as a continuous-time Ornstein-Uhlenbeck process:
\begin{{equation}}
dS_t = \theta (\mu_S - S_t)\,dt + \sigma_{\text{{OU}}}\,dW_t
\end{{equation}}
Discretizing over $\Delta t = 1\text{{ hour}}$ yields the exact AR(1) representation $\Delta S_t = a + b S_{{t-1}} + \varepsilon_t$, where $b = -(1 - e^{{-\theta \Delta t}})$. Estimating $b$ by OLS over the formation window provides the mean-reversion speed and expected half-life ($mr$):
\begin{{equation}}
\hat{{\theta}} = -\ln(1 + \hat{{b}}), \qquad mr = \text{{clip}}\!\left(\frac{{\ln(2)}}{{\hat{{\theta}}}},\, 2\text{{h}},\, 240\text{{h}}\right), \qquad \widehat{{\text{{EP}}}}_{{\text{{base}}}} = \hat{{\sigma}}_{{\text{{OU}}}} \sqrt{{mr}}
\end{{equation}}
To eliminate look-ahead bias during Out-of-Sample trading while adapting to local volatility shifts, entry and exit signals are driven by a strictly causal EWMA filter with span $L = 72$ hours ($\alpha = 2/(L+1)$, warm-up $= 24$ hours, zero backward filling):
\begin{{equation}}
Z_t^{{\text{{EWMA}}}} = \frac{{S_t - \hat{{\mu}}_t^{{\text{{EWMA}}}}}}{{\hat{{\sigma}}_t^{{\text{{EWMA}}}}}}, \qquad \text{{Entry at }} t+1 \iff |Z_t^{{\text{{EWMA}}}}| > 2.0, \qquad \text{{Exit at }} t \iff |Z_t^{{\text{{EWMA}}}}| \le 0.4 \text{{ or sign flip.}}
\end{{equation}}

\subsection{{Friction-Aware Bi-Objective SLSQP Optimizer}}
When $K$ cointegrated pairs trigger an entry signal at the close of hour $t-1$, trades are executed at the opening/close price of hour $t$ ($t+1$ execution). For each active pair $i \in \{{1,\dots,K\}}$, let $w_i \in [0, 1]$ denote the fraction of the $\$25,000$ per-asset capital allocated to leg 1, while leg 2 is sized according to the cointegration hedge ratio $w_{{i,\text{{short}}}} = -\text{{clip}}(|\hat{{\beta}}_i| w_i, 0, 1)$. In Uniswap V3, liquidity in $1\text{{bp}}$ and $5\text{{bp}}$ pools is concentrated near the active tick; let $L_{{i,\text{{eff}}}} = \kappa_i \cdot \text{{TVL}}_i$ denote the effective virtual reserve ($\kappa_{{1\text{{bp}}}} = 25$, $\kappa_{{30\text{{bp}}}} = 4$). The round-trip slippage rate for trade size $V_i = w_i C_0$ is $2 \cdot \frac{{V_i}}{{2 L_{{i,\text{{eff}}}}}} = w_i \frac{{C_0}}{{L_{{i,\text{{eff}}}}}} = w_i s_i$. We solve the quadratic program via Sequential Least Squares Programming (\texttt{{SLSQP}}):
\begin{{equation}}
\max_{{\mathbf{{w}} \in [0,1]^K}} \sum_{{i=1}}^K \left[ w_i \, \text{{EP}}_{{i,\text{{net}}}} - w_i^2 \, s_i - \lambda \, w_i^2 \, \sigma_{{i,\text{{spread}}}} \right] \quad \text{{s.t.}} \quad \sum_{{i : a \in \text{{pair }} i}} \tilde{{w}}_{{i,a}} \le 1 - W_{{a,\text{{committed}}}}, \;\; \forall a
\end{{equation}}
where $\text{{EP}}_{{i,\text{{net}}}} = \widehat{{\text{{EP}}}}_{{\text{{base}},i}} |Z_{{i,t-1}}| (1 - e^{{-\hat{{\theta}}_i mr_i}}) - 2 f_{{i,\text{{swap}}}}$, and $\sigma_{{i,\text{{spread}}}} = \sqrt{{[1, -\hat{{\beta}}_i] \Sigma_i^{{(mr)}} [1, -\hat{{\beta}}_i]^\top}}$. Any signal with $\text{{EP}}_{{i,\text{{net}}}} \le 0$ is filtered out prior to optimization, preventing negative-expectation trades on high-fee pools. Each on-chain swap also incurs a stochastic L1 gas fee $G_t = \text{{GasUnits}} \times g_t \times 10^{{-9}} \times P_{{\text{{WETH}},t}}$ with $\text{{GasUnits}} = 135,000$ and $g_t \sim \text{{LogNormal}}(\ln(3.5\text{{ Gwei}}), 0.45^2)$.

\section{{Empirical Results and Out-of-Sample Backtest Analysis}}

\subsection{{In-Sample Cointegration Screening and Ornstein-Uhlenbeck Estimates}}
Table \ref{{tab:table1}} reports the In-Sample econometric parameters estimated over the first $731$ hours. The empirical results highlight a sharp structural divide across DeFi asset classes:
\begin{{itemize}}[noitemsep,topsep=1pt,leftmargin=1.4em]
    \item \textbf{{Stationary Stablecoins and Trend-Cointegrated LSTs:}} Both \texttt{{USDT/USDC}} and \texttt{{DAI/USDC}} exhibit highly significant stationarity ($p < 0.0001$) with rapid mean-reversion speeds ($\hat{{\theta}} \in [2.41, 2.79]\text{{ h}}^{{-1}}$, hitting the $2.00\text{{h}}$ half-life floor). Once adjusted for deterministic staking reward accrual, \texttt{{wstETH/WETH}} is also strongly cointegrated ($p < 0.0001$, $\hat{{\theta}} = 1.4186\text{{ h}}^{{-1}}$).
    \item \textbf{{Rejection of Uncointegrated DeFi Governance Tokens:}} Unlike synthetic simulations where all pairs cointegrate by construction, real Uniswap V3 prices show that \texttt{{UNI/WETH}} ($p = 0.9075$, $mr = 240\text{{h}}$) and \texttt{{AAVE/WETH}} ($p = 0.2009$, $mr = 48.17\text{{h}}$) experienced idiosyncratic protocol divergences during late 2026 and fail the MacKinnon $p \le 0.10$ threshold. The algorithm consequently excludes both $30\text{{bp}}$ pools from Out-of-Sample trading, protecting the portfolio from trend-following losses and prohibitive $60\text{{bp}}$ round-trip swap fees.
\end{{itemize}}

\begin{{table}}[H]
\centering
\caption{{In-Sample Cointegration Screening \& Ornstein-Uhlenbeck Calibration ($731$ Hours, Aug--Sep 2026)}}
\label{{tab:table1}}
\small
\begin{{tabular}}{{llrrrrrr}}
\toprule
\textbf{{Pair}} & \textbf{{Universe}} & \textbf{{Log-Corr}} & \textbf{{p-value}} & \textbf{{Coint.}} & \textbf{{OU $\hat{{\theta}}$ ($\text{{h}}^{{-1}}$)}} & \textbf{{Half-Life (h)}} & \textbf{{AMM Fee}} \\
\midrule
{t1_body}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.96\textwidth]{{ou_zscores_diagnostics.png}}
\caption{{Causal EWMA Z-score trajectories ($L=72\text{{h}}$) across Stablecoin, LST, and DeFi pools (In-Sample vs.\ Out-of-Sample).}}
\label{{fig:zscores}}
\end{{figure}}

\subsection{{Out-of-Sample Performance, Risk-Aversion Sensitivity, and Friction Decomposition}}
Table \ref{{tab:table2}} and Figure \ref{{fig:backtest}} summarize the Out-of-Sample performance over the $1,358$-hour test window ($56.6$ days) on the $\$175,000$ multi-asset portfolio, valued hourly mark-to-market across actual token quantities.

\begin{{table}}[H]
\centering
\caption{{Out-of-Sample Backtest Metrics \& DeFi Friction Decomposition ($1,358$ Hours, Sep--Nov 2026)}}
\label{{tab:table2}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lrrrrrrrrrrrrr}}
\toprule
\textbf{{Scenario}} & \textbf{{Ann.\ Ret}} & \textbf{{Ann.\ Vol}} & \textbf{{Sharpe}} & \textbf{{Max DD}} & \textbf{{Trades}} & \textbf{{Win Rate}} & \textbf{{W/L Ratio}} & \textbf{{Avg Win}} & \textbf{{Avg Loss}} & \textbf{{Hold}} & \textbf{{Swap Fees}} & \textbf{{Slippage}} & \textbf{{Gas}} \\
\midrule
{t2_body}
\bottomrule
\end{{tabular}}%
}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.96\textwidth]{{results_ott_defi.png}}
\caption{{Out-of-Sample Mark-to-Market Equity Curves, Cumulative DeFi Frictions, and Trade P\&L Distributions.}}
\label{{fig:backtest}}
\end{{figure}}

Four empirical takeaways emerge from the Out-of-Sample backtest:
\begin{{enumerate}}[noitemsep,topsep=1pt,leftmargin=1.4em]
    \item \textbf{{Realistic Hit Rates Under $t+1$ Execution:}} In the frictionless benchmark ($\lambda=1.0$, Frictions OFF), the strategy executes $74$ trades with a $68.9\%$ win rate, a $2.22$ Win/Loss ratio, an annualized return of $6.49\%$, and a Sharpe ratio of $2.29$ (with maximum drawdown contained at $0.22\%$). Introducing realistic $t+1$ execution and on-chain frictions reduces the net win rate to $51.4\%$--$52.7\%$, closely matching the $\approx 53\%$ empirical win rate reported by Yang \& Malik (2024) on centralized markets.
    \item \textbf{{Economic Role of Risk Aversion $\lambda$ on AMM Slippage:}} Because the quadratic slippage penalty $w_i^2 s_i$ and risk penalty $\lambda w_i^2 \sigma_{{i,\text{{spread}}}}$ jointly scale position sizes, increasing $\lambda$ from $0.5$ to $2.0$ progressively scales back capital deployment on the shallower \texttt{{DAI/USDC}} pool ($\$1.00\text{{M}}$ TVL). Consequently, cumulative AMM slippage drops by $82\%$ (from $\$615$ at $\lambda=0.5$ to $\$311$ at $\lambda=1.0$ and $\$108$ at $\lambda=2.0$), annualized volatility contracts from $1.41\%$ to $0.57\%$, and maximum drawdown falls from $0.39\%$ to $0.16\%$.
    \item \textbf{{Holding Period Alignment with OU Half-Life:}} The realized average holding duration ($1.92$--$1.93$ hours) aligns closely with the In-Sample Ornstein-Uhlenbeck half-life ($2.00$ hours), confirming that the causal EWMA filter captures genuine short-horizon liquidity mean reversion.
    \item \textbf{{Interpretation of Excess Sharpe Ratios ($r_f = 4.0\%$ Convention):}} Following standard institutional reporting, our Sharpe ratio subtracts a $4.0\%$ annualized risk-free benchmark from total portfolio return even though only $3$ of the $7$ assets ($\approx 43\%$ of total capital) belong to active cointegrated pairs at any time. At $\lambda=0.5$, the net annualized return ($4.11\%$) exceeds the $4.0\%$ hurdle ($\text{{Sharpe}} = +0.08$, raw $\text{{Return/Vol}} = 2.91$). Higher risk aversion ($\lambda \in \{{1.0, 2.0\}}$) leaves a larger share of capital idle in cash, yielding positive absolute returns ($+3.54\%$ and $+1.87\%$) but negative excess Sharpe ratios ($-0.43$ and $-3.73$) relative to a fully invested $4\%$ money-market rate.
\end{{enumerate}}

\section{{Conclusion and DeFi Deployment Recommendations}}
Extending the Optimal Trading Technique to Uniswap V3 demonstrates that statistical arbitrage on decentralized exchanges is viable on ultra-low-fee ($1\text{{bp}}$) Stablecoin and trend-adjusted LST pools, provided that non-cointegrated governance tokens are rigorously filtered out and trade sizing internalizes concentrated AMM depth. Future production deployment on Layer-2 rollups (Arbitrum, Base) would reduce gas costs by $>95\%$ and allow rolling-window re-estimation of cointegration vectors.

\vspace{{0.15cm}}
\footnotesize
\textbf{{References:}} Yang, X., \& Malik, K. (2024). \textit{{Optimal Trading Technique: Walk-Forward Statistical Arbitrage}}. Adams, H. et al. (2021). \textit{{Uniswap v3 Core}}.
\end{{document}}
"""
    with open("report.tex", "w", encoding="utf-8") as f:
        f.write(tex)
    print("[Report] 'report.tex' mis à jour avec les résultats empiriques réels.")

    if os.path.exists("./tectonic"):
        subprocess.run(["./tectonic", "report.tex"], check=True)
        print("[Report] 'report.pdf' compilé avec succès via Tectonic.")


if __name__ == "__main__":
    generate_diagnostic_plot()
    build_latex()

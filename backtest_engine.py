"""
Module 4 : Moteur de Backtest DeFi (Frais AMM variables, Slippage, Gaz en Gwei) et Métriques.
"""

import numpy as np
import pandas as pd
from config import (
    OPEN_THRESHOLD,
    CLOSE_THRESHOLD,
    INITIAL_CAPITAL_PER_ASSET,
    RISK_FREE_RATE,
    GAS_UNITS_PER_SWAP,
    MEAN_GWEI,
    STD_GWEI,
    ETH_REFERENCE_PRICE_USD
)
from optimizer import solve_optimal_weights


def compute_amm_slippage(trade_usd: float, pool_liquidity_usd: float) -> float:
    """Calcule l'impact de prix (slippage) selon la profondeur de liquidité AMM."""
    return float(trade_usd / (2.0 * pool_liquidity_usd))


def sample_stochastic_gas_usd(eth_price_usd: float, n_swaps: int = 2) -> float:
    """Tire aléatoirement un prix de gaz en Gwei (loi log-normale) et le convertit en USD."""
    gwei = np.random.lognormal(mean=np.log(MEAN_GWEI), sigma=STD_GWEI)
    gas_eth = gwei * 1e-9 * GAS_UNITS_PER_SWAP * n_swaps
    return float(gas_eth * eth_price_usd)


def run_backtest(
    df: pd.DataFrame,
    analysis_results: dict,
    start_idx: int,
    risk_aversion_lambda: float = 1.0,
    include_frictions: bool = True
) -> dict:
    """Exécute la simulation hors-échantillon (Out-of-Sample) avec suivi complet du portefeuille."""
    np.random.seed(101)
    spreads = analysis_results["spreads"]
    zscores = analysis_results["zscores"]
    pair_metrics = analysis_results["pair_metrics"]

    assets = [c for c in df.columns if c != "Date"]
    total_initial_capital = INITIAL_CAPITAL_PER_ASSET * len(assets)

    # Suivi des poids engagés par actif et des positions ouvertes par paire
    committed_weights = {a: 0.0 for a in assets}
    open_positions = {p: None for p in pair_metrics.keys()}

    portfolio_history = []
    completed_trades = []
    cumulative_pnl_usd = 0.0
    total_fees_paid = 0.0
    total_slippage_paid = 0.0
    total_gas_paid = 0.0

    dates = df["Date"].to_numpy()
    n_steps = len(df)

    for t in range(start_idx, n_steps):
        eth_price_t = float(df["WETH"].iloc[t]) if "WETH" in df.columns else ETH_REFERENCE_PRICE_USD

        # 1. VÉRIFICATION DES SIGNAUX DE CLÔTURE (Positions ouvertes qui reviennent vers la moyenne)
        for pair_name, pos in list(open_positions.items()):
            if pos is None:
                continue

            z_now = float(zscores[pair_name].iloc[t])
            holding_hours = t - pos["open_idx"]

            # Clôture si le Z-score repasse sous CLOSE_THRESHOLD ou si le signe s'inverse
            crossed_mean = (abs(z_now) <= CLOSE_THRESHOLD) or (z_now * pos["open_z"] < 0)
            if crossed_mean or (t == n_steps - 1):
                pm = pair_metrics[pair_name]
                leg1, leg2 = pm["leg1"], pm["leg2"]

                # Calcul de la variation du spread logarithmique capturée
                spread_open = pos["open_spread"]
                spread_close = float(spreads[pair_name].iloc[t])
                direction = pos["direction"]  # +1 si Long spread (Long leg1, Short leg2), -1 sinon

                gross_spread_return = direction * (spread_close - spread_open)
                notional_usd = pos["notional_usd"]
                gross_pnl = notional_usd * gross_spread_return

                # Frictions à la clôture
                if include_frictions:
                    swap_fee_usd = notional_usd * pm["swap_fee"] * 2.0
                    slip_rate = compute_amm_slippage(notional_usd, pm["pool_liquidity"])
                    slippage_usd = notional_usd * slip_rate * 2.0
                    gas_usd = sample_stochastic_gas_usd(eth_price_t, n_swaps=2)
                else:
                    swap_fee_usd = slippage_usd = gas_usd = 0.0

                net_pnl = gross_pnl - (swap_fee_usd + slippage_usd + gas_usd + pos["entry_frictions_usd"])
                cumulative_pnl_usd += net_pnl
                total_fees_paid += swap_fee_usd + pos["entry_fee_usd"]
                total_slippage_paid += slippage_usd + pos["entry_slip_usd"]
                total_gas_paid += gas_usd + pos["entry_gas_usd"]

                # Libération du capital engagé
                committed_weights[pos["long_asset"]] = max(0.0, committed_weights[pos["long_asset"]] - pos["w_long"])
                committed_weights[pos["short_asset"]] = max(0.0, committed_weights[pos["short_asset"]] - abs(pos["w_short"]))

                completed_trades.append({
                    "pair": pair_name,
                    "universe": pm["universe"],
                    "open_date": dates[pos["open_idx"]],
                    "close_date": dates[t],
                    "holding_hrs": holding_hours,
                    "direction": "Long Spread" if direction == 1 else "Short Spread",
                    "notional_usd": notional_usd,
                    "gross_pnl": gross_pnl,
                    "net_pnl": net_pnl
                })
                open_positions[pair_name] = None

        # 2. DÉTECTION DES SIGNAUX D'OUVERTURE (Franchissement de OPEN_THRESHOLD)
        active_signals = []
        for pair_name, pm in pair_metrics.items():
            if open_positions[pair_name] is not None or not pm["cointegrated"]:
                continue

            z_now = float(zscores[pair_name].iloc[t])
            if abs(z_now) > OPEN_THRESHOLD:
                if z_now > OPEN_THRESHOLD:
                    # Spread trop haut : on shorte leg1 et on achète (long) leg2
                    active_signals.append({
                        "pair": pair_name,
                        "long_asset": pm["leg2"],
                        "short_asset": pm["leg1"],
                        "direction": -1,
                        "z_abs": abs(z_now),
                        "z_raw": z_now
                    })
                else:
                    # Spread trop bas : on achète (long) leg1 et on shorte leg2
                    active_signals.append({
                        "pair": pair_name,
                        "long_asset": pm["leg1"],
                        "short_asset": pm["leg2"],
                        "direction": 1,
                        "z_abs": abs(z_now),
                        "z_raw": z_now
                    })

        # 3. ALLOCATION OPTIMALE DU CAPITAL (Solveur Bi-Objectif)
        if len(active_signals) > 0 and t < n_steps - 1:
            allocations = solve_optimal_weights(
                active_signals=active_signals,
                pair_metrics=pair_metrics,
                committed_weights=committed_weights,
                risk_aversion_lambda=risk_aversion_lambda,
                capital_per_asset=INITIAL_CAPITAL_PER_ASSET
            )

            for sig in active_signals:
                p_name = sig["pair"]
                if p_name not in allocations:
                    continue

                w_l, w_s = allocations[p_name]
                pm = pair_metrics[p_name]
                notional_usd = w_l * INITIAL_CAPITAL_PER_ASSET

                if include_frictions:
                    entry_fee = notional_usd * pm["swap_fee"] * 2.0
                    entry_slip = notional_usd * compute_amm_slippage(notional_usd, pm["pool_liquidity"]) * 2.0
                    entry_gas = sample_stochastic_gas_usd(eth_price_t, n_swaps=2)
                else:
                    entry_fee = entry_slip = entry_gas = 0.0

                entry_frictions = entry_fee + entry_slip + entry_gas

                open_positions[p_name] = {
                    "open_idx": t,
                    "open_z": sig["z_raw"],
                    "open_spread": float(spreads[p_name].iloc[t]),
                    "direction": sig["direction"],
                    "long_asset": sig["long_asset"],
                    "short_asset": sig["short_asset"],
                    "w_long": w_l,
                    "w_short": w_s,
                    "notional_usd": notional_usd,
                    "entry_fee_usd": entry_fee,
                    "entry_slip_usd": entry_slip,
                    "entry_gas_usd": entry_gas,
                    "entry_frictions_usd": entry_frictions
                }

                committed_weights[sig["long_asset"]] += w_l
                committed_weights[sig["short_asset"]] += abs(w_s)

        # Valorisation Mark-to-Market du portefeuille à l'instant t
        unrealized_pnl = 0.0
        for p_name, pos in open_positions.items():
            if pos is not None:
                sp_now = float(spreads[p_name].iloc[t])
                unrealized_pnl += pos["notional_usd"] * pos["direction"] * (sp_now - pos["open_spread"])

        portfolio_history.append({
            "Date": dates[t],
            "Equity": total_initial_capital + cumulative_pnl_usd + unrealized_pnl
        })

    equity_df = pd.DataFrame(portfolio_history)
    trades_df = pd.DataFrame(completed_trades)
    metrics = compute_performance_metrics(
        equity_df, trades_df, total_initial_capital, total_fees_paid, total_slippage_paid, total_gas_paid
    )

    return {
        "equity_curve": equity_df,
        "trades": trades_df,
        "metrics": metrics
    }


def compute_performance_metrics(
    equity_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    initial_capital: float,
    fees_usd: float,
    slippage_usd: float,
    gas_usd: float
) -> dict:
    """Calcule l'ensemble des métriques académiques exigées par l'énoncé (Section 5)."""
    final_equity = float(equity_df["Equity"].iloc[-1])
    total_return = (final_equity - initial_capital) / initial_capital

    n_hours = len(equity_df)
    years = max(n_hours / (365.25 * 24.0), 1e-4)
    annualized_return = (1.0 + total_return) ** (1.0 / years) - 1.0

    hourly_rets = equity_df["Equity"].pct_change().dropna()
    ann_volatility = float(hourly_rets.std() * np.sqrt(365.25 * 24.0))
    sharpe_ratio = (annualized_return - RISK_FREE_RATE) / ann_volatility if ann_volatility > 1e-6 else 0.0

    # Maximum Drawdown
    running_max = equity_df["Equity"].cummax()
    drawdowns = (equity_df["Equity"] - running_max) / running_max
    max_drawdown = float(abs(drawdowns.min()))

    if len(trades_df) > 0:
        wins = trades_df[trades_df["net_pnl"] > 0]
        losses = trades_df[trades_df["net_pnl"] <= 0]
        win_rate = len(wins) / len(trades_df)
        win_loss_ratio = len(wins) / max(len(losses), 1)
        avg_win = float(wins["net_pnl"].mean()) if len(wins) > 0 else 0.0
        avg_loss = float(losses["net_pnl"].mean()) if len(losses) > 0 else 0.0
        avg_holding_hrs = float(trades_df["holding_hrs"].mean())
    else:
        win_rate = win_loss_ratio = avg_win = avg_loss = avg_holding_hrs = 0.0

    return {
        "Annualized Return (%)": round(annualized_return * 100, 2),
        "Annualized Volatility (%)": round(ann_volatility * 100, 2),
        "Sharpe Ratio": round(sharpe_ratio, 2),
        "Max Drawdown (%)": round(max_drawdown * 100, 2),
        "Total Trades": int(len(trades_df)),
        "Win Rate (%)": round(win_rate * 100, 2),
        "Win/Loss Ratio": round(win_loss_ratio, 2),
        "Avg Win ($)": round(avg_win, 2),
        "Avg Loss ($)": round(avg_loss, 2),
        "Avg Holding (hrs)": round(avg_holding_hrs, 2),
        "Swap Fees Paid ($)": round(fees_usd, 2),
        "Slippage Cost ($)": round(slippage_usd, 2),
        "Gas Cost ($)": round(gas_usd, 2)
    }
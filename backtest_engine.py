"""
Module 4 : Moteur de Backtest DeFi.
- Clôture vérifiée uniquement pour les positions ouvertes à une heure antérieure (t > open_idx).
- Exécution décalée à t+1 (aucun trade au prix qui génère le signal).
- Slippage Uniswap V3 tenant compte de la liquidité concentrée par tier de frais.
- Valorisation horaire mark-to-market complète du portefeuille.
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
    ETH_REFERENCE_PRICE_USD,
)
from optimizer import solve_optimal_weights, effective_v3_liquidity


def compute_amm_slippage(trade_usd: float, pool_liquidity_usd: float, swap_fee: float = 0.0001) -> float:
    """Impact de prix sur un pool Uniswap V3 de liquidité effective concentrée L_eff."""
    eff_liq = effective_v3_liquidity(pool_liquidity_usd, swap_fee)
    return float(trade_usd / (2.0 * eff_liq))


def sample_stochastic_gas_usd(eth_price_usd: float, n_swaps: int = 1) -> float:
    """Coût de gaz L1 stochastique log-normal pour n_swaps transactions."""
    gwei = np.random.lognormal(mean=np.log(MEAN_GWEI), sigma=STD_GWEI)
    gas_eth = gwei * 1e-9 * GAS_UNITS_PER_SWAP * n_swaps
    return float(gas_eth * eth_price_usd)


def run_backtest(
    df: pd.DataFrame,
    analysis_results: dict,
    start_idx: int,
    risk_aversion_lambda: float = 1.0,
    include_frictions: bool = True,
) -> dict:
    np.random.seed(101)
    zscores = analysis_results["zscores"]
    pm_dict = analysis_results["pair_metrics"]

    assets = [c for c in df.columns if c != "Date"]
    total_initial_capital = INITIAL_CAPITAL_PER_ASSET * len(assets)

    committed_weights = {a: 0.0 for a in assets}
    open_positions = {p: None for p in pm_dict.keys()}
    pending_signals = []

    portfolio_history = []
    completed_trades = []
    realized_cash_pnl = 0.0
    total_fees = 0.0
    total_slip = 0.0
    total_gas = 0.0

    dates = df["Date"].to_numpy()
    n_steps = len(df)

    for t in range(start_idx, n_steps):
        eth_price = float(df["WETH"].iloc[t]) if "WETH" in df.columns else ETH_REFERENCE_PRICE_USD

        # 1. CLÔTURE DES POSITIONS OUVERTES LORS D'HEURES PRÉCÉDENTES (t > open_idx)
        for p_name, pos in list(open_positions.items()):
            if pos is None or t <= pos["open_idx"]:
                continue
            z_now = float(zscores[p_name].iloc[t])
            crossed_mean = (abs(z_now) <= CLOSE_THRESHOLD) or (z_now * pos["open_z"] < 0)

            if crossed_mean or (t == n_steps - 1):
                pm = pm_dict[p_name]
                p1_now = float(df[pos["leg1"]].iloc[t])
                p2_now = float(df[pos["leg2"]].iloc[t])

                gross_pnl = pos["qty1"] * (p1_now - pos["open_p1"]) + pos["qty2"] * (p2_now - pos["open_p2"])
                close_notional = abs(pos["qty1"]) * p1_now

                if include_frictions:
                    close_fee = close_notional * pm["swap_fee"]
                    close_slip = close_notional * compute_amm_slippage(close_notional, pm["pool_liquidity"], pm["swap_fee"])
                    close_gas = sample_stochastic_gas_usd(eth_price, n_swaps=1)
                else:
                    close_fee = close_slip = close_gas = 0.0

                total_frictions = (
                    pos["entry_fee"] + pos["entry_slip"] + pos["entry_gas"]
                    + close_fee + close_slip + close_gas
                )
                net_pnl = gross_pnl - total_frictions

                realized_cash_pnl += net_pnl
                total_fees += pos["entry_fee"] + close_fee
                total_slip += pos["entry_slip"] + close_slip
                total_gas += pos["entry_gas"] + close_gas

                committed_weights[pos["long_asset"]] = max(0.0, committed_weights[pos["long_asset"]] - pos["w_long"])
                committed_weights[pos["short_asset"]] = max(0.0, committed_weights[pos["short_asset"]] - abs(pos["w_short"]))

                completed_trades.append({
                    "pair": p_name,
                    "universe": pm["universe"],
                    "open_date": dates[pos["open_idx"]],
                    "close_date": dates[t],
                    "holding_hrs": t - pos["open_idx"],
                    "gross_pnl": gross_pnl,
                    "net_pnl": net_pnl,
                })
                open_positions[p_name] = None

        # 2. EXÉCUTION À t DES SIGNAUX DÉTECTÉS EN t-1 (exécution à t+1)
        if pending_signals and t < n_steps - 1:
            allocations = solve_optimal_weights(
                active_signals=pending_signals,
                pair_metrics=pm_dict,
                committed_weights=committed_weights,
                risk_aversion_lambda=risk_aversion_lambda,
                capital_per_asset=INITIAL_CAPITAL_PER_ASSET,
            )
            for sig in pending_signals:
                p_name = sig["pair"]
                if p_name not in allocations or open_positions[p_name] is not None:
                    continue
                w_l, w_s = allocations[p_name]
                pm = pm_dict[p_name]
                leg1, leg2 = pm["leg1"], pm["leg2"]
                p1_now, p2_now = float(df[leg1].iloc[t]), float(df[leg2].iloc[t])

                notional_leg1 = w_l * INITIAL_CAPITAL_PER_ASSET
                notional_leg2 = abs(w_s) * INITIAL_CAPITAL_PER_ASSET

                if sig["direction"] == 1:
                    qty1 = +notional_leg1 / p1_now
                    qty2 = -notional_leg2 / p2_now
                else:
                    qty1 = -notional_leg1 / p1_now
                    qty2 = +notional_leg2 / p2_now

                if include_frictions:
                    entry_fee = notional_leg1 * pm["swap_fee"]
                    entry_slip = notional_leg1 * compute_amm_slippage(notional_leg1, pm["pool_liquidity"], pm["swap_fee"])
                    entry_gas = sample_stochastic_gas_usd(eth_price, n_swaps=1)
                else:
                    entry_fee = entry_slip = entry_gas = 0.0

                open_positions[p_name] = {
                    "open_idx": t,
                    "open_z": sig["z_raw"],
                    "direction": sig["direction"],
                    "long_asset": sig["long_asset"],
                    "short_asset": sig["short_asset"],
                    "w_long": w_l,
                    "w_short": w_s,
                    "leg1": leg1,
                    "leg2": leg2,
                    "open_p1": p1_now,
                    "open_p2": p2_now,
                    "qty1": qty1,
                    "qty2": qty2,
                    "notional_leg1": notional_leg1,
                    "entry_fee": entry_fee,
                    "entry_slip": entry_slip,
                    "entry_gas": entry_gas,
                }
                committed_weights[sig["long_asset"]] += w_l
                committed_weights[sig["short_asset"]] += abs(w_s)
            pending_signals = []

        # 3. DÉTECTION DES SIGNAUX À L'HEURE t (seront exécutés en t+1)
        pending_signals = []
        for p_name, pm in pm_dict.items():
            if open_positions[p_name] is not None or not pm["cointegrated"]:
                continue
            z_now = float(zscores[p_name].iloc[t])
            if abs(z_now) > OPEN_THRESHOLD:
                direction = -1 if z_now > OPEN_THRESHOLD else 1
                long_a, short_a = (pm["leg2"], pm["leg1"]) if direction == -1 else (pm["leg1"], pm["leg2"])
                pending_signals.append({
                    "pair": p_name,
                    "long_asset": long_a,
                    "short_asset": short_a,
                    "direction": direction,
                    "z_abs": abs(z_now),
                    "z_raw": z_now,
                })

        # 4. VALORISATION MARK-TO-MARKET HORAIRE
        unrealized_pnl = 0.0
        for p_name, pos in open_positions.items():
            if pos is not None:
                p1_now = float(df[pos["leg1"]].iloc[t])
                p2_now = float(df[pos["leg2"]].iloc[t])
                mtm = pos["qty1"] * (p1_now - pos["open_p1"]) + pos["qty2"] * (p2_now - pos["open_p2"])
                unrealized_pnl += mtm - (pos["entry_fee"] + pos["entry_slip"] + pos["entry_gas"])

        portfolio_history.append({
            "Date": dates[t],
            "Equity": total_initial_capital + realized_cash_pnl + unrealized_pnl,
        })

    eq_df = pd.DataFrame(portfolio_history)
    tr_df = pd.DataFrame(completed_trades)

    n_hours = max(len(eq_df), 1)
    total_return = (eq_df["Equity"].iloc[-1] / total_initial_capital) - 1.0
    ann_ret = ((1.0 + total_return) ** (8760.0 / n_hours)) - 1.0
    hourly_ret = eq_df["Equity"].pct_change().dropna()
    vol = float(hourly_ret.std() * np.sqrt(8760.0)) if len(hourly_ret) > 1 else 0.0
    sharpe = (ann_ret - RISK_FREE_RATE) / vol if vol > 1e-6 else 0.0
    running_max = eq_df["Equity"].cummax()
    max_dd = float(abs(((eq_df["Equity"] - running_max) / running_max).min()))

    if not tr_df.empty:
        wins = tr_df[tr_df["net_pnl"] > 0]
        losses = tr_df[tr_df["net_pnl"] <= 0]
        win_rate = len(wins) / len(tr_df) * 100.0
        win_loss_ratio = len(wins) / max(len(losses), 1)
        avg_win = float(wins["net_pnl"].mean()) if not wins.empty else 0.0
        avg_loss = float(losses["net_pnl"].mean()) if not losses.empty else 0.0
        avg_hold = float(tr_df["holding_hrs"].mean())
    else:
        win_rate = win_loss_ratio = avg_win = avg_loss = avg_hold = 0.0

    metrics = {
        "Annualized Return (%)": round(ann_ret * 100, 2),
        "Annualized Volatility (%)": round(vol * 100, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Max Drawdown (%)": round(max_dd * 100, 2),
        "Total Trades": int(len(tr_df)),
        "Win Rate (%)": round(win_rate, 2),
        "Win/Loss Ratio": round(win_loss_ratio, 2),
        "Avg Win ($)": round(avg_win, 2),
        "Avg Loss ($)": round(avg_loss, 2),
        "Avg Holding (hrs)": round(avg_hold, 2),
        "Swap Fees Paid ($)": round(total_fees, 2),
        "Slippage Cost ($)": round(total_slip, 2),
        "Gas Cost ($)": round(total_gas, 2),
    }

    return {"equity_curve": eq_df, "trades": tr_df, "metrics": metrics}

"""
Module 1 : Acquisition de données historiques DEX réelles (GeckoTerminal, Uniswap V3, 1h).
"""

import json
import os
import re
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from config import START_DATE, END_DATE, DEX_PAIRS, TOKENS, GECKO_NETWORK

BASE_URL = "https://api.geckoterminal.com/api/v2"
HEADERS = {"Accept": "application/json;version=20230302"}
REQUEST_PAUSE_S = 2.2
MAX_RETRIES = 5
MAX_FFILL_HOURS = 12       # Sur DEX, absence de swap = prix on-chain inchangé
MAX_MISSING_FRAC = 0.05    # Seuil maximal de trous après ffill de 12h
DATA_DIR = "data"
POOLS_JSON = os.path.join(DATA_DIR, "pools_resolved.json")


def _get(path: str, params: dict | None = None) -> dict:
    for attempt in range(MAX_RETRIES):
        resp = requests.get(BASE_URL + path, params=params, headers=HEADERS, timeout=20)
        if resp.status_code == 200:
            time.sleep(REQUEST_PAUSE_S)
            return resp.json()
        if resp.status_code == 429:
            time.sleep(10 * (attempt + 1))
            continue
        resp.raise_for_status()
    raise RuntimeError(f"GeckoTerminal : échec après {MAX_RETRIES} tentatives sur {path}")


def _ts(date_str: str) -> int:
    return int(datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc).timestamp())


def discover_pool(addr_a: str, addr_b: str, fee_pct: float, pages: int = 3) -> dict:
    addr_a, addr_b = addr_a.lower(), addr_b.lower()
    candidates = []
    for page in range(1, pages + 1):
        payload = _get(f"/networks/{GECKO_NETWORK}/tokens/{addr_a}/pools", {"page": page})
        for pool in payload.get("data", []):
            attrs, rel = pool["attributes"], pool["relationships"]
            ids = {rel["base_token"]["data"]["id"].split("_")[-1].lower(),
                   rel["quote_token"]["data"]["id"].split("_")[-1].lower()}
            m = re.search(r"(\d+(?:\.\d+)?)\s*%", attrs.get("name", ""))
            fee = float(m.group(1)) if m else None
            dex = rel.get("dex", {}).get("data", {}).get("id", "")
            if ids == {addr_a, addr_b} and fee is not None and abs(fee - fee_pct) < 1e-9 \
                    and "uniswap_v3" in dex:
                candidates.append({
                    "address": attrs["address"].lower(),
                    "name": attrs["name"],
                    "reserve_in_usd": float(attrs.get("reserve_in_usd") or 0.0),
                })
    if not candidates:
        raise LookupError(f"Aucun pool Uniswap V3 {addr_a}/{addr_b} au tier {fee_pct}% trouvé")
    return max(candidates, key=lambda c: c["reserve_in_usd"])


def resolve_pools() -> dict:
    resolved = {}
    for pair, cfg in DEX_PAIRS.items():
        info = discover_pool(TOKENS[cfg["leg1"]], TOKENS[cfg["leg2"]], cfg["swap_fee"] * 100)
        info["leg1"], info["leg2"], info["fee_pct"] = cfg["leg1"], cfg["leg2"], cfg["swap_fee"] * 100
        resolved[pair] = info
        print(f"[Data] {pair:12s} -> {info['address']}  '{info['name']}'  TVL ${info['reserve_in_usd']:,.0f}")
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(POOLS_JSON, "w") as f:
        json.dump({"resolved_at": datetime.now(timezone.utc).isoformat(), "pools": resolved}, f, indent=2)
    return resolved


def apply_resolved_pools(path: str = POOLS_JSON) -> None:
    with open(path) as f:
        pools = json.load(f)["pools"]
    for pair, info in pools.items():
        if pair in DEX_PAIRS:
            if info["reserve_in_usd"] > 0:
                DEX_PAIRS[pair]["pool_liquidity"] = info["reserve_in_usd"]
            DEX_PAIRS[pair]["gecko_pool"] = info["address"]


def fetch_token_hourly(pool: str, token_addr: str, start_ts: int, end_ts: int) -> pd.Series:
    rows, before = {}, end_ts + 3600
    while True:
        payload = _get(
            f"/networks/{GECKO_NETWORK}/pools/{pool}/ohlcv/hour",
            {"aggregate": 1, "limit": 1000, "before_timestamp": before,
             "currency": "usd", "token": token_addr},
        )
        batch = payload["data"]["attributes"]["ohlcv_list"]
        if not batch:
            break
        for ts, _o, _h, _l, close, _v in batch:
            rows[int(ts)] = float(close)
        oldest = min(int(r[0]) for r in batch)
        if oldest <= start_ts or oldest >= before:
            break
        before = oldest
    s = pd.Series(rows, dtype=float).sort_index()
    s.index = pd.to_datetime(s.index, unit="s")
    return s[(s.index >= pd.to_datetime(start_ts, unit="s")) & (s.index <= pd.to_datetime(end_ts, unit="s"))]


def build_dataset(resolved: dict) -> pd.DataFrame:
    start_ts, end_ts = _ts(START_DATE), _ts(END_DATE)
    source = {}
    for pair, info in resolved.items():
        source.setdefault(info["leg1"], info["address"])
    for pair, info in resolved.items():
        source.setdefault(info["leg2"], info["address"])

    grid = pd.date_range(START_DATE, END_DATE, freq="1h")
    cols = {}
    for token, pool in source.items():
        print(f"[Data] Téléchargement {token} (pool {pool[:10]}...)")
        s = fetch_token_hourly(pool, TOKENS[token], start_ts, end_ts)
        if s.empty:
            raise RuntimeError(f"{token} : aucune donnée reçue")
        raw_obs = len(s)
        s = s.reindex(grid).ffill(limit=MAX_FFILL_HOURS)
        missing = s.isna().mean()
        print(f"[Data]   {token}: {raw_obs} bougies brutes, {missing:.2%} de trous après ffill, "
              f"début {s.first_valid_index()}")
        if missing > MAX_MISSING_FRAC:
            raise RuntimeError(f"{token} : {missing:.1%} d'heures manquantes (> {MAX_MISSING_FRAC:.0%}).")
        cols[token] = s.ffill().bfill()
    df = pd.DataFrame(cols)
    df.insert(0, "Date", grid)
    return df.reset_index(drop=True)


def load_or_fetch_data(csv_path: str = os.path.join(DATA_DIR, "dex_hourly_prices.csv")) -> pd.DataFrame:
    meta_path = csv_path.replace(".csv", ".meta.json")
    if os.path.exists(csv_path) and os.path.exists(meta_path) and os.path.exists(POOLS_JSON):
        apply_resolved_pools()
        return pd.read_csv(csv_path, parse_dates=["Date"])

    os.makedirs(DATA_DIR, exist_ok=True)
    print("[Data] Extraction des données réelles via GeckoTerminal...")
    resolved = resolve_pools()
    df = build_dataset(resolved)
    df.to_csv(csv_path, index=False)
    with open(meta_path, "w") as f:
        json.dump({"source": "GeckoTerminal", "network": GECKO_NETWORK, "start": START_DATE,
                   "end": END_DATE, "fetched_at": datetime.now(timezone.utc).isoformat(),
                   "rows": len(df)}, f, indent=2)
    apply_resolved_pools()
    print(f"[Data] Sauvegardé : '{csv_path}' ({len(df)} lignes).")
    return df


if __name__ == "__main__":
    df = load_or_fetch_data()
    print(df.head())

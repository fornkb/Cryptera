"""
Outcome-label saved snapshots by replaying the realized 15m / 1h candles that
followed each snapshot timestamp.

For every snapshot that contains an `analysis.trade_decision` with non-null
entry / stop_loss / take_profit, we fetch the next `horizon_hours` of candles
and record:
  * triggered:           did price reach the entry level?
  * tp_first:            did TP fill before SL?
  * sl_first:            did SL fill before TP?
  * timeout:             neither side hit within the horizon
  * mfe / mae (pct):     max favourable / adverse excursion vs. entry
  * realized_rr:         (price extreme towards TP - entry) / (entry - sl)

Forward-scenario contingent setups are labelled the same way under
`forward_outcome` when their entry/SL/TP are present.

Usage:
    python -m tools.label_snapshots                # label everything
    python -m tools.label_snapshots --symbol BTC/USDT --horizon 24
    python -m tools.label_snapshots --force        # re-label already-labelled snapshots

Output: each snapshot file is rewritten in place with an `outcome` block.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone

import ccxt.async_support as ccxt
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAPSHOTS_DIR = os.path.join(ROOT_DIR, "snapshots")


def _parse_snapshot_time(filename: str) -> datetime | None:
    """snapshot_<base>_<quote>_<YYYYMMDD>_<HHMMSS>.json -> datetime"""
    parts = filename.replace(".json", "").split("_")
    if len(parts) < 5:
        return None
    date_p, time_p = parts[-2], parts[-1]
    try:
        return datetime.strptime(f"{date_p}{time_p}", "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _symbol_from_filename(filename: str) -> str | None:
    parts = filename.replace(".json", "").split("_")
    if len(parts) < 5:
        return None
    return f"{parts[1]}/{parts[2]}"


async def _fetch_future_candles(exchange, symbol: str, since_ms: int, horizon_hours: int):
    """Fetch up to `horizon_hours` of 15m candles starting from `since_ms`."""
    needed = max(4, int(horizon_hours * 4))
    try:
        candles = await exchange.fetch_ohlcv(symbol, "15m", since=since_ms, limit=needed + 4)
    except Exception as e:
        print(f"[WARN] fetch failed for {symbol} @ {since_ms}: {e}")
        return None
    if not candles:
        return None
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    cutoff = pd.Timestamp(since_ms, unit="ms", tz="UTC") + pd.Timedelta(hours=horizon_hours)
    return df[df["timestamp"] <= cutoff].reset_index(drop=True)


def _label_one(side: str, entry: float, sl: float, tp: float, df: pd.DataFrame) -> dict:
    """
    side: 'long' or 'short'.
    Walk forward bar by bar. A bar that touches BOTH SL and TP is conservatively
    counted as `sl_first` (assume worst case — pessimistic estimator).
    """
    if df is None or df.empty:
        return {"triggered": False, "outcome": "no_data"}

    triggered = False
    entry_idx = None
    for i, row in df.iterrows():
        if side == "long":
            if row["low"] <= entry <= row["high"]:
                triggered = True
                entry_idx = i
                break
        else:
            if row["low"] <= entry <= row["high"]:
                triggered = True
                entry_idx = i
                break

    if not triggered:
        return {"triggered": False, "outcome": "not_triggered",
                "bars_observed": int(len(df))}

    post = df.iloc[entry_idx:].reset_index(drop=True)
    mfe = 0.0
    mae = 0.0
    outcome = "timeout"
    bars_to_close = None

    for i, row in post.iterrows():
        if side == "long":
            fav = (row["high"] - entry) / entry * 100
            adv = (row["low"] - entry) / entry * 100
            mfe = max(mfe, fav)
            mae = min(mae, adv)
            sl_hit = row["low"] <= sl
            tp_hit = row["high"] >= tp
        else:
            fav = (entry - row["low"]) / entry * 100
            adv = (entry - row["high"]) / entry * 100
            mfe = max(mfe, fav)
            mae = min(mae, adv)
            sl_hit = row["high"] >= sl
            tp_hit = row["low"] <= tp

        if sl_hit and tp_hit:
            outcome = "sl_first"
            bars_to_close = i
            break
        if sl_hit:
            outcome = "sl_first"
            bars_to_close = i
            break
        if tp_hit:
            outcome = "tp_first"
            bars_to_close = i
            break

    risk = abs(entry - sl)
    realized_rr = None
    if risk > 0:
        if side == "long":
            extreme = entry * (1 + mfe / 100)
        else:
            extreme = entry * (1 - mfe / 100)
        realized_rr = round(abs(extreme - entry) / risk, 3)

    return {
        "triggered": True,
        "outcome": outcome,
        "bars_to_close": bars_to_close,
        "mfe_pct": round(mfe, 3),
        "mae_pct": round(mae, 3),
        "realized_rr": realized_rr,
        "bars_observed": int(len(df)),
    }


def _direction_to_side(direction: str) -> str | None:
    if not direction:
        return None
    d = direction.upper()
    if d in ("BUY", "LONG"):
        return "long"
    if d in ("SELL", "SHORT"):
        return "short"
    return None


async def label_snapshot_file(exchange, path: str, horizon_hours: int, force: bool):
    with open(path, "r") as f:
        snapshot = json.load(f)

    if not force and snapshot.get("outcome"):
        return False

    filename = os.path.basename(path)
    snap_time = _parse_snapshot_time(filename)
    symbol = snapshot.get("symbol") or _symbol_from_filename(filename)
    if snap_time is None or symbol is None:
        print(f"[SKIP] cannot parse symbol/time from {filename}")
        return False

    # Bail out if the horizon end is in the future
    if snap_time + pd.Timedelta(hours=horizon_hours) > datetime.now(timezone.utc):
        return False

    since_ms = int(snap_time.timestamp() * 1000)
    df = await _fetch_future_candles(exchange, symbol, since_ms, horizon_hours)
    if df is None or df.empty:
        print(f"[WARN] no future data for {filename}")
        return False

    analysis = snapshot.get("analysis") or {}
    td = analysis.get("trade_decision") or {}
    fwd = analysis.get("forward_scenario") or {}

    trade_outcome = None
    side = _direction_to_side(td.get("primary", {}).get("direction"))
    if side and td.get("entry") is not None and td.get("stop_loss") is not None and td.get("take_profit") is not None:
        trade_outcome = _label_one(side, float(td["entry"]), float(td["stop_loss"]),
                                   float(td["take_profit"]), df)

    forward_outcome = None
    fwd_side = _direction_to_side(fwd.get("direction"))
    if fwd_side and fwd.get("entry") is not None and fwd.get("stop_loss") is not None and fwd.get("take_profit") is not None:
        forward_outcome = _label_one(fwd_side, float(fwd["entry"]), float(fwd["stop_loss"]),
                                     float(fwd["take_profit"]), df)

    snapshot["outcome"] = {
        "labeled_at": datetime.utcnow().isoformat() + "Z",
        "horizon_hours": horizon_hours,
        "bars_observed": int(len(df)),
        "trade_decision_outcome": trade_outcome,
        "forward_scenario_outcome": forward_outcome,
    }

    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2)
    return True


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default=None, help="restrict to one symbol (e.g. BTC/USDT)")
    p.add_argument("--horizon", type=int, default=24, help="hours of look-forward (default 24)")
    p.add_argument("--force", action="store_true", help="re-label already labelled snapshots")
    args = p.parse_args()

    if not os.path.isdir(SNAPSHOTS_DIR):
        print(f"[ERR] snapshots dir not found: {SNAPSHOTS_DIR}")
        sys.exit(1)

    targets = []
    for name in sorted(os.listdir(SNAPSHOTS_DIR)):
        if not name.startswith("snapshot_") or not name.endswith(".json"):
            continue
        if args.symbol:
            sym = _symbol_from_filename(name)
            if sym != args.symbol:
                continue
        targets.append(os.path.join(SNAPSHOTS_DIR, name))

    if not targets:
        print("[INFO] no snapshots match the filter.")
        return

    exchange = ccxt.binance({"enableRateLimit": True})
    updated = 0
    try:
        for path in targets:
            try:
                if await label_snapshot_file(exchange, path, args.horizon, args.force):
                    updated += 1
                    print(f"[OK] labelled {os.path.basename(path)}")
            except Exception as e:
                print(f"[ERR] {os.path.basename(path)}: {e}")
    finally:
        await exchange.close()
    print(f"\nDone. Labelled {updated} / {len(targets)} snapshot(s).")


if __name__ == "__main__":
    asyncio.run(main())

"""
Aggregate labelled snapshots into per-bucket performance stats.

After running tools.label_snapshots, each snapshot file carries an `outcome`
block. This script walks every labelled snapshot and reports:
  * overall hit rate, expectancy, average MFE/MAE
  * stats by score bucket (45-59 conditional, 60-74 active, 75+ high-conviction)
  * stats by trend bias (bullish / bearish)
  * stats by volatility regime
  * stats by event_guard.active state
  * stats by volume_gate

Usage:
    python -m tools.eval_engine
    python -m tools.eval_engine --symbol BTC/USDT
    python -m tools.eval_engine --out reports/eval_2026q2.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Iterable

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAPSHOTS_DIR = os.path.join(ROOT_DIR, "snapshots")


def _iter_snapshots(symbol: str | None) -> Iterable[dict]:
    for name in sorted(os.listdir(SNAPSHOTS_DIR)):
        if not name.startswith("snapshot_") or not name.endswith(".json"):
            continue
        if symbol:
            parts = name.split("_")
            if len(parts) < 3 or f"{parts[1]}/{parts[2]}" != symbol:
                continue
        path = os.path.join(SNAPSHOTS_DIR, name)
        try:
            with open(path) as f:
                yield json.load(f)
        except Exception:
            continue


def _bucket_score(score: int) -> str:
    if score >= 75:
        return "75+"
    if score >= 60:
        return "60-74"
    if score >= 45:
        return "45-59"
    return "<45"


def _accumulator():
    return {
        "n": 0,
        "triggered": 0,
        "tp": 0,
        "sl": 0,
        "timeout": 0,
        "no_data": 0,
        "sum_mfe": 0.0,
        "sum_mae": 0.0,
        "sum_rr": 0.0,
        "rr_n": 0,
    }


def _record(acc, outcome):
    acc["n"] += 1
    if not outcome:
        return
    if outcome.get("outcome") == "no_data":
        acc["no_data"] += 1
        return
    if outcome.get("triggered"):
        acc["triggered"] += 1
        o = outcome.get("outcome")
        if o == "tp_first":
            acc["tp"] += 1
        elif o == "sl_first":
            acc["sl"] += 1
        else:
            acc["timeout"] += 1
        if outcome.get("mfe_pct") is not None:
            acc["sum_mfe"] += float(outcome["mfe_pct"])
        if outcome.get("mae_pct") is not None:
            acc["sum_mae"] += float(outcome["mae_pct"])
        if outcome.get("realized_rr") is not None:
            acc["sum_rr"] += float(outcome["realized_rr"])
            acc["rr_n"] += 1


def _finalize(acc):
    n = max(acc["n"], 1)
    triggered = max(acc["triggered"], 1)
    rr_n = max(acc["rr_n"], 1)
    return {
        "snapshots": acc["n"],
        "triggered": acc["triggered"],
        "tp_first": acc["tp"],
        "sl_first": acc["sl"],
        "timeout": acc["timeout"],
        "no_data": acc["no_data"],
        "trigger_rate": round(acc["triggered"] / n, 3),
        "hit_rate": round(acc["tp"] / triggered, 3) if acc["triggered"] else None,
        "avg_mfe_pct": round(acc["sum_mfe"] / triggered, 3) if acc["triggered"] else None,
        "avg_mae_pct": round(acc["sum_mae"] / triggered, 3) if acc["triggered"] else None,
        "avg_realized_rr": round(acc["sum_rr"] / rr_n, 3) if acc["rr_n"] else None,
    }


def _expectancy(stats):
    """Naive expectancy in pct (avg_mfe * hit_rate + avg_mae * (1 - hit_rate))."""
    hr = stats.get("hit_rate")
    mfe = stats.get("avg_mfe_pct")
    mae = stats.get("avg_mae_pct")
    if hr is None or mfe is None or mae is None:
        return None
    return round(hr * mfe + (1 - hr) * mae, 3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    buckets = {
        "overall": _accumulator(),
        "by_score": defaultdict(_accumulator),
        "by_bias": defaultdict(_accumulator),
        "by_vol_regime_15m": defaultdict(_accumulator),
        "by_event_guard": defaultdict(_accumulator),
        "by_volume_gate": defaultdict(_accumulator),
        "by_action": defaultdict(_accumulator),
    }

    labelled = 0
    seen = 0
    for snap in _iter_snapshots(args.symbol):
        seen += 1
        outcome = (snap.get("outcome") or {}).get("trade_decision_outcome")
        if outcome is None:
            continue
        labelled += 1

        strategies = snap.get("strategies") or {}
        analysis = snap.get("analysis") or {}
        header = analysis.get("header") or {}

        score = int(header.get("score") or strategies.get("confluence_score") or 0)
        bias = (header.get("bias") or (strategies.get("trend_bias") or "neutral").upper())
        gate = (header.get("volume_gate") or (strategies.get("volume_gate") or {}).get("state") or "CLEAR")
        action = header.get("action") or "HOLD"
        vr_15m = (snap.get("volatility_regime") or {}).get("15m", {}).get("regime", "unknown")
        event_active = "active" if (snap.get("event_guard") or {}).get("active") else "clear"

        _record(buckets["overall"], outcome)
        _record(buckets["by_score"][_bucket_score(score)], outcome)
        _record(buckets["by_bias"][bias], outcome)
        _record(buckets["by_vol_regime_15m"][vr_15m], outcome)
        _record(buckets["by_event_guard"][event_active], outcome)
        _record(buckets["by_volume_gate"][gate], outcome)
        _record(buckets["by_action"][action], outcome)

    report = {
        "snapshots_scanned": seen,
        "snapshots_labelled": labelled,
        "overall": _finalize(buckets["overall"]),
    }
    report["overall"]["expectancy_pct"] = _expectancy(report["overall"])

    for k, v in buckets.items():
        if k == "overall":
            continue
        bucket_report = {}
        for sub_k, acc in v.items():
            s = _finalize(acc)
            s["expectancy_pct"] = _expectancy(s)
            bucket_report[sub_k] = s
        report[k] = bucket_report

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote {args.out}")
    else:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

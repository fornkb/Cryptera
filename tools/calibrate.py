"""
Calibrate the confluence engine from labelled snapshots (T1-8).

Reads every snapshot that has an `outcome.engine_trade_outcome` (produced by
tools.label_snapshots) and emits `data/calibration.json` containing:

  * weights      : per-component multipliers for the trend (C1-C8) rubric, derived
                   from each component's point-biserial-style edge (how much more
                   often the component fired on winners than losers), normalised to
                   mean 1.0 so the score range stays ~0-100.
  * mr_weights   : same for the mean-revert (M1-M6) rubric.
  * score_to_winrate : empirical TP-before-SL hit-rate per score bucket.

The engine (strategies.load_calibration) loads this file when present and
otherwise runs at equal weights. This script is safe to run with few samples —
it refuses to emit weights below a minimum support and only writes buckets that
have enough trades.

Usage:
    python -m tools.calibrate                 # write data/calibration.json
    python -m tools.calibrate --min-samples 200 --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAPSHOTS_DIR = os.path.join(ROOT_DIR, "snapshots")
OUT_PATH = os.path.join(ROOT_DIR, "data", "calibration.json")

TREND_KEYS = ["c1_trend_alignment", "c2_ob_proximity", "c3_liquidity_sweep", "c4_momentum",
              "c5_fvg_magnet", "c6_ote_bonus", "c7_cvd_alignment", "c8_stochrsi"]
MR_KEYS = ["m1_edge_distance", "m2_edge_sweep", "m3_cvd_absorption",
           "m4_stoch_extreme", "m5_rejection", "m6_range_intact"]


def _iter_labelled():
    if not os.path.isdir(SNAPSHOTS_DIR):
        return
    for name in sorted(os.listdir(SNAPSHOTS_DIR)):
        if not (name.startswith("snapshot_") and name.endswith(".json")):
            continue
        try:
            with open(os.path.join(SNAPSHOTS_DIR, name)) as f:
                snap = json.load(f)
        except Exception:
            continue
        outcome = (snap.get("outcome") or {}).get("engine_trade_outcome")
        if outcome and outcome.get("triggered") and outcome.get("outcome") in ("tp_first", "sl_first"):
            yield snap, outcome


def _component_weights(rows, keys):
    """
    rows: list of (breakdown_dict, won_bool). Weight each component by the ratio
    of its mean points on winners vs. its mean points on all trades, normalised to
    mean 1.0. Components that don't discriminate stay near 1.0.
    """
    win_sum = defaultdict(float)
    win_n = 0
    all_sum = defaultdict(float)
    all_n = 0
    for bd, won in rows:
        all_n += 1
        for k in keys:
            all_sum[k] += float(bd.get(k, 0) or 0)
        if won:
            win_n += 1
            for k in keys:
                win_sum[k] += float(bd.get(k, 0) or 0)
    if all_n == 0 or win_n == 0:
        return {}
    raw = {}
    for k in keys:
        mean_all = all_sum[k] / all_n
        mean_win = win_sum[k] / win_n
        # edge ratio: >1 means the component fired more on winners
        raw[k] = (mean_win + 1e-6) / (mean_all + 1e-6)
    mean_w = sum(raw.values()) / len(raw)
    if mean_w <= 0:
        return {}
    # normalise to mean 1.0 and clamp to a sane band
    return {k: round(max(0.5, min(1.5, v / mean_w)), 4) for k, v in raw.items()}


def _score_buckets(rows, edges=(0, 45, 60, 75, 101)):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sub = [won for sc, won in rows if lo <= sc < hi]
        if not sub:
            continue
        out.append({"min": lo, "max": hi - 1, "n": len(sub),
                    "win_rate": round(sum(1 for w in sub if w) / len(sub), 3)})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--min-samples", type=int, default=150,
                   help="minimum trades per rubric before emitting weights")
    p.add_argument("--min-bucket", type=int, default=20,
                   help="minimum trades per score bucket before emitting its win-rate")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    trend_rows, mr_rows = [], []      # (breakdown, won)
    score_rows = []                   # (score, won)
    for snap, outcome in _iter_labelled():
        strat = snap.get("strategies") or {}
        bd = strat.get("confluence_breakdown") or {}
        won = outcome.get("outcome") == "tp_first"
        score = int(strat.get("confluence_score") or 0)
        score_rows.append((score, won))
        if strat.get("score_mode") == "mean_revert":
            mr_rows.append((bd, won))
        else:
            trend_rows.append((bd, won))

    n_total = len(score_rows)
    print(f"Labelled engine trades found: {n_total}  (trend={len(trend_rows)}, mean_revert={len(mr_rows)})")

    calib = {"weights": {}, "mr_weights": {}, "score_to_winrate": []}
    if len(trend_rows) >= args.min_samples:
        calib["weights"] = _component_weights(trend_rows, TREND_KEYS)
        print(f"  trend weights fitted on {len(trend_rows)} trades")
    else:
        print(f"  trend weights skipped (need {args.min_samples}, have {len(trend_rows)})")
    if len(mr_rows) >= args.min_samples:
        calib["mr_weights"] = _component_weights(mr_rows, MR_KEYS)
        print(f"  mean-revert weights fitted on {len(mr_rows)} trades")
    else:
        print(f"  mean-revert weights skipped (need {args.min_samples}, have {len(mr_rows)})")

    buckets = [b for b in _score_buckets(score_rows) if b["n"] >= args.min_bucket]
    calib["score_to_winrate"] = buckets
    if buckets:
        print("  score->winrate:")
        for b in buckets:
            print(f"    {b['min']:>3}-{b['max']:<3}  n={b['n']:<4} win_rate={b['win_rate']}")
    else:
        print(f"  no score buckets with >= {args.min_bucket} trades yet")

    if args.dry_run:
        print("\n[dry-run] would write:")
        print(json.dumps(calib, indent=2))
        return

    if not (calib["weights"] or calib["mr_weights"] or calib["score_to_winrate"]):
        print("\nNothing to calibrate yet — not writing the file (engine stays at equal weights).")
        return

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(calib, f, indent=2)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()

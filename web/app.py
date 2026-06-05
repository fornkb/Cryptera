"""
Flask backend for the Cryptera v3.1 dashboard.

Returns the snapshot AND the parsed Gemini JSON analysis from `/api/run` and
`/api/history/<filename>` so the SPA can render structured fields directly.
"""

import os
import sys
import json
import glob
import asyncio
from flask import Flask, render_template, request, jsonify

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

from main import run_analysis

app = Flask(__name__)
SNAPSHOTS_DIR = os.path.join(PARENT_DIR, "snapshots")
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/run", methods=["POST"])
def run():
    data = request.get_json() or {}
    symbol = data.get("symbol", "SOL/USDT").strip()
    if not symbol:
        return jsonify({"error": "Symbol is required"}), 400

    print(f"[API] Starting live market analysis for: {symbol}")
    try:
        snapshot, analysis = asyncio.run(run_analysis(symbol))
        return jsonify({
            "success": True,
            "snapshot": snapshot,
            "analysis": analysis,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/history", methods=["GET"])
def get_history():
    pattern = os.path.join(SNAPSHOTS_DIR, "snapshot_*.json")
    files = glob.glob(pattern)
    history = []
    for file_path in files:
        filename = os.path.basename(file_path)
        try:
            parts = filename.replace(".json", "").split("_")
            if len(parts) >= 5:
                symbol = f"{parts[1]}/{parts[2]}"
                date_p, time_p = parts[3], parts[4]
                timestamp = f"{date_p[:4]}-{date_p[4:6]}-{date_p[6:8]} {time_p[:2]}:{time_p[2:4]}:{time_p[4:6]}"
            else:
                symbol = "Unknown"
                timestamp = "Unknown"

            with open(file_path, "r") as f:
                data = json.load(f)
            strategies = data.get("strategies", {}) or {}
            bias = (strategies.get("trend_bias") or "neutral").upper()
            score = strategies.get("confluence_score", 0) or 0
            regime = data.get("market_regime", "Unknown")
            price = (
                data.get("smc_context", {}).get("15m", {}).get("current_price")
                or strategies.get("current_price")
                or 0.0
            )

            analysis = data.get("analysis") or {}
            action = (analysis.get("header") or {}).get("action")
            if not action:
                if score >= 60:
                    action = "ACTIVE_TRADE"
                elif score >= 45:
                    action = "CONDITIONAL_ENTRY"
                else:
                    action = "HOLD"

            history.append({
                "filename": filename,
                "symbol": symbol,
                "timestamp": timestamp,
                "bias": bias,
                "action": action,
                "score": score,
                "regime": regime,
                "price": price,
                "file_time": os.path.getmtime(file_path),
            })
        except Exception as e:
            print(f"[API] Error parsing history file {filename}: {e}")
            continue

    history.sort(key=lambda x: x["file_time"], reverse=True)
    return jsonify(history)


@app.route("/api/history/<filename>", methods=["GET"])
def get_snapshot(filename):
    filename = os.path.basename(filename)
    file_path = os.path.join(SNAPSHOTS_DIR, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "Snapshot file not found"}), 404
    try:
        with open(file_path, "r") as f:
            snapshot_data = json.load(f)
        return jsonify({
            "snapshot": snapshot_data,
            "analysis": snapshot_data.get("analysis"),
        })
    except Exception as e:
        return jsonify({"error": f"Failed to load snapshot: {e}"}), 500


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("=== CRYPTERA v3.1 WEB TERMINAL RUNNING ===")
    print("Open http://127.0.0.1:5000 in your web browser.")
    print("=" * 50 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=True)

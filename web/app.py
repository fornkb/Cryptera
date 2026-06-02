import os
import sys
import json
import glob
import asyncio
from flask import Flask, render_template, request, jsonify

# Add parent directory to system path so we can cleanly import our trading core
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

# Import the core analysis function from main
from main import run_analysis

app = Flask(__name__)

# Cache folder is the snapshots subdirectory where snapshots are stored
SNAPSHOTS_DIR = os.path.join(PARENT_DIR, "snapshots")
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)


@app.route("/")
def index():
    """Renders the main single-page trading dashboard."""
    return render_template("index.html")


@app.route("/api/run", methods=["POST"])
def run():
    """
    Triggers the async Cryptera SMC/PA analysis pipeline for a specified coin.
    Returns the generated JSON snapshot and the trade plan narration.
    """
    data = request.get_json() or {}
    symbol = data.get("symbol", "SOL/USDT").strip()
    
    if not symbol:
        return jsonify({"error": "Symbol is required"}), 400

    print(f"[API] Starting live market analysis for: {symbol}")
    
    try:
        # Run the async analysis synchronously inside Flask's thread using asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        snapshot, narration = loop.run_until_complete(run_analysis(symbol))
        loop.close()
        
        return jsonify({
            "success": True,
            "snapshot": snapshot,
            "narration": narration
        })
    except Exception as e:
        import traceback
        error_msg = str(e)
        print(f"[API] Error running analysis for {symbol}: {error_msg}")
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": error_msg
        }), 500


@app.route("/api/history", methods=["GET"])
def get_history():
    """
    Scans the directory for existing saved snapshot JSON files.
    Returns a sorted list of runs (most recent first) with summary metrics.
    """
    pattern = os.path.join(SNAPSHOTS_DIR, "snapshot_*.json")
    files = glob.glob(pattern)
    
    history = []
    for file_path in files:
        filename = os.path.basename(file_path)
        try:
            # Parse timestamp and symbol from filename e.g. snapshot_BTC_USDT_20260523_144717.json
            parts = filename.replace(".json", "").split("_")
            # snapshot, symbol_base, symbol_quote, date, time
            if len(parts) >= 5:
                symbol = f"{parts[1]}/{parts[2]}"
                timestamp = f"{parts[3][:4]}-{parts[3][4:6]}-{parts[3][6:8]} {parts[4][:2]}:{parts[4][2:4]}:{parts[4][4:6]}"
            else:
                symbol = "Unknown"
                timestamp = "Unknown"
                
            # Read a quick summary from the file to display in the UI history list
            with open(file_path, "r") as f:
                data = json.load(f)
                bias = data.get("strategies", {}).get("trend_bias", "NEUTRAL").upper()
                score = data.get("strategies", {}).get("confluence_score", 0)
                regime = data.get("market_regime", "Unknown")
                price = data.get("smc_context", {}).get("15m", {}).get("current_price", 0.0)
                
                # Determine action label based on new thresholds
                if score >= 60:
                    action_label = "TRADE READY"
                elif score >= 45:
                    action_label = "CONDITIONAL"
                else:
                    action_label = "HOLD"
                
            history.append({
                "filename": filename,
                "symbol": symbol,
                "timestamp": timestamp,
                "bias": bias,
                "action": action_label,
                "score": score,
                "regime": regime,
                "price": price,
                "file_time": os.path.getmtime(file_path)
            })
        except Exception as e:
            print(f"[API] Error parsing history file {filename}: {e}")
            continue
            
    # Sort history by file modification time (most recent first)
    history = sorted(history, key=lambda x: x["file_time"], reverse=True)
    return jsonify(history)


@app.route("/api/history/<filename>", methods=["GET"])
def get_snapshot(filename):
    """
    Retrieves and returns the full JSON payload of a specific saved snapshot file.
    """
    # Prevent directory traversal attacks
    filename = os.path.basename(filename)
    file_path = os.path.join(SNAPSHOTS_DIR, filename)
    
    if not os.path.exists(file_path):
        return jsonify({"error": "Snapshot file not found"}), 404
        
    try:
        with open(file_path, "r") as f:
            snapshot_data = json.load(f)
        return jsonify(snapshot_data)
    except Exception as e:
        return jsonify({"error": f"Failed to load snapshot: {str(e)}"}), 500


if __name__ == "__main__":
    print("\n" + "="*50)
    print("=== CRYPTERA GLASSMORPHIC WEB TERMINAL RUNNING ===")
    print("Open http://127.0.0.1:5000 in your web browser.")
    print("="*50 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=True)

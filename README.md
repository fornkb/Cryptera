# Cryptera v3.0

Crypto price prediction and trade advisory engine using Smart Money Concepts (SMC), Price Action (PA), and Gemini AI. Includes a CLI scanner and a Flask-based web interface.

## Prerequisites
* Python 3.10+
* ccxt
* pandas
* pandas-ta
* numpy
* requests
* python-dotenv
* google-generativeai
* Flask

## Installation
1. Clone repository:
   ```bash
   git clone https://github.com/fornkb/Cryptera.git
   cd Cryptera
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create `.env` file in the root directory:
   ```env
   GEMINI_API_KEY=your_gemini_api_key
   ```

## Usage

### Command Line Interface (CLI)
Run analysis on a specific coin (defaults to `SOL/USDT` if empty):

python main.py [SOL|BTC|ETH|LINK/USDT]

This runs indicators, computes market structure, dumps a JSON snapshot to `snapshots/`, and outputs a generated trade plan from Gemini.

### Web Dashboard
Start the Flask application:

python web/app.py

Open lh in your browser.

## File Mapping
* `main.py` : Entry point, ccxt data fetcher, and Gemini rest client.
* `smc.py`: Swings, BOS/CHoCH detection, Order Blocks (OB), and Fair Value Gaps (FVG).
* `price_action.py`: Volume profile (VAH/VAL/POC), candle patterns, S/R levels, and previous day statistics.
* `indicators.py`: Calculations for EMA, ATR, RSI, MACD, Supertrend, StochRSI, ADX, and CVD.
* `strategies.py`: Scoring engine (0-100 matrix) and trade parameter validation.
* `web/app.py`: Flask dashboard backend.
* `web/templates/index.html` & `web/static/`: Glassmorphic web frontend.

## Scoring System
Confluence score (max 100) is evaluated using:
1. **Trend Alignment (max 25)**: Directional agreement between 4H and 1H structures, scaled by 4H ADX.
2. **OB Proximity (max 15)**: Closeness to unfilled 15m/1H Order Blocks.
3. **Liquidity Sweep (10)**: Sweeping of key highs/lows with candle close confirmation.
4. **Momentum (max 15)**: RSI slope on 1H and MACD alignment.
5. **FVG Magnet (15)**: Unfilled FVG within 3% of price.
6. **OTE Bonus (10)**: Price in the 61.8%-78.6% Fib retracement zone.
7. **CVD Alignment (10)**: Delta sign alignment across timeframes.
8. **StochRSI Confirmation (5)**: Overbought/oversold trigger values.

* **Volume Gate (Execution Gate)**: Hard block if 15m relative volume < 0.1, low-volume warning if 0.1-0.3.

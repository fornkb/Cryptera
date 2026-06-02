# ⚡ Cryptera v3.0: Institutional Smart Money Concepts (SMC) Engine & Web Terminal

Cryptera is a state-of-the-art algorithmic trading and analysis engine designed to detect institutional market structures and generate precise trade plans. By combining rule-based quantitative finance logic—derived from **Inner Circle Trader (ICT)** and **Smart Money Concepts (SMC)**—with the advanced reasoning of **Gemini 2.5 Flash**, Cryptera bridges the gap between raw market mechanics and natural language trade execution plans.

The project features both a **high-speed CLI scanner** and a **premium glassmorphic Web Dashboard** that visualizes liquidity pools, value areas, and AI-generated narration in real time.

---

## 🚀 Key Features

* **Advanced Smart Money Mechanics (`smc.py`):** Automatically detects market structure changes, including **Break of Structure (BOS)** and **Change of Character (CHoCH)**. Locates unmitigated Bullish/Bearish **Order Blocks (OB)** and unfilled **Fair Value Gaps (FVG)** with dynamic mitigation checks.
* **Volume Profile & Price Action (`price_action.py`):** Calculates volume-based market levels including **Value Area High (VAH)**, **Value Area Low (VAL)**, and **Point of Control (POC)** using a 70% volume distribution, alongside Previous Day High (PDH), Low (PDL), and Close (PDC).
* **Multi-Timeframe Regime Detection (`indicators.py`):** Gauges the global market regime (Trending Bullish, Trending Bearish, Ranging) using a weighted score across HTF (4H), MTF (1H), and LTF (15m) timeframes.
* **Confluence Scoring Matrix (`strategies.py`):** Employs a strict 100-point scoring algorithm evaluating Trend Alignment, OB Proximity, Liquidity Sweeps, Momentum (RSI + MACD), FVG Magnets, Orderbook Skew, Funding Rates, and Optimal Trade Entry (OTE) zones.
* **HTTP/REST Gemini Narration (`main.py`):** Generates zero-hallucination market write-ups, stop loss/take profit levels (enforcing $\ge 1:2$ Risk-to-Reward ratio), and contingent if-then scenarios using Google's Gemini API over a stateless, leak-proof REST transport.
* **Glassmorphic Web Dashboard (`web/app.py`):** A high-end single-page application dashboard featuring a dynamic circular confluence ring, interactive liquidity grids, historical snapshot storage, and CVD absorption alerts.

---

## 📁 Repository Architecture

```text
├── main.py                 # Core analysis loop, ccxt data fetcher & Gemini client
├── smc.py                  # SMC math: Swings, BOS/CHoCH, Order Blocks, and FVGs
├── price_action.py         # Price Action: Value Areas (VAH/VAL/POC), PDH/PDL/PDC
├── indicators.py           # Technical Indicators (EMA, VWAP, ATR, RSI, MACD, OBV)
├── strategies.py           # Strategy evaluation, confluence calculation, & scoring
├── requirements.txt        # Project dependencies
├── .env.example            # Environment template file
├── .gitignore              # Git ignore rules (.env, __pycache__, snapshots/)
├── snapshots/              # [Auto-generated] Directory for JSON market snapshots
└── web/                    # Flask Web Terminal
    ├── app.py              # Flask server, route controller & local history indexer
    ├── templates/
    │   └── index.html      # Responsive glassmorphic layout
    └── static/
        ├── css/style.css   # Custom CSS styling (gradients, glow-effects, dark mode)
        └── js/app.js       # Core dashboard controller (WS rendering & AI parsing)
```

---

## 🛠️ Setup & Installation

### 1. Prerequisites
Make sure you have Python 3.10+ installed on your system.

### 2. Clone the Repository
```bash
git clone https://github.com/fornkb/Cryptera.git
cd Cryptera
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

---

## 💻 How to Use

### Run the CLI Scanner
You can run the engine directly from your terminal to analyze a specific asset (default is `SOL/USDT` if no argument is provided):
```bash
# Analyze Bitcoin
python main.py BTC

# Analyze Ethereum
python main.py ETH

# Analyze a Custom Ticker
python main.py LINK/USDT
```

This runs the mathematical analysis, prints the results with live current price debug logging to the console, saves a snapshot to `snapshots/`, and outputs a Gemini Trade Plan.

### Launch the Web Terminal
To start the glassmorphic trading dashboard:
```bash
cd web
python app.py
```
Open your browser and navigate to **`http://127.0.0.1:5000`**.

---

## 📈 Confluence Scoring System

Every setup is quantitatively evaluated before the AI narration phase. Points are distributed across the following factors (Max 100):

| Confluence Metric | Max Weight | Condition for Maximum Score |
| :--- | :---: | :--- |
| **Trend Alignment** | 20 pts | 4H structure, 1H EMA, and 15m price direction agree. |
| **OB Proximity** | 20 pts | Current price is within 0.5% of an unmitigated Order Block. |
| **Liquidity Sweep** | 20 pts | 15m candle swept swing highs/lows and closed inside. |
| **Momentum** | 15 pts | RSI is not overextended & MACD is aligned with the bias. |
| **FVG Magnet** | 15 pts | Price is within 1.0% of an unfilled Fair Value Gap. |
| **Orderbook & Funding** | 10 pts | Orderbook depth skew aligns and funding rate is neutral. |
| **OTE Zone Bonus** | 10 pts | Price resides inside the 0.618 - 0.786 Fibonacci retracement zone. |

---

## ⚠️ Disclaimer
*This project is for educational and research purposes only. Algorithmic and manual trading carry significant risk. None of the trade plans generated by this engine constitute financial advice.*

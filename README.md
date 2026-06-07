# Cryptera v3.2

A crypto price-prediction and trade-advisory engine built on multi-timeframe
**Smart Money Concepts (SMC)** and **Price Action (PA)**. A deterministic
rule engine scores confluence, proposes the trade geometry, and hands a compact
snapshot to **Gemini 3.5 Flash**, which returns a schema-validated JSON trade
plan. Everything is surfaced in a CLI and a glassmorphic Flask web terminal with
an interactive, level-annotated price chart.

The defining idea: **the rules decide, the LLM narrates.** The engine computes
the score, the gates and the entry/SL/TP itself — the model verifies and explains
within bounds, it does not invent levels.

---

## Highlights

- **Regime-conditional scoring.** Trending markets use a trend-continuation
  rubric (C1-C8); ranging markets switch to a mean-reversion rubric (M1-M6) that
  fades value-area extremes instead of chasing a non-existent trend.
- **Closed-candle signals.** The still-forming bar is dropped; indicators,
  structure and the volume gate run on closed candles only. `live_price` is
  exposed separately for distance/trigger context. No more intra-bar flicker.
- **Real order-flow CVD.** Cumulative volume delta is built from real
  taker-buy volume (`2·takerBuy − volume`), not a candle-shape proxy. A
  `cvd_is_real` flag marks the fallback case.
- **Range-distributed volume profile.** VAH/VAL/POC spread each candle's volume
  across its high-low range rather than dumping it at the close.
- **Order flow in the score.** L2 depth imbalance, funding crowding, OI build
  and Fear & Greed extremes apply a bounded ±8 modifier on top of the base score.
- **Deterministic trade geometry.** The engine computes entry / SL / TP / R:R
  (`engine_trade`); the LLM refines within bounds.
- **Event-based BOS / CHoCH.** First-close-beyond with displacement, plus a
  `fresh` recency flag. Developing swings are an early-warning side channel that
  never corrupts structure.
- **Multi-timeframe liquidity sweeps.** 4H / 1H pools + PDH/PDL, not just 15m.
- **Calibration loop.** `tools/calibrate.py` fits per-component weights and a
  score → win-rate table from labelled outcomes; the engine loads it when present
  and runs at equal weights otherwise.
- **Interactive chart.** A `lightweight-charts` candlestick view (4H/1H/15M
  switcher) overlaid with engine entry/SL/TP, POC/VAH/VAL, PDH/PDL, BSL/SSL,
  nearest OB/FVG and untested POCs.

---

## Architecture

```
        ┌──────────────── main.py (pipeline) ────────────────┐
        │  Binance OHLCV + taker flow (ccxt + REST klines)   │
        │  REST: L2 depth · funding · OI · F&G · BTC.D        │
        │  drop forming candle → calculate_indicators()      │
        └───────────────┬───────────────────────┬────────────┘
                        │                       │
          indicators.py │                       │ smc.py / price_action.py
   EMA·ATR·RSI·MACD·    │                       │  swings · BOS/CHoCH · OB · FVG
   Supertrend·StochRSI· │                       │  liquidity · premium/discount
   ADX·BB·real CVD·     │                       │  volume profile · S/R · POCs
   fractal swings       │                       │
                        ▼                       ▼
                  ┌──────────────── strategies.py ───────────────┐
                  │  regime-conditional score (C1-C8 / M1-M6)    │
                  │  + order-flow modifier  + trade geometry     │
                  │  + calibration weights / empirical win-rate  │
                  └───────────────────┬──────────────────────────┘
                                      ▼
                       snapshot JSON  →  Gemini 3.5 Flash
                       (response_schema-enforced trade plan)
                                      ▼
                       persist snapshot + analysis + chart_series
                                      ▼
                       CLI stdout   |   Flask web terminal
```

Per run the engine fetches **4H / 1H / 15M** candles (200 / 200 / 300 bars),
drops the forming bar, computes indicators on a thread, builds the SMC and PA
contexts, scores confluence for the detected regime, computes deterministic
trade geometry, sends a ~11 KB snapshot to Gemini, then persists the snapshot +
the parsed analysis + compact chart OHLC.

---

## Prerequisites

- Python 3.10+
- Dependencies in `requirements.txt`: `ccxt`, `pandas`, `pandas-ta`, `numpy`,
  `requests`, `python-dotenv`, `google-genai`, `Flask`.
- A Gemini API key with access to `gemini-3.5-flash`.

## Installation

```bash
git clone https://github.com/fornkb/Cryptera.git
cd Cryptera
pip install -r requirements.txt
cp .env.example .env        # then edit .env and add your key
```

`.env`:

```env
GEMINI_API_KEY=your_gemini_api_key
```

## Usage

### CLI

```bash
python main.py [SOL|BTC|ETH|LINK/USDT]
```

Defaults to `SOL/USDT`. Writes a timestamped `snapshots/snapshot_*.json`
containing the deterministic snapshot, the parsed Gemini analysis and the chart
series, then prints the analysis JSON to stdout.

### Web terminal

```bash
python web/app.py        # http://127.0.0.1:5000
```

Run analysis on demand, browse past snapshots in the history panel, and review
each setup on the annotated price chart.

### Backtest & calibration

```bash
python -m tools.label_snapshots --horizon 24   # label realised TP/SL outcomes
python -m tools.eval_engine                     # per-bucket performance metrics
python -m tools.calibrate                       # fit weights + score→win-rate
```

See [tools/README.md](tools/README.md) for details.

---

## Scoring (regime-conditional, max 100)

The engine reads `market_regime` and selects a rubric, exposed as
`strategies.score_mode`.

**Trending — continuation (C1-C8):**

| component        | max | rule                                                           |
| ---------------- | --- | -------------------------------------------------------------- |
| C1 trend         | 25  | ADX-tiered when 4H/1H structure agrees; +10 base on conflict   |
| C2 OB proximity  | 15  | 0/5/10/15 by **ATR-scaled** distance to nearest 15m/1H OB/FVG edge |
| C3 sweep         | 10  | HTF sweep+reclaim (4H/1H/PDH-PDL), **wick-confirmed**; 6 if 15m-only |
| C4 momentum      | 15  | +10 if 1H RSI/MACD aligns with bias; ±5 RSI-slope adjustment   |
| C5 FVG magnet    | 15  | unfilled draw-on-liquidity FVG within **3× ATR** on 1H or 4H   |
| C6 OTE bonus     | 10  | `in_ote = true` on 1H or 4H                                    |
| C7 CVD alignment | 10  | 2+ TF real-CVD signs match bias; 0 if 15m opposes 4H           |
| C8 StochRSI      | 5   | overbought (sell) / oversold (buy) on 1H                       |

**Ranging — mean reversion (M1-M6), fades the value-area edge:**

| component           | max | rule                                                        |
| ------------------- | --- | ----------------------------------------------------------- |
| M1 edge distance    | 25  | proximity to VAH/VAL (the fade edge)                        |
| M2 edge sweep       | 15  | sweep of the edge / HTF pool                                |
| M3 CVD absorption   | 15  | CVD opposing the push into the edge                         |
| M4 StochRSI extreme | 15  | 1H StochRSI overbought (fade short) / oversold (fade long)  |
| M5 rejection        | 10  | reversal candle (pin/engulfing/star) in the fade direction  |
| M6 range intact     | 10  | no fresh 1H BOS/CHoCH (range still valid)                   |

**Order-flow modifier (both modes):** bounded ±8 from L2 depth imbalance
(de-weighted — L2 is spoofable), funding crowding + trajectory, OI build vs.
price, F&G extremes, and **BTC-dominance** macro headwind/tailwind for alts —
`final = clamp(base + modifier, 0, 100)`.

**Trade geometry (honest R:R):** the engine targets only real structural levels
(liquidity pools, untested POCs, anchored VWAP, value-area edges). For trend
trades it picks the nearest target that satisfies 2R, else the farthest real
level with an honest sub-2R `rr` — it **never invents a TP** to force 2R. If no
structural target exists, `engine_trade.valid = false` and the action is HOLD.

**Decision gates:**

- Thresholds: `>= 60` ACTIVE_TRADE · `45-59` CONDITIONAL_ENTRY · `< 45` HOLD.
- Volume gate: `HARD_GATE` (15m rel-vol < 0.1) suspends execution;
  `LOW_VOL_WARNING` (0.1-0.3) halves size.
- Event guard: when a high-impact event is within ±2h, action is capped at
  CONDITIONAL_ENTRY.

---

## Snapshot schema (`schema_version = 3.2.1`)

Top-level keys written to each `snapshots/snapshot_*.json`:

| key                   | contents                                                                 |
| --------------------- | ------------------------------------------------------------------------ |
| `live_price`          | freshest price (forming bar); signals themselves use closed bars         |
| `market_regime`       | Trending Bullish / Trending Bearish / Ranging / Sideways                 |
| `volatility_regime`   | per-TF compressed / normal / expanded (ATR percentile + BB width)        |
| `event_guard`         | active macro-event window state                                          |
| `previous_day`        | PDH / PDL / PDC                                                           |
| `orderbook`           | L2 depth bins (±0.25/0.5/1/2%) + per-band imbalance + skew               |
| `funding`             | current / 24h avg / trend / percentile-within-window                     |
| `open_interest`       | value + 1h/4h delta + 30-day percentile                                  |
| `btc_dominance_proxy` | BTCDOM/USDT 4h/24h change (non-BTC symbols only)                         |
| `smc_context`         | per-TF structure, BOS/CHoCH (fresh+displacement), OB, FVG, BSL/SSL, P/D, OTE, dealing range, developing swings |
| `price_action`        | per-TF value area, S/R, PDH/PDL/PDC, untested 4H POCs, anchored VWAP, candle pattern |
| `strategies`          | score_mode, score + breakdown + notes, order-flow modifier, volume gate, `engine_trade`, empirical win-rate |
| `windowed_indicators` | per-TF compressed indicator summary (incl. `cvd_is_real`)                |
| `analysis`            | the parsed Gemini JSON trade plan                                        |
| `chart_series`        | compact OHLC per TF for the chart — added **after** the LLM call         |

---

## File map

| path                        | role                                                                              |
| --------------------------- | --------------------------------------------------------------------------------- |
| `main.py`                   | pipeline · ccxt + taker-flow fetch · closed-candle handling · Gemini client · persistence |
| `indicators.py`             | EMA/ATR/RSI/MACD/Supertrend/StochRSI/ADX/BB · real-or-proxy CVD · volatility regime · shared fractal swings |
| `smc.py`                    | unified fractal swings · event-based BOS/CHoCH+displacement · OBs · FVGs · BSL/SSL · single-leg premium/discount + OTE |
| `price_action.py`           | confirmation candle patterns · range-distributed volume profile · ATR-scaled S/R · PDH/PDL/PDC · untested session POCs |
| `strategies.py`             | regime-conditional scoring (C1-C8 / M1-M6) · order-flow modifier · deterministic trade geometry · calibration loader |
| `web/app.py`                | Flask backend: `/api/run`, `/api/history`, `/api/history/<filename>`             |
| `web/templates/index.html`  | dashboard markup + price-chart container                                          |
| `web/static/js/app.js`      | renders the structured analysis + chart overlays (no regex parsing)              |
| `web/static/css/style.css`  | glassmorphic styling                                                              |
| `data/events.json`          | operator-maintained list of high-impact macro events (tracked)                   |
| `data/oi_history.json`      | engine-managed OI history fallback (git-ignored)                                  |
| `data/calibration.json`     | optional fitted weights + score→win-rate (git-ignored)                           |
| `tools/label_snapshots.py`  | label realised TP/SL outcomes (LLM plan, forward scenario, engine geometry)      |
| `tools/eval_engine.py`      | aggregate per-bucket performance metrics                                          |
| `tools/calibrate.py`        | fit component weights + score→win-rate from labelled snapshots                    |
| `improvements_report.md`    | production-grade engine review (Tier 1-3, bugs, removals)                         |

---

## Roadmap

Tier 1 of [improvements_report.md](improvements_report.md), all bugs, and all
removals are implemented in 3.2. Still open:

- **Tier 2** — closed-candle volume-gate seasonality, BTC-beta gating for alts,
  funding-fade directional bias, naked-POC target wiring, probability calibration
  display.
- **Tier 3** — real liquidation feed, options gamma / max-pain, WebSocket
  trade-tape, ML auxiliary probability, on-chain/news context, RL weight tuning.

---

## Changelog

**3.2.1** — honest trade geometry (never inflates TP to force 2R — targets real
levels only, reports sub-2R R:R, or HOLDs when no structural target); ATR-scaled
C2/C5 proximity tiers; BTC-dominance macro gate + funding-trajectory term in the
order-flow modifier (L2 depth de-weighted); wick-confirmed liquidity sweeps;
anchored VWAP (from last swing high/low) in the snapshot, geometry targets, and
chart overlay.

**3.2** — regime-conditional scoring (trend/mean-revert); closed-candle signals;
real taker-flow CVD; range-distributed volume profile; order flow in the score;
deterministic trade geometry; event-based BOS/CHoCH; MTF liquidity sweeps;
unified fractal swing engine; calibration loop; interactive level-annotated
chart; regime-aware dashboard.

**3.1** — C1-C8 rubric aligned to the LLM; broken liquidations call removed;
structured-JSON output via `response_schema` on Gemini 3.5 Flash (`thinking_level`,
deprecated sampling knobs dropped); ATR-normalised thresholds; L2 depth bins; OI
delta + percentile; funding trajectory; volatility regime; BTC.D proxy;
event-window guard; schema versioning; LRU cache; snapshot retention; backtest
harness.

**3.0** — initial rule-based SMC engine + Gemini narration + Flask dashboard.

---

## Disclaimer

Cryptera is research/educational tooling, not financial advice. It does not place
orders or move funds. Trade at your own risk.

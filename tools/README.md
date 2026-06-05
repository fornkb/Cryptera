# tools/

Offline analytics on the saved snapshots.

## `label_snapshots.py`

Walks `snapshots/` and re-fetches the 15-minute candles that followed each
snapshot timestamp. Marks the `trade_decision` (and `forward_scenario` when
present) as:

* `triggered` / `not_triggered`
* `tp_first` / `sl_first` / `timeout`
* `mfe_pct`, `mae_pct`, `realized_rr`

Writes the result back into the snapshot under `outcome`. Only labels
snapshots whose horizon window has fully elapsed.

```bash
python -m tools.label_snapshots                      # everything
python -m tools.label_snapshots --symbol BTC/USDT
python -m tools.label_snapshots --horizon 48 --force # re-label with a longer horizon
```

## `eval_engine.py`

Aggregates labelled snapshots into win-rate, trigger-rate, average MFE/MAE,
average realised R:R and a naive expectancy per bucket:

* score (`<45`, `45-59`, `60-74`, `75+`)
* trend bias
* volatility regime (15m)
* event-guard state
* volume gate state
* recommended action

```bash
python -m tools.eval_engine
python -m tools.eval_engine --symbol BTC/USDT --out reports/eval_btc.json
```

Use this output to retune the C1-C8 weights and confluence thresholds against
ground truth instead of intuition.

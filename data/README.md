# data/

Runtime state and operator-maintained inputs used by the engine.

| file               | role                                                                                    |
| ------------------ | --------------------------------------------------------------------------------------- |
| `events.json`      | Operator-maintained list of upcoming high-impact macro events. The engine matches each entry's `datetime` (ISO 8601, local-naive treated as UTC) against the current time and, if within `±2h`, sets `event_guard.active = true` in the snapshot. The LLM is instructed to cap the action at `CONDITIONAL_ENTRY` while a window is active. |
| `oi_history.json`  | Auto-managed. The engine appends an `{ts, oi}` record per symbol on every run and keeps the last 2000 points as a local fallback for the 30-day OI percentile when the Binance `openInterestHist` endpoint is unavailable. |

Update `events.json` before known macro releases. Both fields are optional;
leaving the file empty (`[]`) disables the guard.

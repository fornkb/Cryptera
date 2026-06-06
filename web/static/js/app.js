/*
 * Cryptera v3.2 dashboard controller.
 *
 * Consumes the structured Gemini JSON analysis AND the deterministic snapshot.
 * Regime-aware: renders the C1-C8 (trend) or M1-M6 (mean-revert) breakdown
 * depending on strategies.score_mode. Surfaces engine_trade geometry, the
 * order-flow modifier, empirical win-rate, BOS/CHoCH quality, untested POCs and
 * the real-vs-proxy CVD flag.
 */

document.addEventListener("DOMContentLoaded", () => {
    const $ = (id) => document.getElementById(id);

    // ---- component metadata (label + max) per score mode ----
    const TREND_COMPONENTS = [
        ["c1_trend_alignment", "C1 Trend", 25],
        ["c2_ob_proximity", "C2 OB Prox", 15],
        ["c3_liquidity_sweep", "C3 Sweep", 10],
        ["c4_momentum", "C4 Momentum", 15],
        ["c5_fvg_magnet", "C5 FVG Magnet", 15],
        ["c6_ote_bonus", "C6 OTE", 10],
        ["c7_cvd_alignment", "C7 CVD", 10],
        ["c8_stochrsi", "C8 StochRSI", 5],
    ];
    const MR_COMPONENTS = [
        ["m1_edge_distance", "M1 Edge Dist", 25],
        ["m2_edge_sweep", "M2 Edge Sweep", 15],
        ["m3_cvd_absorption", "M3 CVD Absorp", 15],
        ["m4_stoch_extreme", "M4 Stoch Extr", 15],
        ["m5_rejection", "M5 Rejection", 10],
        ["m6_range_intact", "M6 Range OK", 10],
    ];
    // legacy key fallbacks for old snapshots
    const LEGACY = {
        c1_trend_alignment: "trend_alignment", c2_ob_proximity: "ob_proximity",
        c3_liquidity_sweep: "liquidity_sweep", c4_momentum: "momentum",
        c5_fvg_magnet: "fvg_magnet", c6_ote_bonus: "ote_bonus",
        c7_cvd_alignment: "cvd_divergence",
    };

    // ---- DOM refs ----
    const symbolSelector = $("symbol-selector");
    const customSymbolGroup = $("custom-symbol-group");
    const customSymbolInput = $("custom-symbol-input");
    const btnRunAnalysis = $("btn-run-analysis");
    const btnRefreshHistory = $("btn-refresh-history");
    const historyList = $("history-list");
    const historyListLoading = $("history-list-loading");
    const historyListEmpty = $("history-list-empty");

    const workspaceWelcome = $("workspace-welcome");
    const workspaceAnalysis = $("workspace-analysis");

    const valMarketRegime = $("val-market-regime");
    const valFearGreed = $("val-fear-greed");
    const valFundingRate = $("val-funding-rate");
    const valSupertrend = $("val-supertrend");
    const valVolRegime = $("val-vol-regime");
    const valEventGuard = $("val-event-guard");

    const valConfluenceScore = $("val-confluence-score");
    const confluenceScoreRing = $("confluence-score-ring");
    const valLivePrice = $("val-live-price");
    const badgeScoreMode = $("badge-score-mode");
    const badgeBias = $("badge-bias");
    const badgeAction = $("badge-action");
    const badgeVolumeGate = $("badge-volume-gate");
    const valScoreMath = $("val-score-math");
    const valWinrate = $("val-winrate");
    const planSourceLabel = $("plan-source-label");
    const srcEntry = $("src-entry");
    const srcSl = $("src-sl");
    const srcTp = $("src-tp");
    const valEntry = $("val-entry");
    const valStopLoss = $("val-stop-loss");
    const valTakeProfit = $("val-take-profit");
    const valRR = $("val-rr");
    const valOrderFlow = $("val-order-flow");

    const scoreBreakdownTitle = $("score-breakdown-title");
    const scoreBreakdownGrid = $("score-breakdown-grid");

    const vaValLabel = $("va-val-label");
    const vaVahLabel = $("va-vah-label");
    const vaPocVal = $("va-poc-val");
    const valObContainer = $("val-ob-container");
    const valFvgContainer = $("val-fvg-container");
    const valSrContainer = $("val-sr-container");
    const valCvdContainer = $("val-cvd-container");
    const valDepthContainer = $("val-depth-container");
    const valDerivsContainer = $("val-derivs-container");
    const valStructureContainer = $("val-structure-container");
    const valTargetsContainer = $("val-targets-container");

    const valMarketNarrative = $("val-market-narrative");
    const valPrimaryDraw = $("val-primary-draw");

    const hypoDirection = $("hypo-direction");
    const hypoTrigger = $("hypo-trigger");
    const hypoEntry = $("hypo-entry");
    const hypoSl = $("hypo-sl");
    const hypoTp = $("hypo-tp");
    const hypoRR = $("hypo-rr");
    const hypoVolume = $("hypo-volume");
    const hypoEvidence = $("hypo-evidence");

    const reasoningStruct = $("reasoning-struct");
    const reasoningLiq = $("reasoning-liq");
    const reasoningMom = $("reasoning-mom");
    const reasoningBook = $("reasoning-book");

    let activeFilename = null;
    let priceChart = null;
    let chartTf = "15m";
    let lastSnapshot = null;

    // ---- helpers ----
    const fmtNum = (v, d = 2) => (v === null || v === undefined || Number.isNaN(Number(v))) ? "—" : Number(v).toFixed(d);
    const fmtSigned = (v, d = 2) => {
        if (v === null || v === undefined || Number.isNaN(Number(v))) return "—";
        const n = Number(v);
        return `${n > 0 ? "+" : ""}${n.toFixed(d)}`;
    };
    const safeText = (el, v, fb = "—") => { if (el) el.textContent = (v === undefined || v === null || v === "") ? fb : String(v); };
    const setBadge = (el, label, bg) => { if (el) { el.textContent = label; el.className = `badge ${bg || "bg-neutral"}`; } };

    // ---- history ----
    const refreshHistory = async () => {
        historyListLoading.classList.remove("hidden");
        historyListEmpty.classList.add("hidden");
        historyList.innerHTML = "";
        try {
            const res = await fetch("/api/history");
            const data = await res.json();
            historyListLoading.classList.add("hidden");
            if (!Array.isArray(data) || data.length === 0) {
                historyListEmpty.classList.remove("hidden");
                return;
            }
            data.forEach(item => {
                const li = document.createElement("li");
                li.className = `history-item ${activeFilename === item.filename ? "active" : ""}`;
                li.setAttribute("data-filename", item.filename);
                const biasClass = item.bias === "BULLISH" ? "text-green" : item.bias === "BEARISH" ? "text-red" : "";
                const actionLabel = item.action || "HOLD";
                const actionBadgeClass = actionLabel === "ACTIVE_TRADE" ? "bg-bullish"
                    : actionLabel === "CONDITIONAL_ENTRY" ? "bg-warning" : "bg-hold";
                li.innerHTML = `
                    <div class="history-item-top">
                        <span class="history-item-symbol">${item.symbol}</span>
                        <span class="history-item-time">${(item.timestamp || "").substring(5, 16)}</span>
                    </div>
                    <div class="history-item-bottom">
                        <span class="history-item-meta">Bias: <span class="${biasClass}">${item.bias}</span> | Score: <span>${item.score}</span></span>
                        <span class="badge ${actionBadgeClass}" style="padding:1px 6px;font-size:8px;">${actionLabel.replace(/_/g, " ")}</span>
                    </div>`;
                li.addEventListener("click", () => {
                    document.querySelectorAll(".history-item").forEach(el => el.classList.remove("active"));
                    li.classList.add("active");
                    loadSnapshot(item.filename);
                });
                historyList.appendChild(li);
            });
        } catch (err) {
            console.error("Failed to load history list:", err);
            historyListLoading.classList.add("hidden");
            historyListEmpty.classList.remove("hidden");
            historyListEmpty.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> Error loading history`;
        }
    };

    const loadSnapshot = async (filename) => {
        activeFilename = filename;
        try {
            const res = await fetch(`/api/history/${filename}`);
            const payload = await res.json();
            const snapshot = payload.snapshot || payload;
            const analysis = payload.analysis || (snapshot && snapshot.analysis) || {};
            renderWorkspace(snapshot, analysis);
        } catch (err) {
            console.error("Failed to load snapshot details:", err);
            alert("Failed to load snapshot details.");
        }
    };

    // ---- main render ----
    const renderWorkspace = (snapshot, analysis) => {
        workspaceWelcome.classList.add("hidden");
        workspaceAnalysis.classList.remove("hidden");

        analysis = analysis || {};
        const header = analysis.header || {};
        const narrative = analysis.narrative || {};
        const tradeDecision = analysis.trade_decision || {};
        const reasoning = tradeDecision.reasoning || {};
        const forwardScenario = analysis.forward_scenario || {};
        const strategies = snapshot.strategies || {};
        const scoreMode = strategies.score_mode || "trend";
        const engine = snapshot.engine_trade || strategies.engine_trade || {};

        // -- top dials --
        safeText(valMarketRegime, snapshot.market_regime, "RANGING / SIDEWAYS");
        valMarketRegime.className = "";
        if (snapshot.market_regime === "Trending Bullish") valMarketRegime.classList.add("text-green");
        else if (snapshot.market_regime === "Trending Bearish") valMarketRegime.classList.add("text-red");

        safeText(valFearGreed, snapshot.fear_greed_index != null ? `${snapshot.fear_greed_index} / 100` : "—", "—");

        const funding = snapshot.funding || {};
        valFundingRate.textContent = (funding.current != null)
            ? `${(funding.current * 100).toFixed(4)}% / ${funding.trend ? funding.trend.toUpperCase() : "—"}`
            : "—";

        const st = snapshot.supertrend || {};
        const stDir = (st.direction || "—").toUpperCase();
        valSupertrend.textContent = `${stDir} // ${st.level != null ? st.level.toFixed(2) : "0.00"}`;
        valSupertrend.className = stDir === "BULLISH" ? "text-green" : stDir === "BEARISH" ? "text-red" : "";

        const vr = snapshot.volatility_regime || {};
        const tag = (k) => (vr[k] && vr[k].regime) ? vr[k].regime.toUpperCase() : "—";
        safeText(valVolRegime, `${tag("4h")} / ${tag("1h")} / ${tag("15m")}`);

        const event = snapshot.event_guard || {};
        if (event.active && Array.isArray(event.events) && event.events.length) {
            const e = event.events[0];
            valEventGuard.textContent = `${e.name} (${e.minutes_until != null ? e.minutes_until.toFixed(0) : "?"} min)`;
            valEventGuard.className = "text-red";
        } else {
            valEventGuard.textContent = "NO ACTIVE WINDOW";
            valEventGuard.className = "";
        }

        // -- score ring (deterministic score is authoritative) --
        const score = strategies.confluence_score != null ? strategies.confluence_score : (header.score || 0);
        valConfluenceScore.textContent = score;
        confluenceScoreRing.style.strokeDashoffset = 439.8 - (439.8 * score / 100);
        confluenceScoreRing.style.stroke = score >= 60 ? "#10b981" : score >= 45 ? "#f59e0b" : "#ef4444";

        safeText(valLivePrice, snapshot.live_price != null ? fmtNum(snapshot.live_price, 4) : "—");

        // -- score mode badge --
        setBadge(badgeScoreMode, scoreMode === "mean_revert" ? "MEAN-REVERT (fade)" : "TREND (continuation)",
            scoreMode === "mean_revert" ? "bg-mode-mr" : "bg-mode-trend");

        // -- setup direction (the actual trade dir; differs from 4H trend in MR mode) --
        const sd = (strategies.setup_direction || "neutral").toLowerCase();
        const sdLabel = sd === "bullish" ? "LONG / BUY" : sd === "bearish" ? "SHORT / SELL" : "NEUTRAL";
        const sdClass = sd === "bullish" ? "bg-bullish" : sd === "bearish" ? "bg-bearish" : "bg-neutral";
        setBadge(badgeBias, sdLabel, sdClass);
        badgeBias.title = `4H trend bias: ${(strategies.trend_bias || "neutral").toUpperCase()}`;

        // -- action --
        const action = header.action || (score >= 60 ? "ACTIVE_TRADE" : score >= 45 ? "CONDITIONAL_ENTRY" : "HOLD");
        const actionClass = action === "ACTIVE_TRADE" ? "bg-bullish" : action === "CONDITIONAL_ENTRY" ? "bg-warning" : "bg-hold";
        setBadge(badgeAction, action.replace(/_/g, " "), actionClass);

        // -- volume gate --
        const gateState = (strategies.volume_gate && strategies.volume_gate.state) || header.volume_gate || "CLEAR";
        const gateClass = gateState === "HARD_GATE" ? "bg-bearish" : gateState === "LOW_VOL_WARNING" ? "bg-warning" : "bg-bullish";
        setBadge(badgeVolumeGate, gateState.replace(/_/g, " "), gateClass);

        // -- score math: base ± order-flow = final --
        const base = strategies.base_score;
        const mod = strategies.order_flow_modifier;
        if (base != null && mod != null) {
            valScoreMath.textContent = `${base} ${mod >= 0 ? "+" : "−"}${Math.abs(mod)} = ${score}`;
            valScoreMath.className = `metric-value font-mono ${mod > 0 ? "text-green" : mod < 0 ? "text-red" : ""}`;
        } else {
            valScoreMath.textContent = `${score}`;
            valScoreMath.className = "metric-value font-mono";
        }

        // -- empirical win-rate --
        if (strategies.empirical_win_rate != null) {
            valWinrate.textContent = `${(strategies.empirical_win_rate * 100).toFixed(0)}% (hist)`;
            valWinrate.className = "metric-value font-mono text-green";
        } else {
            valWinrate.textContent = "— (uncalibrated)";
            valWinrate.className = "metric-value font-mono";
        }

        // -- trade plan: prefer deterministic engine_trade, fall back to LLM --
        renderTradePlan(engine, tradeDecision);

        // -- order-flow notes --
        const ofNotes = strategies.order_flow_notes || [];
        valOrderFlow.textContent = ofNotes.length ? ofNotes.join("; ") : "—";
        valOrderFlow.title = ofNotes.join("\n");

        // -- score breakdown (regime-conditional) --
        renderBreakdown(strategies, scoreMode);

        // -- value area --
        const pa = snapshot.price_action || {};
        const va = pa.value_area_1h || {};
        vaValLabel.textContent = fmtNum(va.val || 0);
        vaVahLabel.textContent = fmtNum(va.vah || 0);
        vaPocVal.textContent = fmtNum(va.poc || 0);
        const price = (snapshot.smc_context && snapshot.smc_context["15m"] && snapshot.smc_context["15m"].current_price)
            || snapshot.live_price || strategies.current_price || va.poc || 0;
        const span = (va.vah || 0) - (va.val || 0);
        if (span > 0) {
            const clamped = Math.min(Math.max(((price - va.val) / span) * 100, 0), 100);
            const vaAreaEl = document.querySelector(".va-area");
            if (vaAreaEl) {
                vaAreaEl.style.left = `${Math.max(clamped - 15, 0)}%`;
                vaAreaEl.style.right = `${Math.max(100 - clamped - 15, 0)}%`;
            }
        }

        // -- microstructure panels --
        renderOBs(snapshot, valObContainer);
        renderFVGs(snapshot, valFvgContainer);
        renderSR(snapshot, valSrContainer);
        renderCVD(snapshot, valCvdContainer, strategies);
        renderDepth(snapshot, valDepthContainer);
        renderDerivatives(snapshot, valDerivsContainer);
        renderStructure(snapshot, valStructureContainer);
        renderTargets(snapshot, valTargetsContainer);

        // -- price chart with SMC level overlays --
        lastSnapshot = snapshot;
        renderChart(snapshot);

        // -- narrative + forward scenario --
        safeText(valMarketNarrative, narrative.summary, "No narrative generated yet.");
        safeText(valPrimaryDraw, narrative.primary_draw, "—");

        const fwdDir = forwardScenario.direction || "NEUTRAL";
        hypoDirection.textContent = fwdDir;
        hypoDirection.className = `hypo-value ${fwdDir === "LONG" ? "text-green" : fwdDir === "SHORT" ? "text-red" : ""}`;
        safeText(hypoTrigger, forwardScenario.trigger, "Awaiting trigger.");
        safeText(hypoEntry, forwardScenario.entry != null ? fmtNum(forwardScenario.entry) : "—");
        safeText(hypoSl, forwardScenario.stop_loss != null ? fmtNum(forwardScenario.stop_loss) : "—");
        safeText(hypoTp, forwardScenario.take_profit != null ? fmtNum(forwardScenario.take_profit) : "—");
        safeText(hypoRR, forwardScenario.rr != null ? `${Number(forwardScenario.rr).toFixed(2)}:1` : "—");
        safeText(hypoVolume, forwardScenario.volume_condition, "—");
        safeText(hypoEvidence, forwardScenario.supporting_confluence, "—");

        safeText(reasoningStruct, reasoning.structure, "—");
        safeText(reasoningLiq, reasoning.liquidity, "—");
        safeText(reasoningMom, reasoning.momentum, "—");
        safeText(reasoningBook, reasoning.sentiment, "—");
    };

    // ---- trade plan: engine geometry primary, LLM fallback ----
    const renderTradePlan = (engine, tradeDecision) => {
        const useEngine = engine && engine.valid && engine.entry != null;
        if (useEngine) {
            planSourceLabel.textContent = "ENGINE TRADE PLAN (deterministic)";
            valEntry.textContent = fmtNum(engine.entry);
            valStopLoss.textContent = fmtNum(engine.stop_loss);
            valTakeProfit.textContent = fmtNum(engine.take_profit);
            safeText(srcEntry, engine.entry_source ? `· ${engine.entry_source}` : "", "");
            safeText(srcSl, engine.stop_loss_source ? `· ${engine.stop_loss_source}` : "", "");
            safeText(srcTp, engine.take_profit_source ? `· ${engine.take_profit_source}` : "", "");
            if (engine.rr != null) {
                valRR.textContent = `${Number(engine.rr).toFixed(2)}:1 ${engine.rr_passed ? "✓" : "✗"}`;
                valRR.className = `metric-value font-mono ${engine.rr_passed ? "text-green" : "text-red"}`;
            } else {
                valRR.textContent = "—";
                valRR.className = "metric-value font-mono";
            }
        } else {
            planSourceLabel.textContent = "AI TRADE PLAN";
            const td = tradeDecision || {};
            valEntry.textContent = td.entry != null ? fmtNum(td.entry) : "N/A";
            valStopLoss.textContent = td.stop_loss != null ? fmtNum(td.stop_loss) : "N/A";
            valTakeProfit.textContent = td.take_profit != null ? fmtNum(td.take_profit) : "N/A";
            safeText(srcEntry, td.entry_source ? `· ${td.entry_source}` : "", "");
            safeText(srcSl, td.stop_loss_source ? `· ${td.stop_loss_source}` : "", "");
            safeText(srcTp, td.take_profit_source ? `· ${td.take_profit_source}` : "", "");
            const rr = td.rr || {};
            if (rr.ratio != null) {
                valRR.textContent = `${Number(rr.ratio).toFixed(2)}:1 ${rr.passed ? "✓" : "✗"}`;
                valRR.className = `metric-value font-mono ${rr.passed ? "text-green" : "text-red"}`;
            } else {
                valRR.textContent = "—";
                valRR.className = "metric-value font-mono";
            }
        }
    };

    // ---- regime-conditional score breakdown ----
    const renderBreakdown = (strategies, scoreMode) => {
        const components = scoreMode === "mean_revert" ? MR_COMPONENTS : TREND_COMPONENTS;
        const bd = strategies.confluence_breakdown || {};
        const notes = strategies.confluence_notes || {};
        scoreBreakdownTitle.textContent = scoreMode === "mean_revert"
            ? "MEAN-REVERSION BREAKDOWN (M1-M6)" : "TREND BREAKDOWN (C1-C8)";
        scoreBreakdownGrid.innerHTML = "";
        components.forEach(([key, label, max]) => {
            let val = bd[key];
            if (val == null && LEGACY[key] != null) val = bd[LEGACY[key]];
            val = val == null ? 0 : val;
            const note = notes[key] || (LEGACY[key] && notes[LEGACY[key]]) || "";
            const pct = max ? Math.min(100, Math.round((val / max) * 100)) : 0;
            const cls = pct >= 66 ? "bar-strong" : pct >= 33 ? "bar-mid" : "bar-weak";
            const item = document.createElement("div");
            item.className = "breakdown-item";
            item.title = note;
            item.innerHTML = `
                <span class="item-lbl">${label} (${max})</span>
                <span class="item-val font-mono">${val}</span>
                <div class="item-bar"><div class="item-bar-fill ${cls}" style="width:${pct}%"></div></div>`;
            scoreBreakdownGrid.appendChild(item);
        });
    };

    // ---- microstructure renderers ----
    const renderOBs = (snapshot, container) => {
        container.innerHTML = "";
        const ul = document.createElement("ul");
        ul.className = "levels-ul";
        let any = false;
        ["4h", "1h", "15m"].forEach(tf => {
            const obs = (snapshot.smc_context || {})[tf] && (snapshot.smc_context[tf].order_blocks || {});
            if (!obs) return;
            (obs.bullish || []).forEach(ob => {
                any = true;
                const hvn = ob.high_volume_node ? ' <span class="hvn-tag">HVN</span>' : "";
                ul.innerHTML += `<li class="level-li-item"><span class="text-green">${tf.toUpperCase()} BULL OB${hvn}</span><span class="level-lbl-sub">[${fmtNum(ob.bottom)} - ${fmtNum(ob.top)}]</span></li>`;
            });
            (obs.bearish || []).forEach(ob => {
                any = true;
                const hvn = ob.high_volume_node ? ' <span class="hvn-tag">HVN</span>' : "";
                ul.innerHTML += `<li class="level-li-item"><span class="text-red">${tf.toUpperCase()} BEAR OB${hvn}</span><span class="level-lbl-sub">[${fmtNum(ob.bottom)} - ${fmtNum(ob.top)}]</span></li>`;
            });
        });
        if (!any) container.innerHTML = `<div class="box-message">No active unmitigated OBs in lookback.</div>`;
        else container.appendChild(ul);
    };

    const renderFVGs = (snapshot, container) => {
        container.innerHTML = "";
        const ul = document.createElement("ul");
        ul.className = "levels-ul";
        let any = false;
        ["4h", "1h", "15m"].forEach(tf => {
            const fvgs = (snapshot.smc_context || {})[tf] && (snapshot.smc_context[tf].fvg || {});
            if (!fvgs) return;
            const b = fvgs.nearest_bullish_fvg, r = fvgs.nearest_bearish_fvg;
            if (b) { any = true; ul.innerHTML += `<li class="level-li-item"><span class="text-green">${tf.toUpperCase()} BULL FVG</span><span class="level-lbl-sub">[${fmtNum(b.bottom)} - ${fmtNum(b.top)}]</span></li>`; }
            if (r) { any = true; ul.innerHTML += `<li class="level-li-item"><span class="text-red">${tf.toUpperCase()} BEAR FVG</span><span class="level-lbl-sub">[${fmtNum(r.bottom)} - ${fmtNum(r.top)}]</span></li>`; }
        });
        if (!any) container.innerHTML = `<div class="box-message">No active unfilled FVGs in range.</div>`;
        else container.appendChild(ul);
    };

    const renderSR = (snapshot, container) => {
        container.innerHTML = "";
        const ul = document.createElement("ul");
        ul.className = "levels-ul";
        let any = false;
        ["4h", "1h", "15m"].forEach(tf => {
            const liq = (snapshot.smc_context || {})[tf] && (snapshot.smc_context[tf].liquidity_levels || {});
            if (!liq) return;
            (liq.sell_side || []).forEach(lvl => { any = true; ul.innerHTML += `<li class="level-li-item"><span class="text-green">${tf.toUpperCase()} SSL</span><span class="font-mono">${fmtNum(lvl)}</span></li>`; });
            (liq.buy_side || []).forEach(lvl => { any = true; ul.innerHTML += `<li class="level-li-item"><span class="text-red">${tf.toUpperCase()} BSL</span><span class="font-mono">${fmtNum(lvl)}</span></li>`; });
        });
        if (!any) container.innerHTML = `<div class="box-message">No historical S/R cluster zones.</div>`;
        else container.appendChild(ul);
    };

    const renderCVD = (snapshot, container, strategies) => {
        container.innerHTML = "";
        const wi = snapshot.windowed_indicators || {};
        const isReal = wi["15m"] && wi["15m"].cvd_is_real;
        const absorption = (strategies && strategies.cvd_absorption_warning) || false;
        const bd = (strategies && strategies.confluence_breakdown) || {};
        const c7 = bd.c7_cvd_alignment != null ? bd.c7_cvd_alignment : bd.m3_cvd_absorption;
        const fmtDelta = (v) => v == null ? "—" : fmtSigned(v, 2);

        const div = document.createElement("div");
        div.style.cssText = "padding:8px;font-size:11px;line-height:1.45;";
        const badge = isReal
            ? '<span class="cvd-real-badge">REAL taker flow</span>'
            : '<span class="cvd-proxy-badge">proxy (candle-pos)</span>';
        div.innerHTML = `
            <div style="margin-bottom:6px;">${badge}</div>
            <div>4H Δ: <span class="font-mono">${fmtDelta(wi["4h"] && wi["4h"].cvd_window_delta)}</span></div>
            <div>1H Δ: <span class="font-mono">${fmtDelta(wi["1h"] && wi["1h"].cvd_window_delta)}</span></div>
            <div>15M Δ: <span class="font-mono">${fmtDelta(wi["15m"] && wi["15m"].cvd_window_delta)}</span></div>
            <div style="margin-top:6px;">CVD component: <span class="font-mono">${c7 != null ? c7 : "—"}pt</span></div>`;
        container.appendChild(div);

        if (absorption) {
            const warn = document.createElement("div");
            warn.className = "bg-bearish";
            warn.style.cssText = "padding:8px;border-radius:6px;font-size:11px;margin-top:8px;";
            warn.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> <strong>CVD ABSORPTION:</strong> 15M opposes 4H — potential reversal / reduce size.`;
            container.appendChild(warn);
        }
    };

    const renderDepth = (snapshot, container) => {
        container.innerHTML = "";
        const ob = snapshot.orderbook || {};
        const bins = ob.depth_bins || [];
        if (!bins.length) { container.innerHTML = `<div class="box-message">No depth data.</div>`; return; }
        const ul = document.createElement("ul");
        ul.className = "levels-ul";
        bins.forEach(b => {
            const cls = b.imbalance > 0.05 ? "text-green" : b.imbalance < -0.05 ? "text-red" : "";
            ul.innerHTML += `<li class="level-li-item"><span>±${b.band_pct}%</span><span class="font-mono ${cls}">${fmtSigned(b.imbalance * 100, 1)}%</span></li>`;
        });
        if (ob.skew != null) {
            const li = document.createElement("li");
            li.className = "level-li-item";
            const cls = ob.skew > 0.05 ? "text-green" : ob.skew < -0.05 ? "text-red" : "";
            li.innerHTML = `<span>FULL BOOK</span><span class="font-mono ${cls}">${fmtSigned(ob.skew * 100, 1)}%</span>`;
            ul.appendChild(li);
        }
        container.appendChild(ul);
    };

    const renderDerivatives = (snapshot, container) => {
        container.innerHTML = "";
        const oi = snapshot.open_interest || {};
        const funding = snapshot.funding || {};
        const btcd = snapshot.btc_dominance_proxy;
        const ul = document.createElement("ul");
        ul.className = "levels-ul";
        ul.innerHTML += `
            <li class="level-li-item"><span>OI now</span><span class="font-mono">${oi.value != null ? Number(oi.value).toLocaleString() : "—"}</span></li>
            <li class="level-li-item"><span>OI Δ 1h / 4h</span><span class="font-mono">${fmtSigned(oi.change_1h_pct, 2)}% / ${fmtSigned(oi.change_4h_pct, 2)}%</span></li>
            <li class="level-li-item"><span>OI 30d pctile</span><span class="font-mono">${oi.percentile_30d != null ? oi.percentile_30d + "%" : "—"}</span></li>
            <li class="level-li-item"><span>Funding now / 24h</span><span class="font-mono">${funding.current != null ? (funding.current * 100).toFixed(4) + "%" : "—"} / ${funding.avg_24h != null ? (funding.avg_24h * 100).toFixed(4) + "%" : "—"}</span></li>
            <li class="level-li-item"><span>Funding pctile</span><span class="font-mono">${funding.percentile_window != null ? funding.percentile_window + "%" : "—"}</span></li>`;
        if (btcd) {
            ul.innerHTML += `<li class="level-li-item"><span>BTC.D 4h / 24h</span><span class="font-mono">${fmtSigned(btcd.change_4h_pct, 2)}% / ${fmtSigned(btcd.change_24h_pct, 2)}%</span></li>`;
        }
        container.appendChild(ul);
    };

    // ---- NEW: market structure (BOS / CHoCH / OTE) ----
    const renderStructure = (snapshot, container) => {
        container.innerHTML = "";
        const ul = document.createElement("ul");
        ul.className = "levels-ul";
        let any = false;
        ["4h", "1h", "15m"].forEach(tf => {
            const ctx = (snapshot.smc_context || {})[tf];
            if (!ctx) return;
            any = true;
            const bos = ctx.bos || {};
            const choch = ctx.choch || {};
            const pd = ctx.premium_discount || {};
            const dirClass = (d) => d === "BULLISH" ? "text-green" : d === "BEARISH" ? "text-red" : "";

            let structHtml = `<span class="font-mono">${ctx.structure || "—"}</span>`;
            ul.innerHTML += `<li class="level-li-item"><span class="text-muted-lbl">${tf.toUpperCase()} structure</span>${structHtml}</li>`;

            if (bos.level != null) {
                const fresh = bos.fresh ? '<span class="fresh-tag">FRESH</span>' : "";
                const disp = bos.displacement ? '<span class="disp-tag">DISP</span>' : "";
                ul.innerHTML += `<li class="level-li-item"><span class="${dirClass(bos.direction)}">${tf.toUpperCase()} BOS ${bos.direction || ""}${fresh}${disp}</span><span class="font-mono">${fmtNum(bos.level)}</span></li>`;
            }
            if (choch.detected) {
                const fresh = choch.fresh ? '<span class="fresh-tag">FRESH</span>' : "";
                ul.innerHTML += `<li class="level-li-item"><span class="${dirClass(choch.direction)}">${tf.toUpperCase()} CHoCH ${choch.direction || ""}${fresh}</span><span class="font-mono">${fmtNum(choch.level)}</span></li>`;
            }
            if (pd.in_ote) {
                const ote = pd.ote_zone || {};
                ul.innerHTML += `<li class="level-li-item"><span class="text-amber">${tf.toUpperCase()} IN OTE (${pd.zone})</span><span class="level-lbl-sub">[${fmtNum(ote.low)} - ${fmtNum(ote.high)}]</span></li>`;
            }
        });
        if (!any) container.innerHTML = `<div class="box-message">No structure data.</div>`;
        else container.appendChild(ul);
    };

    // ---- NEW: untested POC magnet targets ----
    const renderTargets = (snapshot, container) => {
        container.innerHTML = "";
        const pocs = (snapshot.price_action || {}).untested_pocs_4h || [];
        if (!pocs.length) { container.innerHTML = `<div class="box-message">No untested magnet levels.</div>`; return; }
        const ul = document.createElement("ul");
        ul.className = "levels-ul";
        pocs.forEach(p => {
            ul.innerHTML += `<li class="level-li-item"><span class="text-amber">4H untested POC</span><span class="font-mono">${fmtNum(p.poc)} <span class="src-note">(${p.distance_pct != null ? p.distance_pct + "%" : "—"})</span></span></li>`;
        });
        container.appendChild(ul);
    };

    // ---- price chart with SMC overlays (lightweight-charts) ----
    const renderChart = (snapshot) => {
        const el = document.getElementById("price-chart");
        const legendEl = document.getElementById("chart-legend");
        if (!el) return;

        if (!window.LightweightCharts) {
            el.innerHTML = `<div class="box-message">Charting library unavailable (offline / blocked).</div>`;
            if (legendEl) legendEl.innerHTML = "";
            return;
        }

        if (priceChart) { try { priceChart.remove(); } catch (e) { /* noop */ } priceChart = null; }
        el.innerHTML = "";

        const series = (snapshot.chart_series || {})[chartTf] || [];
        if (!series.length) {
            el.innerHTML = `<div class="box-message">No chart data in this snapshot (regenerate to populate).</div>`;
            if (legendEl) legendEl.innerHTML = "";
            return;
        }

        const LS = LightweightCharts.LineStyle;
        const chart = LightweightCharts.createChart(el, {
            width: el.clientWidth || 600,
            height: 360,
            layout: { background: { color: "transparent" }, textColor: "#cbd5e1", fontSize: 11,
                      fontFamily: "JetBrains Mono, monospace" },
            grid: { vertLines: { color: "rgba(255,255,255,0.04)" }, horzLines: { color: "rgba(255,255,255,0.04)" } },
            rightPriceScale: { borderColor: "rgba(255,255,255,0.10)" },
            timeScale: { borderColor: "rgba(255,255,255,0.10)", timeVisible: true, secondsVisible: false },
            crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
        });
        const candle = chart.addCandlestickSeries({
            upColor: "#10b981", downColor: "#ef4444",
            borderUpColor: "#10b981", borderDownColor: "#ef4444",
            wickUpColor: "#10b981", wickDownColor: "#ef4444",
        });
        candle.setData(series);

        const legend = [];
        const addLine = (price, color, title, style, axisLabel) => {
            if (price == null || Number.isNaN(Number(price)) || Number(price) <= 0) return;
            candle.createPriceLine({
                price: Number(price), color, lineWidth: 1,
                lineStyle: style != null ? style : LS.Dashed,
                axisLabelVisible: !!axisLabel, title,
            });
            legend.push({ title, color });
        };

        const strategies = snapshot.strategies || {};
        const dir = (strategies.setup_direction || "neutral").toLowerCase();

        // 1) Engine trade geometry (most important — solid, axis-labelled)
        const eng = snapshot.engine_trade || strategies.engine_trade || {};
        if (eng.valid) {
            addLine(eng.entry, "#3b82f6", "ENTRY", LS.Solid, true);
            addLine(eng.stop_loss, "#ef4444", "SL", LS.Solid, true);
            addLine(eng.take_profit, "#10b981", "TP", LS.Solid, true);
        }

        // 2) Value area for the selected timeframe
        const pa = snapshot.price_action || {};
        const va = pa["value_area_" + chartTf] || pa.value_area_1h || {};
        addLine(va.poc, "#a855f7", "POC", LS.Dashed, true);
        addLine(va.vah, "rgba(168,85,247,0.5)", "VAH", LS.Dotted, false);
        addLine(va.val, "rgba(168,85,247,0.5)", "VAL", LS.Dotted, false);

        // 3) Previous day
        const pday = snapshot.previous_day || {};
        addLine(pday.pdh, "rgba(148,163,184,0.55)", "PDH", LS.Dotted, false);
        addLine(pday.pdl, "rgba(148,163,184,0.55)", "PDL", LS.Dotted, false);

        // 4) Liquidity pools (nearest BSL/SSL on this TF)
        const ctx = (snapshot.smc_context || {})[chartTf] || {};
        const liq = ctx.liquidity_levels || {};
        if (liq.buy_side && liq.buy_side[0] != null) addLine(liq.buy_side[0], "#f59e0b", "BSL", LS.Dotted, false);
        if (liq.sell_side && liq.sell_side[0] != null) addLine(liq.sell_side[0], "#f59e0b", "SSL", LS.Dotted, false);

        // 5) Nearest OB in the setup direction (top + bottom edges)
        const obs = ctx.order_blocks || {};
        const obSet = dir === "bullish" ? obs.bullish : dir === "bearish" ? obs.bearish : ((obs.bullish || []).concat(obs.bearish || []));
        if (obSet && obSet.length) {
            const ob = obSet[obSet.length - 1];
            const c = dir === "bearish" ? "rgba(239,68,68,0.65)" : "rgba(16,185,129,0.65)";
            addLine(ob.top, c, "OB", LS.Dashed, false);
            addLine(ob.bottom, c, "OB", LS.Dashed, false);
        }

        // 6) Nearest FVG in the setup direction
        const fvg = ctx.fvg || {};
        const f = dir === "bearish" ? fvg.nearest_bearish_fvg : fvg.nearest_bullish_fvg;
        if (f) {
            addLine(f.top, "rgba(59,130,246,0.45)", "FVG", LS.Dotted, false);
            addLine(f.bottom, "rgba(59,130,246,0.45)", "FVG", LS.Dotted, false);
        }

        // 7) Untested 4H POC magnets
        (pa.untested_pocs_4h || []).slice(0, 2).forEach(p => addLine(p.poc, "rgba(245,158,11,0.85)", "naked POC", LS.LargeDashed, false));

        chart.timeScale().fitContent();
        priceChart = chart;

        // legend (dedup by title, preserve first colour)
        if (legendEl) {
            const seen = new Set();
            legendEl.innerHTML = legend
                .filter(l => !seen.has(l.title) && seen.add(l.title))
                .map(l => `<span class="legend-chip"><span class="legend-dot" style="background:${l.color}"></span>${l.title}</span>`)
                .join("");
        }
    };

    // ---- controls ----
    symbolSelector.addEventListener("change", () => {
        if (symbolSelector.value === "CUSTOM") {
            customSymbolGroup.classList.remove("hidden-animation");
            customSymbolGroup.classList.add("show-animation");
            customSymbolInput.focus();
        } else {
            customSymbolGroup.classList.remove("show-animation");
            customSymbolGroup.classList.add("hidden-animation");
        }
    });

    btnRunAnalysis.addEventListener("click", async () => {
        let symbol = symbolSelector.value;
        if (symbol === "CUSTOM") {
            symbol = customSymbolInput.value.trim().toUpperCase();
            if (!symbol || !symbol.includes("/")) { alert("Please enter a valid symbol format (e.g. LINK/USDT)."); return; }
        }
        btnRunAnalysis.disabled = true;
        const btnText = btnRunAnalysis.querySelector(".btn-text");
        const loader = btnRunAnalysis.querySelector(".loader");
        btnText.classList.add("hidden");
        loader.classList.remove("hidden");
        try {
            const response = await fetch("/api/run", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ symbol }),
            });
            const data = await response.json();
            btnRunAnalysis.disabled = false;
            btnText.classList.remove("hidden");
            loader.classList.add("hidden");
            if (data.success) {
                if (data.analysis && data.analysis.error) {
                    console.warn("Gemini failed; rendering deterministic snapshot only:", data.analysis.error);
                }
                renderWorkspace(data.snapshot, data.analysis || {});
                refreshHistory();
            } else {
                alert(`Error executing Cryptera: ${data.error || "Server error"}`);
            }
        } catch (err) {
            console.error("API Call error:", err);
            btnRunAnalysis.disabled = false;
            btnText.classList.remove("hidden");
            loader.classList.add("hidden");
            alert("Failed to connect to the Cryptera Flask backend API.");
        }
    });

    btnRefreshHistory.addEventListener("click", refreshHistory);

    // chart timeframe switcher
    document.querySelectorAll(".chart-tf-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".chart-tf-btn").forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            chartTf = btn.getAttribute("data-tf") || "15m";
            if (lastSnapshot) renderChart(lastSnapshot);
        });
    });

    // keep the chart sized to its container
    window.addEventListener("resize", () => {
        if (!priceChart) return;
        const el = document.getElementById("price-chart");
        if (el && el.clientWidth) priceChart.applyOptions({ width: el.clientWidth });
    });

    refreshHistory();
});

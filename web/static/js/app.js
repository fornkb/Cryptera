/*
 * Cryptera v3.1 dashboard controller.
 *
 * Consumes the structured Gemini JSON analysis directly — no regex parsing.
 * The backend always returns `{ snapshot, analysis }` from `/api/run` and
 * `/api/history/<filename>`; the snapshot also carries an `analysis` key for
 * older flows.
 */

document.addEventListener("DOMContentLoaded", () => {
    // ------- DOM ------- //
    const $ = (id) => document.getElementById(id);

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
    const badgeBias = $("badge-bias");
    const badgeAction = $("badge-action");
    const badgeVolumeGate = $("badge-volume-gate");
    const valEntry = $("val-entry");
    const valStopLoss = $("val-stop-loss");
    const valTakeProfit = $("val-take-profit");
    const valRR = $("val-rr");

    const scoreEls = {
        c1: $("score-c1"),
        c2: $("score-c2"),
        c3: $("score-c3"),
        c4: $("score-c4"),
        c5: $("score-c5"),
        c6: $("score-c6"),
        c7: $("score-c7"),
        c8: $("score-c8"),
    };

    const vaValLabel = $("va-val-label");
    const vaVahLabel = $("va-vah-label");
    const vaPocVal = $("va-poc-val");
    const valObContainer = $("val-ob-container");
    const valFvgContainer = $("val-fvg-container");
    const valSrContainer = $("val-sr-container");
    const valCvdContainer = $("val-cvd-container");
    const valDepthContainer = $("val-depth-container");
    const valDerivsContainer = $("val-derivs-container");

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

    // ------- helpers ------- //

    const fmtNum = (v, digits = 2) => {
        if (v === null || v === undefined || Number.isNaN(Number(v))) return "—";
        return Number(v).toFixed(digits);
    };
    const fmtPct = (v, digits = 2) => {
        if (v === null || v === undefined || Number.isNaN(Number(v))) return "—";
        return `${Number(v).toFixed(digits)}%`;
    };
    const fmtSigned = (v, digits = 2) => {
        if (v === null || v === undefined || Number.isNaN(Number(v))) return "—";
        const n = Number(v);
        const sign = n > 0 ? "+" : "";
        return `${sign}${n.toFixed(digits)}`;
    };
    const safeText = (el, value, fallback = "—") => {
        if (!el) return;
        el.textContent = (value === undefined || value === null || value === "") ? fallback : String(value);
    };
    const setBadge = (el, label, bg) => {
        if (!el) return;
        el.textContent = label;
        el.className = `badge ${bg || "bg-neutral"}`;
    };

    // ------- history list ------- //

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

                const biasClass = item.bias === "BULLISH" ? "text-green"
                    : item.bias === "BEARISH" ? "text-red" : "";
                const actionLabel = item.action || "HOLD";
                const actionBadgeClass =
                    actionLabel === "ACTIVE_TRADE" ? "bg-bullish"
                    : actionLabel === "CONDITIONAL_ENTRY" ? "bg-warning"
                    : "bg-hold";

                li.innerHTML = `
                    <div class="history-item-top">
                        <span class="history-item-symbol">${item.symbol}</span>
                        <span class="history-item-time">${(item.timestamp || "").substring(5, 16)}</span>
                    </div>
                    <div class="history-item-bottom">
                        <span class="history-item-meta">
                            Bias: <span class="${biasClass}">${item.bias}</span> | Score: <span>${item.score}</span>
                        </span>
                        <span class="badge ${actionBadgeClass}" style="padding: 1px 6px; font-size: 8px;">${actionLabel}</span>
                    </div>
                `;

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

    // ------- render workspace ------- //

    const renderWorkspace = (snapshot, analysis) => {
        workspaceWelcome.classList.add("hidden");
        workspaceAnalysis.classList.remove("hidden");

        analysis = analysis || {};
        const header = analysis.header || {};
        const mtf = analysis.mtf_context || {};
        const narrative = analysis.narrative || {};
        const tradeDecision = analysis.trade_decision || {};
        const reasoning = tradeDecision.reasoning || {};
        const forwardScenario = analysis.forward_scenario || {};

        // -- top state dials -- //
        safeText(valMarketRegime, snapshot.market_regime, "RANGING / SIDEWAYS");
        valMarketRegime.className = "";
        if (snapshot.market_regime === "Trending Bullish") valMarketRegime.classList.add("text-green");
        else if (snapshot.market_regime === "Trending Bearish") valMarketRegime.classList.add("text-red");

        safeText(valFearGreed, snapshot.fear_greed_index != null ? `${snapshot.fear_greed_index} / 100` : "—", "—");

        const funding = snapshot.funding || {};
        if (funding && funding.current != null) {
            const trend = funding.trend ? funding.trend.toUpperCase() : "—";
            valFundingRate.textContent = `${(funding.current * 100).toFixed(4)}% / ${trend}`;
        } else {
            valFundingRate.textContent = "—";
        }

        const st = snapshot.supertrend || {};
        const stDir = (st.direction || "—").toUpperCase();
        const stLvl = st.level != null ? st.level.toFixed(2) : "0.00";
        valSupertrend.textContent = `${stDir} // ${stLvl}`;
        valSupertrend.className = stDir === "BULLISH" ? "text-green" : stDir === "BEARISH" ? "text-red" : "";

        const vr = snapshot.volatility_regime || {};
        const tag = (k) => (vr[k] && vr[k].regime) ? vr[k].regime.toUpperCase() : "—";
        safeText(valVolRegime, `${tag("4h")} / ${tag("1h")} / ${tag("15m")}`);

        const event = snapshot.event_guard || {};
        if (event.active && Array.isArray(event.events) && event.events.length) {
            const e = event.events[0];
            valEventGuard.textContent = `${e.name} (${e.minutes_until.toFixed(0)} min)`;
            valEventGuard.className = "text-red";
        } else {
            valEventGuard.textContent = "NO ACTIVE WINDOW";
            valEventGuard.className = "";
        }

        // -- confluence ring + bias + action -- //
        const strategies = snapshot.strategies || {};
        const score = header.score != null ? header.score : (strategies.confluence_score || 0);
        valConfluenceScore.textContent = score;
        const offset = 439.8 - (439.8 * score / 100);
        confluenceScoreRing.style.strokeDashoffset = offset;
        if (score >= 60) confluenceScoreRing.style.stroke = "#10b981";
        else if (score >= 45) confluenceScoreRing.style.stroke = "#f59e0b";
        else confluenceScoreRing.style.stroke = "#ef4444";

        const bias = (header.bias || (strategies.trend_bias || "neutral").toUpperCase());
        const biasClass = bias === "BULLISH" ? "bg-bullish" : bias === "BEARISH" ? "bg-bearish" : "bg-neutral";
        setBadge(badgeBias, bias, biasClass);

        const action = header.action || (score >= 60 ? "ACTIVE_TRADE" : score >= 45 ? "CONDITIONAL_ENTRY" : "HOLD");
        const actionLabel = action.replace(/_/g, " ");
        const actionClass =
            action === "ACTIVE_TRADE" ? "bg-bullish"
            : action === "CONDITIONAL_ENTRY" ? "bg-warning"
            : "bg-hold";
        setBadge(badgeAction, actionLabel, actionClass);

        const gateState = header.volume_gate || (strategies.volume_gate && strategies.volume_gate.state) || "CLEAR";
        const gateClass =
            gateState === "HARD_GATE" ? "bg-bearish"
            : gateState === "LOW_VOL_WARNING" ? "bg-warning"
            : "bg-bullish";
        setBadge(badgeVolumeGate, gateState.replace(/_/g, " "), gateClass);

        // -- entry / SL / TP -- //
        const td = tradeDecision || {};
        safeText(valEntry, td.entry != null ? fmtNum(td.entry, 2) : "N/A", "N/A");
        safeText(valStopLoss, td.stop_loss != null ? fmtNum(td.stop_loss, 2) : "N/A", "N/A");
        safeText(valTakeProfit, td.take_profit != null ? fmtNum(td.take_profit, 2) : "N/A", "N/A");
        const rr = td.rr || {};
        if (rr.ratio != null) {
            const passed = rr.passed ? "✓" : "✗";
            valRR.textContent = `${Number(rr.ratio).toFixed(2)}:1 ${passed}`;
            valRR.className = `metric-value font-mono ${rr.passed ? "text-green" : "text-red"}`;
        } else {
            valRR.textContent = "—";
            valRR.className = "metric-value font-mono";
        }

        // -- score breakdown -- //
        const breakdown = (header.score_breakdown && Object.keys(header.score_breakdown).length)
            ? header.score_breakdown
            : remapBreakdown(strategies.confluence_breakdown || {});
        scoreEls.c1.textContent = breakdown.c1_trend ?? 0;
        scoreEls.c2.textContent = breakdown.c2_ob_prox ?? 0;
        scoreEls.c3.textContent = breakdown.c3_sweep ?? 0;
        scoreEls.c4.textContent = breakdown.c4_momentum ?? 0;
        scoreEls.c5.textContent = breakdown.c5_fvg_magnet ?? 0;
        scoreEls.c6.textContent = breakdown.c6_ote ?? 0;
        scoreEls.c7.textContent = breakdown.c7_cvd ?? 0;
        scoreEls.c8.textContent = breakdown.c8_stoch ?? 0;

        // -- value area -- //
        const pa = snapshot.price_action || {};
        const va = pa.value_area_1h || {};
        const vah = va.vah || 0;
        const val = va.val || 0;
        const poc = va.poc || 0;
        vaValLabel.textContent = fmtNum(val);
        vaVahLabel.textContent = fmtNum(vah);
        vaPocVal.textContent = fmtNum(poc);

        const price =
            (snapshot.smc_context && snapshot.smc_context["15m"] && snapshot.smc_context["15m"].current_price)
            || header.price
            || snapshot.strategies && snapshot.strategies.current_price
            || poc
            || 0;
        const span = vah - val;
        if (span > 0) {
            const rel = ((price - val) / span) * 100;
            const clamped = Math.min(Math.max(rel, 0), 100);
            const vaAreaEl = document.querySelector(".va-area");
            if (vaAreaEl) {
                vaAreaEl.style.left = `${Math.max(clamped - 15, 0)}%`;
                vaAreaEl.style.right = `${Math.max(100 - clamped - 15, 0)}%`;
            }
        }

        // -- OB / FVG / SR lists -- //
        renderOBs(snapshot, valObContainer);
        renderFVGs(snapshot, valFvgContainer);
        renderSR(snapshot, valSrContainer);
        renderCVD(snapshot, valCvdContainer, strategies);
        renderDepth(snapshot, valDepthContainer);
        renderDerivatives(snapshot, valDerivsContainer);

        // -- narrative + forward scenario -- //
        safeText(valMarketNarrative, narrative.summary, "No narrative generated yet.");
        safeText(valPrimaryDraw, narrative.primary_draw, "—");

        // Forward scenario card
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

        // Reasoning bullets
        safeText(reasoningStruct, reasoning.structure, "—");
        safeText(reasoningLiq, reasoning.liquidity, "—");
        safeText(reasoningMom, reasoning.momentum, "—");
        safeText(reasoningBook, reasoning.sentiment, "—");
    };

    // Back-compat shim if a snapshot was generated before the C1-C8 naming
    const remapBreakdown = (b) => ({
        c1_trend: b.c1_trend_alignment ?? b.trend_alignment ?? 0,
        c2_ob_prox: b.c2_ob_proximity ?? b.ob_proximity ?? 0,
        c3_sweep: b.c3_liquidity_sweep ?? b.liquidity_sweep ?? 0,
        c4_momentum: b.c4_momentum ?? b.momentum ?? 0,
        c5_fvg_magnet: b.c5_fvg_magnet ?? b.fvg_magnet ?? 0,
        c6_ote: b.c6_ote_bonus ?? b.ote_bonus ?? 0,
        c7_cvd: b.c7_cvd_alignment ?? b.cvd_divergence ?? 0,
        c8_stoch: b.c8_stochrsi ?? 0,
    });

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
                ul.innerHTML += `
                    <li class="level-li-item">
                        <span class="text-green">${tf.toUpperCase()} BULL OB${hvn}</span>
                        <span class="level-lbl-sub">[${fmtNum(ob.bottom)} - ${fmtNum(ob.top)}]</span>
                    </li>`;
            });
            (obs.bearish || []).forEach(ob => {
                any = true;
                const hvn = ob.high_volume_node ? ' <span class="hvn-tag">HVN</span>' : "";
                ul.innerHTML += `
                    <li class="level-li-item">
                        <span class="text-red">${tf.toUpperCase()} BEAR OB${hvn}</span>
                        <span class="level-lbl-sub">[${fmtNum(ob.bottom)} - ${fmtNum(ob.top)}]</span>
                    </li>`;
            });
        });
        if (!any) {
            container.innerHTML = `<div class="box-message">No active unmitigated OBs in lookback.</div>`;
        } else {
            container.appendChild(ul);
        }
    };

    const renderFVGs = (snapshot, container) => {
        container.innerHTML = "";
        const ul = document.createElement("ul");
        ul.className = "levels-ul";
        let any = false;
        ["4h", "1h", "15m"].forEach(tf => {
            const fvgs = (snapshot.smc_context || {})[tf] && (snapshot.smc_context[tf].fvg || {});
            if (!fvgs) return;
            const b = fvgs.nearest_bullish_fvg;
            const r = fvgs.nearest_bearish_fvg;
            if (b) {
                any = true;
                ul.innerHTML += `
                    <li class="level-li-item">
                        <span class="text-green">${tf.toUpperCase()} BULL FVG</span>
                        <span class="level-lbl-sub">[${fmtNum(b.bottom)} - ${fmtNum(b.top)}]</span>
                    </li>`;
            }
            if (r) {
                any = true;
                ul.innerHTML += `
                    <li class="level-li-item">
                        <span class="text-red">${tf.toUpperCase()} BEAR FVG</span>
                        <span class="level-lbl-sub">[${fmtNum(r.bottom)} - ${fmtNum(r.top)}]</span>
                    </li>`;
            }
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
            (liq.sell_side || []).forEach(lvl => {
                any = true;
                ul.innerHTML += `
                    <li class="level-li-item">
                        <span class="text-green">${tf.toUpperCase()} SSL</span>
                        <span class="font-mono">${fmtNum(lvl)}</span>
                    </li>`;
            });
            (liq.buy_side || []).forEach(lvl => {
                any = true;
                ul.innerHTML += `
                    <li class="level-li-item">
                        <span class="text-red">${tf.toUpperCase()} BSL</span>
                        <span class="font-mono">${fmtNum(lvl)}</span>
                    </li>`;
            });
        });
        if (!any) container.innerHTML = `<div class="box-message">No historical S/R cluster zones.</div>`;
        else container.appendChild(ul);
    };

    const renderCVD = (snapshot, container, strategies) => {
        container.innerHTML = "";
        const wi = snapshot.windowed_indicators || {};
        const absorption = (strategies && strategies.cvd_absorption_warning) || false;
        const c7 = strategies && strategies.confluence_breakdown && strategies.confluence_breakdown.c7_cvd_alignment;

        const div = document.createElement("div");
        div.style.padding = "8px";
        div.style.fontSize = "11px";
        div.style.lineHeight = "1.45";

        const fmtDelta = (v) => v == null ? "—" : fmtSigned(v, 2);
        div.innerHTML = `
            <div>4H Δ: <span class="font-mono">${fmtDelta(wi["4h"] && wi["4h"].cvd_window_delta)}</span></div>
            <div>1H Δ: <span class="font-mono">${fmtDelta(wi["1h"] && wi["1h"].cvd_window_delta)}</span></div>
            <div>15M Δ: <span class="font-mono">${fmtDelta(wi["15m"] && wi["15m"].cvd_window_delta)}</span></div>
            <div style="margin-top:6px;">Component C7: <span class="font-mono">${c7 != null ? c7 : "—"}pt</span></div>
        `;
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
        if (!bins.length) {
            container.innerHTML = `<div class="box-message">No depth data.</div>`;
            return;
        }
        const ul = document.createElement("ul");
        ul.className = "levels-ul";
        bins.forEach(b => {
            const cls = b.imbalance > 0.05 ? "text-green" : b.imbalance < -0.05 ? "text-red" : "";
            ul.innerHTML += `
                <li class="level-li-item">
                    <span>±${b.band_pct}%</span>
                    <span class="font-mono ${cls}">${fmtSigned(b.imbalance * 100, 1)}%</span>
                </li>`;
        });
        const sk = ob.skew != null ? `Total skew: ${fmtSigned(ob.skew * 100, 1)}%` : "Total skew: —";
        const li = document.createElement("li");
        li.className = "level-li-item";
        li.innerHTML = `<span>FULL BOOK</span><span class="font-mono">${sk.replace("Total skew: ", "")}</span>`;
        ul.appendChild(li);
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
            <li class="level-li-item">
                <span>OI now</span>
                <span class="font-mono">${oi.value != null ? Number(oi.value).toLocaleString() : "—"}</span>
            </li>
            <li class="level-li-item">
                <span>OI Δ 1h / 4h</span>
                <span class="font-mono">${fmtSigned(oi.change_1h_pct, 2)}% / ${fmtSigned(oi.change_4h_pct, 2)}%</span>
            </li>
            <li class="level-li-item">
                <span>OI 30d percentile</span>
                <span class="font-mono">${oi.percentile_30d != null ? oi.percentile_30d + "%" : "—"}</span>
            </li>
            <li class="level-li-item">
                <span>Funding now / 24h avg</span>
                <span class="font-mono">${funding.current != null ? (funding.current * 100).toFixed(4) + "%" : "—"} / ${funding.avg_24h != null ? (funding.avg_24h * 100).toFixed(4) + "%" : "—"}</span>
            </li>
            <li class="level-li-item">
                <span>Funding trend</span>
                <span class="font-mono">${funding.trend ? funding.trend.toUpperCase() : "—"}</span>
            </li>
        `;
        if (btcd) {
            ul.innerHTML += `
                <li class="level-li-item">
                    <span>BTC.D (BTCDOM) 4h/24h</span>
                    <span class="font-mono">${fmtSigned(btcd.change_4h_pct, 2)}% / ${fmtSigned(btcd.change_24h_pct, 2)}%</span>
                </li>`;
        }
        container.appendChild(ul);
    };

    // ------- run analysis ------- //

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
            if (!symbol || !symbol.includes("/")) {
                alert("Please enter a valid symbol format (e.g. LINK/USDT).");
                return;
            }
        }
        btnRunAnalysis.disabled = true;
        const btnText = btnRunAnalysis.querySelector(".btn-text");
        const loader = btnRunAnalysis.querySelector(".loader");
        btnText.classList.add("hidden");
        loader.classList.remove("hidden");

        try {
            const response = await fetch("/api/run", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({ symbol }),
            });
            const data = await response.json();
            btnRunAnalysis.disabled = false;
            btnText.classList.remove("hidden");
            loader.classList.add("hidden");

            if (data.success) {
                if (data.analysis && data.analysis.error) {
                    alert(`Engine ran but Gemini failed: ${data.analysis.error}`);
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
    refreshHistory();
});

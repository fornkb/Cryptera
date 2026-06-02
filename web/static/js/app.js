document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const symbolSelector = document.getElementById("symbol-selector");
    const customSymbolGroup = document.getElementById("custom-symbol-group");
    const customSymbolInput = document.getElementById("custom-symbol-input");
    const btnRunAnalysis = document.getElementById("btn-run-analysis");
    const btnRefreshHistory = document.getElementById("btn-refresh-history");
    const historyList = document.getElementById("history-list");
    const historyListLoading = document.getElementById("history-list-loading");
    const historyListEmpty = document.getElementById("history-list-empty");
    
    const workspaceWelcome = document.getElementById("workspace-welcome");
    const workspaceAnalysis = document.getElementById("workspace-analysis");
    
    // State indicators
    const valMarketRegime = document.getElementById("val-market-regime");
    const valFearGreed = document.getElementById("val-fear-greed");
    const valFundingRate = document.getElementById("val-funding-rate");
    const valSupertrend = document.getElementById("val-supertrend");
    
    // Confluence cards
    const valConfluenceScore = document.getElementById("val-confluence-score");
    const confluenceScoreRing = document.getElementById("confluence-score-ring");
    const badgeBias = document.getElementById("badge-bias");
    const badgeAction = document.getElementById("badge-action");
    const valEntry = document.getElementById("val-entry");
    const valStopLoss = document.getElementById("val-stop-loss");
    const valTakeProfit = document.getElementById("val-take-profit");
    
    // Score matrix
    const scoreTrend = document.getElementById("score-trend");
    const scoreOb = document.getElementById("score-ob");
    const scoreCvd = document.getElementById("score-cvd");
    const scoreMom = document.getElementById("score-mom");
    const scoreFvg = document.getElementById("score-fvg");
    const scoreBook = document.getElementById("score-book");
    
    // SMC Grids
    const vaValLabel = document.getElementById("va-val-label");
    const vaVahLabel = document.getElementById("va-vah-label");
    const vaPocVal = document.getElementById("va-poc-val");
    const valObContainer = document.getElementById("val-ob-container");
    const valFvgContainer = document.getElementById("val-fvg-container");
    const valSrContainer = document.getElementById("val-sr-container");
    const valCvdContainer = document.getElementById("val-cvd-container");
    
    // Narrative & Hypothetical Setup
    const valMarketNarrative = document.getElementById("val-market-narrative");
    const hypotheticalSetupCard = document.getElementById("hypothetical-setup-card");
    const hypoDirection = document.getElementById("hypo-direction");
    const hypoTrigger = document.getElementById("hypo-trigger");
    const hypoEntry = document.getElementById("hypo-entry");
    const hypoSl = document.getElementById("hypo-sl");
    const hypoTp = document.getElementById("hypo-tp");
    const hypoEvidence = document.getElementById("hypo-evidence");
    
    // Reasoning bullets
    const reasoningStruct = document.getElementById("reasoning-struct");
    const reasoningLiq = document.getElementById("reasoning-liq");
    const reasoningMom = document.getElementById("reasoning-mom");
    const reasoningBook = document.getElementById("reasoning-book");

    // Dynamic state trackers
    let activeFilename = null;

    // Toggle Custom Symbol field
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

    // Refresh History list
    const refreshHistory = async () => {
        historyListLoading.classList.remove("hidden");
        historyListEmpty.classList.add("hidden");
        historyList.innerHTML = "";
        
        try {
            const res = await fetch("/api/history");
            const data = await res.json();
            
            historyListLoading.classList.add("hidden");
            
            if (!data || data.length === 0) {
                historyListEmpty.classList.remove("hidden");
                return;
            }
            
            data.forEach(item => {
                const li = document.createElement("li");
                li.className = `history-item ${activeFilename === item.filename ? 'active' : ''}`;
                li.setAttribute("data-filename", item.filename);
                
                // Color bias and action
                const biasClass = item.bias === "BULLISH" ? "text-green" : (item.bias === "BEARISH" ? "text-red" : "");
                const actionBadgeClass = item.action === "TRADE READY" ? "bg-bullish" : "bg-hold";
                
                li.innerHTML = `
                    <div class="history-item-top">
                        <span class="history-item-symbol">${item.symbol}</span>
                        <span class="history-item-time">${item.timestamp.substring(5, 16)}</span>
                    </div>
                    <div class="history-item-bottom">
                        <span class="history-item-meta">
                            Bias: <span class="${biasClass}">${item.bias}</span> | Score: <span>${item.score}</span>
                        </span>
                        <span class="badge ${actionBadgeClass}" style="padding: 1px 6px; font-size: 8px;">${item.action}</span>
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

    // Load detailed snapshot from file
    const loadSnapshot = async (filename) => {
        activeFilename = filename;
        try {
            const res = await fetch(`/api/history/${filename}`);
            const snapshot = await res.json();
            
            // Mock a narration plan based on loaded history details
            // Since older snapshots might not have the raw narration text cached directly,
            // we look for a cached summary, or compile it out of strategies
            const narration = snapshot.narration || compileFallbackNarration(snapshot);
            
            renderWorkspace(snapshot, narration);
        } catch (err) {
            alert("Failed to load snapshot details.");
            console.error(err);
        }
    };

    // Render snapshot and narration in main workspace
    const renderWorkspace = (snapshot, narration) => {
        // Toggle view
        workspaceWelcome.classList.add("hidden");
        workspaceAnalysis.classList.remove("hidden");
        
        // 1. Dials & Dials Panel
        valMarketRegime.textContent = snapshot.market_regime || "RANGING / SIDEWAYS";
        valFearGreed.textContent = `${snapshot.fear_greed_index} / 100`;
        valFundingRate.textContent = `${(snapshot.funding_rate * 100).toFixed(4)}%`;
        
        // Color-code Regime
        valMarketRegime.className = "";
        if (snapshot.market_regime === "Trending Bullish") {
            valMarketRegime.classList.add("text-green");
        } else if (snapshot.market_regime === "Trending Bearish") {
            valMarketRegime.classList.add("text-red");
        }
        
        const trendDir = snapshot.supertrend?.direction?.toUpperCase() || "NEUTRAL";
        const trendLvl = snapshot.supertrend?.level || 0.00;
        valSupertrend.textContent = `${trendDir} // ${trendLvl.toFixed(2)}`;
        valSupertrend.className = trendDir === "BULLISH" ? "text-green" : "text-red";
        
        // 2. Confluence Analysis
        const strategies = snapshot.strategies || {};
        const score = strategies.confluence_score || 0;
        valConfluenceScore.textContent = score;
        
        // Animate circular Ring (total length = 439.8)
        const offset = 439.8 - (439.8 * score / 100);
        confluenceScoreRing.style.strokeDashoffset = offset;
        
        // Set Ring color based on score
        if (score >= 60) {
            confluenceScoreRing.style.stroke = "#10b981"; // Emerald (Active)
        } else if (score >= 45) {
            confluenceScoreRing.style.stroke = "#f59e0b"; // Amber (Conditional)
        } else {
            confluenceScoreRing.style.stroke = "#ef4444"; // Crimson (HOLD)
        }
        
        // Score breakdown matrix
        const breakdown = strategies.confluence_breakdown || {};
        scoreTrend.textContent = `${breakdown.trend_alignment || 0}pt`;
        scoreOb.textContent = `${breakdown.ob_proximity || 0}pt`;
        scoreCvd.textContent = `${breakdown.cvd_divergence || 0}pt`;
        scoreMom.textContent = `${breakdown.momentum || 0}pt`;
        scoreFvg.textContent = `${breakdown.fvg_magnet || 0}pt`;
        scoreBook.textContent = `${breakdown.orderbook_funding || 0}pt`;

        // 3. SMC Price Context Table
        const smc_15m = snapshot.smc_context?.["15m"] || {};
        const smc_1h = snapshot.smc_context?.["1h"] || {};
        const smc_4h = snapshot.smc_context?.["4h"] || {};
        
        // Trades are on 1H timeframe primarily
        const price = smc_1h.current_price || smc_15m.current_price || snapshot.previous_day?.pdc || 0.00;
        
        const isReady = strategies.final_setup_ready || false;
        
        // Setup Bias
        const bias = strategies.trend_bias?.toUpperCase() || "NEUTRAL";
        badgeBias.textContent = bias;
        badgeBias.className = `badge ${bias === "BULLISH" ? 'bg-bullish' : (bias === "BEARISH" ? 'bg-bearish' : 'bg-neutral')}`;
        
        // Recommended Action based on new thresholds
        let action = "HOLD";
        if (score >= 60) {
            action = bias === "BULLISH" ? "BUY" : (bias === "BEARISH" ? "SELL" : "HOLD");
        } else if (score >= 45) {
            action = bias === "BULLISH" ? "CONDITIONAL BUY" : (bias === "BEARISH" ? "CONDITIONAL SELL" : "HOLD");
        }
        badgeAction.textContent = action;
        badgeAction.className = `badge ${action.includes("BUY") ? 'bg-bullish' : (action.includes("SELL") ? 'bg-bearish' : 'bg-neutral')}`;
        
        // Extract Entry, SL, TP from snapshot or set N/A
        if (action === "HOLD") {
            valEntry.textContent = "N/A";
            valStopLoss.textContent = "N/A";
            valTakeProfit.textContent = "N/A";
        } else {
            // Entry level calculated from 1H and 15m data
            const bull_ob_1h = smc_1h.order_blocks?.bullish?.[0];
            const bear_ob_1h = smc_1h.order_blocks?.bearish?.[0];
            const bull_ob_15m = smc_15m.order_blocks?.bullish?.[0];
            const bear_ob_15m = smc_15m.order_blocks?.bearish?.[0];
            
            const ob_range = bias === "BULLISH" 
                ? (bull_ob_1h || bull_ob_15m)
                : (bear_ob_1h || bear_ob_15m);
            valEntry.textContent = ob_range ? ob_range.top.toFixed(2) : price.toFixed(2);
            
            // Stop Loss placement calculated from 1H and 15m data (safer/more conservative level)
            const sl_1h = bias === "BULLISH"
                ? (smc_1h.liquidity_levels?.sell_side?.[0] || smc_1h.current_price * 0.99)
                : (smc_1h.liquidity_levels?.buy_side?.[0] || smc_1h.current_price * 1.01);
            const sl_15m = bias === "BULLISH"
                ? (smc_15m.liquidity_levels?.sell_side?.[0] || smc_15m.current_price * 0.99)
                : (smc_15m.liquidity_levels?.buy_side?.[0] || smc_15m.current_price * 1.01);
            
            const sl_val = bias === "BULLISH"
                ? Math.min(sl_1h, sl_15m)
                : Math.max(sl_1h, sl_15m);
            valStopLoss.textContent = sl_val.toFixed(2);
            
            // Take Profit placement targeting key levels from 4H
            const tp_4h = bias === "BULLISH"
                ? (smc_4h.liquidity_levels?.buy_side?.[0] || price * 1.02)
                : (smc_4h.liquidity_levels?.sell_side?.[0] || price * 0.98);
            valTakeProfit.textContent = tp_4h.toFixed(2);
        }
        
        // 4. Value Area bounds (1H)
        const pa = snapshot.price_action || {};
        const va_1h = pa.value_area_1h || {};
        const vah = va_1h.vah || 0;
        const val = va_1h.val || 0;
        const poc = va_1h.poc || 0;
        
        vaValLabel.textContent = val.toFixed(2);
        vaVahLabel.textContent = vah.toFixed(2);
        vaPocVal.textContent = poc.toFixed(2);
        
        // Calculate current price relative positioning in Value Area
        // e.g. where the dashed area boundary is placed
        const vaSpan = vah - val;
        if (vaSpan > 0) {
            const relative_pos = ((price - val) / vaSpan) * 100;
            // Cap visual area boundaries slightly to stay inside 100% UI bounds
            const visually_capped = Math.min(Math.max(relative_pos, 0), 100);
            const vaAreaEl = document.querySelector(".va-area");
            vaAreaEl.style.left = `${Math.max(visually_capped - 15, 0)}%`;
            vaAreaEl.style.right = `${Math.max(100 - visually_capped - 15, 0)}%`;
        }
 
        // 5. Dynamic unmitigated OB container (showing 4H, 1H, and 15m levels)
        valObContainer.innerHTML = "";
        const ulOb = document.createElement("ul");
        ulOb.className = "levels-ul";
        
        let hasOBs = false;
        
        ["4h", "1h", "15m"].forEach(tf => {
            const tf_smc = snapshot.smc_context?.[tf] || {};
            const bull = tf_smc.order_blocks?.bullish || [];
            const bear = tf_smc.order_blocks?.bearish || [];
            
            bull.forEach(ob => {
                hasOBs = true;
                ulOb.innerHTML += `
                    <li class="level-li-item">
                        <span class="text-green">${tf.toUpperCase()} BULL OB</span>
                        <span class="level-lbl-sub">[${ob.bottom.toFixed(2)} - ${ob.top.toFixed(2)}]</span>
                    </li>
                `;
            });
            
            bear.forEach(ob => {
                hasOBs = true;
                ulOb.innerHTML += `
                    <li class="level-li-item">
                        <span class="text-red">${tf.toUpperCase()} BEAR OB</span>
                        <span class="level-lbl-sub">[${ob.bottom.toFixed(2)} - ${ob.top.toFixed(2)}]</span>
                    </li>
                `;
            });
        });
        
        if (!hasOBs) {
            valObContainer.innerHTML = `<div class="box-message">No active unmitigated OBs in lookback.</div>`;
        } else {
            valObContainer.appendChild(ulOb);
        }
        
        // 6. Dynamic unfilled FVG container (showing nearest from 4H, 1H, and 15m levels)
        valFvgContainer.innerHTML = "";
        const ulFvg = document.createElement("ul");
        ulFvg.className = "levels-ul";
        
        let hasFVGs = false;
        
        ["4h", "1h", "15m"].forEach(tf => {
            const tf_smc = snapshot.smc_context?.[tf] || {};
            const fvgs = tf_smc.fvg || {};
            const nearest_bull = fvgs.nearest_bullish_fvg;
            const nearest_bear = fvgs.nearest_bearish_fvg;
            
            if (nearest_bull) {
                hasFVGs = true;
                ulFvg.innerHTML += `
                    <li class="level-li-item">
                        <span class="text-green">${tf.toUpperCase()} BULL FVG</span>
                        <span class="level-lbl-sub">[${nearest_bull.bottom.toFixed(2)} - ${nearest_bull.top.toFixed(2)}]</span>
                    </li>
                `;
            }
            if (nearest_bear) {
                hasFVGs = true;
                ulFvg.innerHTML += `
                    <li class="level-li-item">
                        <span class="text-red">${tf.toUpperCase()} BEAR FVG</span>
                        <span class="level-lbl-sub">[${nearest_bear.bottom.toFixed(2)} - ${nearest_bear.top.toFixed(2)}]</span>
                    </li>
                `;
            }
        });
        
        if (!hasFVGs) {
            valFvgContainer.innerHTML = `<div class="box-message">No active unfilled FVGs in range.</div>`;
        } else {
            valFvgContainer.appendChild(ulFvg);
        }
        
        // 7. Dynamic support/resistance with touches (Key Liquidity levels from 4H, 1H, and 15m)
        valSrContainer.innerHTML = "";
        const ulSr = document.createElement("ul");
        ulSr.className = "levels-ul";
        
        let hasSr = false;
        
        ["4h", "1h", "15m"].forEach(tf => {
            const tf_smc = snapshot.smc_context?.[tf] || {};
            const supports = tf_smc.liquidity_levels?.sell_side || [];
            const resistances = tf_smc.liquidity_levels?.buy_side || [];
            
            supports.forEach(sup => {
                hasSr = true;
                ulSr.innerHTML += `
                    <li class="level-li-item">
                        <span class="text-green">${tf.toUpperCase()} SSL</span>
                        <span>${sup.toFixed(2)}</span>
                    </li>
                `;
            });
            
            resistances.forEach(res => {
                hasSr = true;
                ulSr.innerHTML += `
                    <li class="level-li-item">
                        <span class="text-red">${tf.toUpperCase()} BSL</span>
                        <span>${res.toFixed(2)}</span>
                    </li>
                `;
            });
        });
        
        if (!hasSr) {
            valSrContainer.innerHTML = `<div class="box-message">No historical S/R cluster zones.</div>`;
        } else {
            valSrContainer.appendChild(ulSr);
        }
        
        // 8. CVD Divergence Warnings
        valCvdContainer.innerHTML = "";
        // Read active sub-strategies to detect divergences
        const isBullCvd = breakdown.cvd_divergence > 0 && bias === "BULLISH";
        const isBearCvd = breakdown.cvd_divergence > 0 && bias === "BEARISH";
        
        if (!isBullCvd && !isBearCvd) {
            valCvdContainer.innerHTML = `<div class="box-message">No active CVD absorption divergences.</div>`;
        } else {
            const div = document.createElement("div");
            div.style.padding = "10px";
            div.style.borderRadius = "6px";
            div.style.fontSize = "11px";
            div.style.textAlign = "center";
            
            if (isBullCvd) {
                div.className = "bg-bullish";
                div.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> <strong>BULLISH ABSORPTION DETECTED:</strong> Price Lower Low accompanied by CVD Higher Low. Limit buyers absorbing retail dump.`;
            } else {
                div.className = "bg-bearish";
                div.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> <strong>BEARISH ABSORPTION DETECTED:</strong> Price Higher High accompanied by CVD Lower High. Limit sellers absorbing buy pressure.`;
            }
            valCvdContainer.appendChild(div);
        }
        
        // 9. Narration text formatting
        valMarketNarrative.innerHTML = formatMarkdown(narration);
        
        // 10. Parse Hypothetical If-Then Setup from narration
        parseAndRenderHypotheticalSetup(narration);
        
        // 11. Parse bullet reasoning
        parseAndRenderReasoning(narration);
        
        // 12. Parse main trade plan from narration (Entry, SL, TP) to align the top Confluence panel with AI plan
        parseAndRenderMainTradePlan(narration);
    };

    // Helper: format raw markdown/narration into readable HTML
    const formatMarkdown = (text) => {
        if (!text) return "No transcript generated.";
        // Clean double asterisks for bolding
        let html = text.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
        // Clean bullet items
        html = html.replace(/^\s*[-*]\s*(.*?)$/gm, "<li>$1</li>");
        // Wrap blocks of lists in ul tags
        html = html.replace(/(<li>.*?<\/li>)+/g, "<ul>$&</ul>");
        // Replace newlines with breaks
        html = html.replace(/\n/g, "<br>");
        return html;
    };

    // Helper: extracts hypothetical setup fields directly from AI text
    const parseAndRenderHypotheticalSetup = (narrationText) => {
        try {
            if (!narrationText) {
                hypotheticalSetupCard.classList.add("hidden");
                return;
            }
            
            // Look for the "HYPOTHETICAL IF-THEN SETUP" section in the text
            const lowerText = narrationText.toLowerCase();
            const sectionIdx = lowerText.indexOf("hypothetical if-then setup");
            
            if (sectionIdx === -1) {
                // If not found in the text directly, show fallback values based on active strategy biases
                hypoDirection.textContent = badgeBias.textContent;
                hypoDirection.className = badgeBias.className.replace("badge", "hypo-value");
                hypoTrigger.textContent = "Price sweep of local liquidity pools followed by a low-timeframe market structure break.";
                hypoEntry.textContent = valEntry.textContent;
                hypoSl.textContent = valStopLoss.textContent;
                hypoTp.textContent = valTakeProfit.textContent;
                hypoEvidence.textContent = "Awaiting market confluence activation.";
                return;
            }
            
            // Extract the section string
            const rawSection = narrationText.substring(sectionIdx);
            
            // Regex matchers
            const dirMatch = rawSection.match(/price\s+direction:\s*([A-Za-z]+)/i);
            const triggerMatch = rawSection.match(/trigger\s+condition:\s*(.*?)(?=\n|$)/i);
            const entryMatch = rawSection.match(/entry\s+level:\s*([\d.]+)/i);
            const slMatch = rawSection.match(/invalidation\s*\(sl\):\s*([\d.]+)/i);
            const tpMatch = rawSection.match(/target\s*\(tp\):\s*([\d.]+)/i);
            const evMatch = rawSection.match(/supporting\s+data\s*&\s*confluence:\s*(.*?)(?=\n|$)/i);
            
            // Set text values
            const dir = dirMatch ? dirMatch[1].toUpperCase() : "NEUTRAL";
            hypoDirection.textContent = dir;
            hypoDirection.className = `hypo-value ${dir === 'LONG' ? 'text-green' : (dir === 'SHORT' ? 'text-red' : '')}`;
            
            hypoTrigger.textContent = triggerMatch ? triggerMatch[1].trim() : "Liquidity pool sweep followed by LTF shift.";
            hypoEntry.textContent = entryMatch ? parseFloat(entryMatch[1]).toFixed(2) : valEntry.textContent;
            hypoSl.textContent = slMatch ? parseFloat(slMatch[1]).toFixed(2) : valStopLoss.textContent;
            hypoTp.textContent = tpMatch ? parseFloat(tpMatch[1]).toFixed(2) : valTakeProfit.textContent;
            hypoEvidence.textContent = evMatch ? evMatch[1].trim() : "Absorption divergences on CVD profile.";
        } catch (err) {
            console.error("Error parsing hypothetical setup:", err);
        }
    };

    // Helper: extracts bullet reasonings from AI text
    const parseAndRenderReasoning = (narrationText) => {
        try {
            if (!narrationText) return;
            
            const lowerText = narrationText.toLowerCase();
            const idx = lowerText.indexOf("reasoning:");
            
            let lines = [];
            if (idx === -1) {
                // If reasoning block is merged, grab any bullet points or technical lines in the narration text
                lines = narrationText.split("\n").filter(line => line.trim().startsWith("-"));
            } else {
                const rawSection = narrationText.substring(idx);
                lines = rawSection.split("\n").filter(line => line.trim().startsWith("-"));
            }
            
            // Search and display specific merged reasoning categories
            let structLine = lines.find(l => l.toLowerCase().includes("structure") || l.toLowerCase().includes("struct"));
            let liqLine = lines.find(l => l.toLowerCase().includes("liquidity") || l.toLowerCase().includes("liq"));
            let momLine = lines.find(l => l.toLowerCase().includes("momentum") || l.toLowerCase().includes("mom"));
            let bookLine = lines.find(l => l.toLowerCase().includes("orderbook") || l.toLowerCase().includes("book") || l.toLowerCase().includes("sentiment"));
            
            if (structLine) {
                reasoningStruct.textContent = structLine.replace(/^[-*\s]*/, "").replace(/.*?(structure|struct):?/i, "").trim();
            } else {
                reasoningStruct.textContent = "HTF & LTF aligned market structures.";
            }
            
            if (liqLine) {
                reasoningLiq.textContent = liqLine.replace(/^[-*\s]*/, "").replace(/.*?(liquidity|liq):?/i, "").trim();
            } else {
                reasoningLiq.textContent = "Residing within key Value Area bounds.";
            }
            
            if (momLine) {
                reasoningMom.textContent = momLine.replace(/^[-*\s]*/, "").replace(/.*?(momentum|mom):?/i, "").trim();
            } else {
                reasoningMom.textContent = "StochRSI and MACD convergence.";
            }
            
            if (bookLine) {
                reasoningBook.textContent = bookLine.replace(/^[-*\s]*/, "").replace(/.*?(orderbook|book|sentiment):?/i, "").trim();
            } else {
                reasoningBook.textContent = "Binance orderbook limit skew delta is stable.";
            }
        } catch (err) {
            console.error("Error parsing reasoning bullets:", err);
        }
    };

    // Helper: extracts main trade plan (Entry, SL, TP) directly from AI narration text
    const parseAndRenderMainTradePlan = (narrationText) => {
        try {
            if (!narrationText) return;
            
            // Look for the "TRADE PLAN:" section in the text
            const lowerText = narrationText.toLowerCase();
            const sectionIdx = lowerText.indexOf("trade plan:");
            if (sectionIdx === -1) return;
            
            const rawSection = narrationText.substring(sectionIdx);
            
            // Regex to find Entry, Stop Loss, and Take Profit in the Trade Plan section
            const entryMatch = rawSection.match(/-\s*entry:\s*([\d.]+)/i);
            const slMatch = rawSection.match(/-\s*stop\s+loss:\s*([\d.]+)/i);
            const tpMatch = rawSection.match(/-\s*take\s+profit:\s*([\d.]+)/i);
            
            if (entryMatch) {
                valEntry.textContent = parseFloat(entryMatch[1]).toFixed(2);
            }
            if (slMatch) {
                valStopLoss.textContent = parseFloat(slMatch[1]).toFixed(2);
            }
            if (tpMatch) {
                valTakeProfit.textContent = parseFloat(tpMatch[1]).toFixed(2);
            }
        } catch (err) {
            console.error("Error parsing main trade plan:", err);
        }
    };

    // Fallback: Compiles text from snapshot if narration is empty
    const compileFallbackNarration = (snapshot) => {
        const regime = snapshot.market_regime;
        const trend = snapshot.strategies?.trend_bias || "neutral";
        const score = snapshot.strategies?.confluence_score || 0;
        
        return `
PAIR: ${snapshot.symbol}
BIAS: ${trend.toUpperCase()}

SMC Context: Structure ${snapshot.smc_context?.["15m"]?.structure || 'N/A'}, Nearest Bull OB [], Premium/Discount zone ${snapshot.smc_context?.["15m"]?.premium_discount?.zone || 'neutral'}

MARKET NARRATIVE:
The asset currently exhibits a ${regime.toLowerCase()} regime. Confluence Score is calculated at ${score} points based on multi-timeframe structural configurations.

TRADE PLAN:
Action: HOLD
Entry: N/A
Stop Loss: N/A
Take Profit: N/A
Confidence: ${score}%

HYPOTHETICAL IF-THEN SETUP (FOR UNCERTAIN/HOLD MARKETS):
- Most Possible Price Direction: ${trend.toUpperCase()}
- Key Levels to Watch: Value Area High at ${snapshot.price_action?.value_area_1h?.vah?.toFixed(2) || '0.00'} and VAL at ${snapshot.price_action?.value_area_1h?.val?.toFixed(2) || '0.00'}
- Trade Setup & Trigger Condition: Price sweep of key support levels followed by structure shift on low timeframe.
- Contingent Entry Level: ${snapshot.price_action?.value_area_1h?.poc?.toFixed(2) || '0.00'}
- Contingent Invalidation (SL): ${snapshot.price_action?.value_area_1h?.val?.toFixed(2) || '0.00'}
- Contingent Target (TP): ${snapshot.price_action?.value_area_1h?.vah?.toFixed(2) || '0.00'}
- Supporting Data & Confluence: Structural confluences on 15m order book profile.

REASONING:
- Structure: HTF structure aligns with dynamic average metrics.
- Liquidity & Volume: Price currently resides inside the Value Area boundaries.
- Momentum & Absorption: Stochastic RSI and volume delta signals are balanced.
- Orderbook & Sentiment: Limit order bid/ask delta is neutral.
        `;
    };

    // Run active Live analysis via API
    btnRunAnalysis.addEventListener("click", async () => {
        let symbol = symbolSelector.value;
        if (symbol === "CUSTOM") {
            symbol = customSymbolInput.value.trim().toUpperCase();
            if (!symbol || !symbol.includes("/")) {
                alert("Please enter a valid symbol format (e.g. LINK/USDT).");
                return;
            }
        }
        
        // Show loader and disable button
        btnRunAnalysis.disabled = true;
        const btnText = btnRunAnalysis.querySelector(".btn-text");
        const loader = btnRunAnalysis.querySelector(".loader");
        
        btnText.classList.add("hidden");
        loader.classList.remove("hidden");
        
        try {
            const response = await fetch("/api/run", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ symbol })
            });
            
            const data = await response.json();
            
            // Reset button
            btnRunAnalysis.disabled = false;
            btnText.classList.remove("hidden");
            loader.classList.add("hidden");
            
            if (data.success) {
                renderWorkspace(data.snapshot, data.narration);
                // Refresh list to include new snapshot
                refreshHistory();
            } else {
                alert(`Error executing Cryptera: ${data.error || 'Server error'}`);
            }
        } catch (err) {
            console.error("API Call error:", err);
            btnRunAnalysis.disabled = false;
            btnText.classList.remove("hidden");
            loader.classList.add("hidden");
            alert("Failed to connect to the Cryptera Flask backend API.");
        }
    });

    // Refresh history click
    btnRefreshHistory.addEventListener("click", refreshHistory);

    // Initializations
    refreshHistory();
});

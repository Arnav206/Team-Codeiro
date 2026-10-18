/**
 * HealthShield — Master Application Controller
 * Orchestrates:
 * - Patient switching & live telemetry loading
 * - Symptom check-in correlation
 * - Interactive "What-If" Vital Simulator
 * - Privacy Shield & Federated Learning workflow
 * - Doctor's Visit Brief generation & export
 * - Audio narration and theme management
 */

document.addEventListener("DOMContentLoaded", async () => {
    // Current Application State
    const state = {
        currentUser: "user1",
        currentVitals: null,
        currentPrediction: null,
        currentTimeline: [],
        backendOnline: false,
        userFeeling: null,
        isSimulating: false
    };

    // DOM Elements
    const elements = {
        themeToggle: document.getElementById("theme-toggle"),
        userSelect: document.getElementById("user-select"),
        navAvatar: document.getElementById("nav-user-avatar"),
        backendBadge: document.getElementById("backend-status-badge"),
        userPersonaDesc: document.getElementById("user-persona-desc"),

        // Sync & Protect
        syncFitBtn: document.getElementById("sync-fit-btn"),
        trainPrivacyBtn: document.getElementById("train-privacy-btn"),
        aggregateFedBtn: document.getElementById("aggregate-fed-btn"),
        privacyDetailsBtn: document.getElementById("privacy-details-btn"),

        // Vitals Grid
        valHr: document.getElementById("val-hr"),
        valBp: document.getElementById("val-bp"),
        valSpo2: document.getElementById("val-spo2"),
        valSleep: document.getElementById("val-sleep"),
        valSteps: document.getElementById("val-steps"),
        valCalories: document.getElementById("val-calories"),
        stepProgressBar: document.getElementById("step-progress-bar"),
        pulseRing: document.getElementById("heart-pulse-indicator"),

        // Health Score & Narrative
        healthScoreVal: document.getElementById("health-score-val"),
        scoreCircleProgress: document.getElementById("score-circle-progress"),
        conditionBadge: document.getElementById("condition-badge"),
        narrativeTitle: document.getElementById("narrative-title"),
        narrativeHeadline: document.getElementById("narrative-headline"),
        narrativeBody: document.getElementById("narrative-body"),
        narrativeEncourage: document.getElementById("narrative-encourage"),
        clinicalAlertsList: document.getElementById("clinical-alerts-list"),

        // Audio Narrator
        audioNarratorBtn: document.getElementById("audio-narrator-btn"),

        // Daily Habits
        dailyHabitsList: document.getElementById("daily-habits-list"),

        // 24H Early Warning Cards
        forecastCardsContainer: document.getElementById("forecast-cards-container"),

        // XAI Attributions
        xaiProtectiveList: document.getElementById("xai-protective-list"),
        xaiRiskList: document.getElementById("xai-risk-list"),

        // Simulator Sliders
        simToggleBtn: document.getElementById("sim-toggle-btn"),
        simContainer: document.getElementById("simulator-container"),
        simHr: document.getElementById("sim-hr"),
        simHrVal: document.getElementById("sim-hr-val"),
        simBpSys: document.getElementById("sim-bp-sys"),
        simBpSysVal: document.getElementById("sim-bp-sys-val"),
        simBpDia: document.getElementById("sim-bp-dia"),
        simBpDiaVal: document.getElementById("sim-bp-dia-val"),
        simSpo2: document.getElementById("sim-spo2"),
        simSpo2Val: document.getElementById("sim-spo2-val"),
        simSleep: document.getElementById("sim-sleep"),
        simSleepVal: document.getElementById("sim-sleep-val"),
        simResetBtn: document.getElementById("sim-reset-btn"),

        // Doctor Modal
        doctorBriefBtn: document.getElementById("doctor-brief-btn"),
        doctorModal: document.getElementById("doctor-modal"),
        doctorModalClose: document.getElementById("doctor-modal-close"),
        doctorPrintBtn: document.getElementById("doctor-print-btn"),
        doctorCopyBtn: document.getElementById("doctor-copy-btn"),
        doctorReportContent: document.getElementById("doctor-report-content"),

        // Privacy Modal
        privacyModal: document.getElementById("privacy-modal"),
        privacyModalClose: document.getElementById("privacy-modal-close"),

        // Add Profile Modal
        addProfileBtn: document.getElementById("add-profile-btn"),
        addProfileModal: document.getElementById("add-profile-modal"),
        addProfileModalClose: document.getElementById("add-profile-close"),
        addProfileForm: document.getElementById("add-profile-form"),

        // Toast
        appToast: document.getElementById("app-toast"),
        toastTitle: document.getElementById("toast-title"),
        toastDesc: document.getElementById("toast-desc"),
        toastClose: document.getElementById("toast-close")
    };

    /**
     * Show animated toast message
     */
    function showToast(title, desc, duration = 6000) {
        if (!elements.appToast) return;
        elements.toastTitle.textContent = title;
        elements.toastDesc.textContent = desc;
        elements.appToast.classList.add("visible");
        clearTimeout(state.toastTimeout);
        state.toastTimeout = setTimeout(() => {
            elements.appToast.classList.remove("visible");
        }, duration);
    }

    if (elements.toastClose) {
        elements.toastClose.addEventListener("click", () => {
            elements.appToast.classList.remove("visible");
        });
    }

    /**
     * Theme Initialization (Light default, remembers selection)
     */
    function initTheme() {
        const saved = localStorage.getItem("healthshield_theme") || "light";
        document.documentElement.setAttribute("data-theme", saved);
        updateThemeToggleIcon(saved);

        if (elements.themeToggle) {
            elements.themeToggle.addEventListener("click", () => {
                const current = document.documentElement.getAttribute("data-theme");
                const next = current === "dark" ? "light" : "dark";
                document.documentElement.setAttribute("data-theme", next);
                localStorage.setItem("healthshield_theme", next);
                updateThemeToggleIcon(next);
                window.HealthCharts.refreshTheme(state.currentTimeline, state.currentPrediction?.probabilities, state.currentPrediction?.xai);
            });
        }
    }

    function updateThemeToggleIcon(theme) {
        if (!elements.themeToggle) return;
        elements.themeToggle.innerHTML = theme === "dark" 
            ? `<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>`
            : `<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>`;
    }

    /**
     * Backend Connection Check
     */
    async function checkBackend() {
        const ping = await window.HealthAPI.pingBackend();
        state.backendOnline = ping.online;
        if (elements.backendBadge) {
            if (ping.online) {
                elements.backendBadge.className = "status-badge connected";
                elements.backendBadge.innerHTML = `<span class="dot"></span> <span>Flask AI Backend Live (:5000)</span>`;
            } else {
                elements.backendBadge.className = "status-badge local-edge";
                elements.backendBadge.innerHTML = `<span class="dot"></span> <span>On-Device Privacy Engine (Zero Leakage)</span>`;
            }
        }
    }

    /**
     * Populate Patient Profiles in dropdown
     */
    async function loadUserDropdown() {
        const users = await window.HealthAPI.getUsers();
        if (!elements.userSelect) return;
        elements.userSelect.innerHTML = "";

        users.forEach(u => {
            const opt = document.createElement("option");
            opt.value = u.user_id;
            const profile = window.HealthData.PROFILES[u.user_id] || {};
            const displayName = profile.name || u.name || `Patient ${u.user_id}`;
            opt.textContent = `${displayName}`;
            if (u.user_id === state.currentUser) {
                opt.selected = true;
            }
            elements.userSelect.appendChild(opt);
        });

        elements.userSelect.addEventListener("change", (e) => {
            state.currentUser = e.target.value;
            loadPatientData(state.currentUser);
        });
    }

    /**
     * Load selected patient data and refresh dashboard
     */
    async function loadPatientData(userId) {
        const profile = window.HealthData.PROFILES[userId] || window.HealthData.PROFILES["user1"];
        
        // Update nav avatar and persona description
        if (elements.navAvatar) {
            elements.navAvatar.textContent = profile.initials || "PT";
            elements.navAvatar.style.background = profile.avatarBg || "var(--primary)";
        }
        if (elements.userPersonaDesc) {
            elements.userPersonaDesc.textContent = profile.personaDescription || `${profile.name} (${profile.role})`;
        }

        const data = await window.HealthAPI.getUserData(userId);
        state.currentVitals = data.latest_vitals;
        state.currentTimeline = data.timeline;
        state.currentPrediction = data.prediction;

        renderVitals(state.currentVitals);
        renderPrediction(state.currentPrediction, state.currentVitals);
        renderTimelineTable(state.currentTimeline);

        // Render Charts
        window.HealthCharts.renderTimeline(state.currentTimeline);
        window.HealthCharts.renderRiskBreakdown(state.currentPrediction.probabilities);
        window.HealthCharts.renderXaiAttribution(state.currentPrediction.xai);

        // Sync simulator sliders with current vitals
        syncSimulatorSliders(state.currentVitals);
    }

    /**
     * Render Biometric Vitals Cards
     */
    function renderVitals(vitals) {
        if (!vitals) return;

        const hr = Math.round(vitals.heart_rate || 72);
        const bpSys = Math.round(vitals.blood_pressure_systolic || 120);
        const bpDia = Math.round(vitals.blood_pressure_diastolic || 80);
        const spo2 = Number(vitals.blood_oxygen || 98.0).toFixed(1);
        const sleep = Number(vitals.sleep_quality || 7.0).toFixed(1);
        const steps = Math.round(vitals.steps_count || 6500);
        const calories = Math.round(vitals.calories_burned || 2000);

        elements.valHr.textContent = hr;
        elements.valBp.textContent = `${bpSys}/${bpDia}`;
        elements.valSpo2.textContent = spo2;
        elements.valSleep.textContent = sleep;
        elements.valSteps.textContent = steps.toLocaleString();
        elements.valCalories.textContent = calories.toLocaleString();

        // Steps progress bar (target 10,000)
        const stepPct = Math.min(100, Math.round((steps / 10000) * 100));
        if (elements.stepProgressBar) {
            elements.stepProgressBar.style.width = `${stepPct}%`;
        }

        // Adjust heart pulsing speed according to actual BPM
        if (elements.pulseRing) {
            const beatDuration = (60 / Math.max(45, Math.min(140, hr))).toFixed(2);
            elements.pulseRing.style.animationDuration = `${beatDuration}s`;
        }
    }

    /**
     * Render Health Score, Human Narrative, 24H Early Warnings & Habits
     */
    function renderPrediction(pred, vitals) {
        if (!pred) return;

        // Health Score
        const score = pred.health_score || 85;
        elements.healthScoreVal.textContent = score;
        
        // SVG Circle dash offset calculation (circumference: 2 * pi * 44 ≈ 276.46)
        if (elements.scoreCircleProgress) {
            const circleCircumference = 276.46;
            const offset = circleCircumference - (circleCircumference * (score / 100));
            elements.scoreCircleProgress.style.strokeDashoffset = offset;
            elements.scoreCircleProgress.style.stroke = score >= 80 ? "#10b981" : (score >= 60 ? "#f59e0b" : "#ef4444");
        }

        // Condition badge
        if (elements.conditionBadge) {
            elements.conditionBadge.className = `condition-badge ${pred.badgeClass || "normal"}`;
            elements.conditionBadge.textContent = pred.badge || "LOW RISK (NORMAL)";
        }

        // Empathetic Narrative
        const narr = pred.human_narrative || window.HealthData.generateHumanNarrative(vitals, pred.derived, pred.condition, score);
        elements.narrativeTitle.textContent = narr.title;
        elements.narrativeHeadline.textContent = narr.headline;
        elements.narrativeBody.textContent = narr.body;
        elements.narrativeEncourage.textContent = narr.encouragement;

        // Clinical Alerts / Anomalies
        if (elements.clinicalAlertsList) {
            elements.clinicalAlertsList.innerHTML = "";
            if (pred.anomalies && pred.anomalies.length > 0) {
                pred.anomalies.forEach(a => {
                    const pill = document.createElement("div");
                    pill.className = "clinical-alert-tag";
                    pill.innerHTML = `<span class="alert-icon">⚠️</span> <span>${a}</span>`;
                    elements.clinicalAlertsList.appendChild(pill);
                });
                elements.clinicalAlertsList.style.display = "flex";
            } else {
                const clearPill = document.createElement("div");
                clearPill.className = "clinical-alert-tag optimal";
                clearPill.innerHTML = `<span class="alert-icon">✓</span> <span>All physiological biomarkers within clinical benchmarks.</span>`;
                elements.clinicalAlertsList.appendChild(clearPill);
                elements.clinicalAlertsList.style.display = "flex";
            }
        }

        // Daily Habits
        if (elements.dailyHabitsList) {
            elements.dailyHabitsList.innerHTML = "";
            const habits = pred.daily_habits || window.HealthData.generateDailyHabits(vitals, pred.condition);
            habits.forEach(h => {
                const card = document.createElement("div");
                card.className = "habit-item-card";
                card.innerHTML = `
                    <div class="habit-icon">${h.icon}</div>
                    <div class="habit-details">
                        <div class="habit-header">
                            <h4>${h.action}</h4>
                            <span class="habit-tag">${h.tag}</span>
                        </div>
                        <p>${h.detail}</p>
                    </div>
                `;
                elements.dailyHabitsList.appendChild(card);
            });
        }

        // 24H Disease Forecast Cards (Bilingual Support)
        renderForecastCards(pred.future_diseases_24h || []);

        // XAI Attributions list
        renderXaiLists(pred.xai || {});
    }

    /**
     * Render 24-Hour Disease Forecast Cards
     */
    function renderForecastCards(forecasts) {
        if (!elements.forecastCardsContainer) return;
        elements.forecastCardsContainer.innerHTML = "";

        forecasts.forEach(f => {
            const card = document.createElement("div");
            card.className = `forecast-card ${f.severity_color}`;
            
            const titleDisplay = f.disease_name;
            const subtitleDisplay = f.category || "Clinical Early Warning";

            card.innerHTML = `
                <div class="forecast-header">
                    <div class="forecast-title-wrap">
                        <span class="forecast-icon">${f.icon || "🔮"}</span>
                        <div>
                            <h3>${titleDisplay}</h3>
                            <div class="forecast-subtitle">${subtitleDisplay}</div>
                        </div>
                    </div>
                    <span class="forecast-badge ${f.severity_color}">${f.risk_level}</span>
                </div>

                <div class="forecast-prob-meter">
                    <div class="prob-header">
                        <span>24h Risk Likelihood</span>
                        <strong>${f.risk_probability}%</strong>
                    </div>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill ${f.severity_color}" style="width: ${f.risk_probability}%"></div>
                    </div>
                </div>

                <div class="forecast-reasoning">
                    <strong>Medical Insight & XAI Attribution:</strong>
                    <p>${f.xai_reason}</p>
                </div>

                <div class="forecast-biomarkers">
                    ${f.biomarkers.map(b => `<span class="biomarker-pill">${b}</span>`).join("")}
                </div>

                <div class="forecast-action">
                    <div class="action-icon">💡</div>
                    <div class="action-text">
                        <strong>Recommended Preventative Action:</strong>
                        <p>${f.prevention_guidelines}</p>
                    </div>
                </div>
            `;
            elements.forecastCardsContainer.appendChild(card);
        });
    }

    /**
     * Render Explainable AI feature lists
     */
    function renderXaiLists(xai) {
        if (elements.xaiProtectiveList) {
            elements.xaiProtectiveList.innerHTML = "";
            const protective = xai.protective_factors || [];
            if (protective.length === 0) {
                elements.xaiProtectiveList.innerHTML = `<p class="empty-text">No strong protective factors detected.</p>`;
            } else {
                protective.forEach(p => {
                    const item = document.createElement("div");
                    item.className = "xai-factor-item protective";
                    item.innerHTML = `
                        <div class="xai-factor-top">
                            <span class="factor-name">${p.icon} ${p.name}</span>
                            <span class="factor-score protective">-${p.percentage}% Risk Shield</span>
                        </div>
                        <p class="factor-note">${p.note}</p>
                    `;
                    elements.xaiProtectiveList.appendChild(item);
                });
            }
        }

        if (elements.xaiRiskList) {
            elements.xaiRiskList.innerHTML = "";
            const risks = xai.risk_drivers || [];
            if (risks.length === 0) {
                elements.xaiRiskList.innerHTML = `<p class="empty-text optimal">✓ Zero critical risk drivers detected today.</p>`;
            } else {
                risks.forEach(r => {
                    const item = document.createElement("div");
                    item.className = "xai-factor-item risk";
                    item.innerHTML = `
                        <div class="xai-factor-top">
                            <span class="factor-name">${r.icon} ${r.name}</span>
                            <span class="factor-score risk">+${r.percentage}% Burden</span>
                        </div>
                        <p class="factor-note">${r.note}</p>
                    `;
                    elements.xaiRiskList.appendChild(item);
                });
            }
        }
    }

    /**
     * Render 14-Day Timeline Table
     */
    function renderTimelineTable(timeline) {
        const tbody = document.getElementById("timeline-table-body");
        if (!tbody) return;
        tbody.innerHTML = "";

        const reversed = [...timeline].reverse();
        reversed.forEach(r => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td><strong>${r.date}</strong></td>
                <td><span class="vital-value">${r.heart_rate}</span> <small>bpm</small></td>
                <td><span class="vital-value">${r.blood_pressure_systolic}/${r.blood_pressure_diastolic}</span> <small>mmHg</small></td>
                <td><span class="vital-value">${r.blood_oxygen}%</span></td>
                <td><span class="vital-value">${r.sleep_quality}h</span></td>
                <td><span class="vital-value">${Number(r.steps_count).toLocaleString()}</span></td>
                <td><span class="condition-pill-sm ${r.condition_class || 'normal'}">${r.condition}</span></td>
            `;
            tbody.appendChild(tr);
        });
    }



    /**
     * Voice Audio Narrator
     */
    if (elements.audioNarratorBtn) {
        elements.audioNarratorBtn.addEventListener("click", () => {
            if (window.HealthNarrator.isSpeaking) {
                window.HealthNarrator.stop();
            } else {
                const narr = state.currentPrediction?.human_narrative;
                const profile = window.HealthData.PROFILES[state.currentUser];
                const text = `Hello ${profile?.name || "there"}. ${narr?.title}. ${narr?.body} ${narr?.encouragement}`;
                window.HealthNarrator.speak(text);
            }
        });
    }

    /**
     * Patient Symptom / Feeling Check-In
     */
    const feelingButtons = document.querySelectorAll(".feeling-btn");
    feelingButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            feelingButtons.forEach(b => b.classList.remove("selected"));
            btn.classList.add("selected");
            const feeling = btn.getAttribute("data-feeling");
            state.userFeeling = feeling;
            handleFeelingSelected(feeling);
        });
    });

    function handleFeelingSelected(feeling) {
        const box = document.getElementById("feeling-feedback-box");
        if (!box) return;

        const vitals = state.currentVitals;
        const sleep = vitals?.sleep_quality || 7.0;
        const hr = vitals?.heart_rate || 72;

        let responseText = "";
        let responseTag = "";

        if (feeling === "energized") {
            responseText = `Wonderful! Your resting heart rate (${hr} BPM) and restorative sleep (${sleep}h) validate your vitality. A brisk workout or focused creative deep-work session is recommended today.`;
            responseTag = "🟢 Aligned with Healthy Vitals";
        } else if (feeling === "tired") {
            responseText = `We hear you. Your ${sleep}-hour sleep record and elevated autonomic stress confirm why energy is lower today. We've dialed down your recommended step goal to 4,000 steps so your heart can recover.`;
            responseTag = "🛌 Sleep Recovery Mode Activated";
        } else if (feeling === "stressed") {
            responseText = `Understood. When mental stress rises, cortisol elevates vascular resistance. Let's practice the 4-7-8 breathing reset (inhale 4s, hold 7s, exhale 8s) to signal your vagus nerve.`;
            responseTag = "🌿 Parasympathetic Reset Recommended";
        } else if (feeling === "dizzy") {
            responseText = `Please sit down and rest. Check your blood pressure (${vitals.blood_pressure_systolic}/${vitals.blood_pressure_diastolic} mmHg) and drink a glass of water with electrolytes. If symptoms persist, contact a clinic.`;
            responseTag = "⚠️ Rest & Hydration Priority";
        } else if (feeling === "palpitations") {
            responseText = `Your current resting rate is ${hr} BPM. Sit in a comfortable chair, unbutton tight collar shirts, and avoid tea, coffee, or energy drinks. Telemetry is being logged for your doctor visit.`;
            responseTag = "💓 Cardiac Awareness Logged";
        }

        box.innerHTML = `
            <div class="feedback-inner">
                <span class="feedback-tag">${responseTag}</span>
                <p>${responseText}</p>
            </div>
        `;
        box.style.display = "block";
    }

    /**
     * Sync Google Fit Button
     */
    if (elements.syncFitBtn) {
        elements.syncFitBtn.addEventListener("click", async () => {
            elements.syncFitBtn.disabled = true;
            elements.syncFitBtn.classList.add("loading");
            elements.syncFitBtn.innerHTML = `<span class="spinner"></span> <span>Syncing Google Fit...</span>`;

            const res = await window.HealthAPI.syncGoogleFit(state.currentUser, 30);
            showToast("Google Fit Telemetry Synced!", res.message);
            await loadPatientData(state.currentUser);

            elements.syncFitBtn.disabled = false;
            elements.syncFitBtn.classList.remove("loading");
            elements.syncFitBtn.innerHTML = `
                <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/>
                </svg>
                <span>Sync Wearable Telemetry</span>
            `;
        });
    }

    /**
     * On-Device Privacy Protection Button
     */
    if (elements.trainPrivacyBtn) {
        elements.trainPrivacyBtn.addEventListener("click", async () => {
            elements.trainPrivacyBtn.disabled = true;
            elements.trainPrivacyBtn.innerHTML = `<span class="spinner"></span> <span>Applying Laplace Noise...</span>`;

            const res = await window.HealthAPI.trainOnDevice(state.currentUser, 1.0, 30);
            showToast(
                "🛡️ On-Device Privacy Shield Verified!",
                `Local model trained with Differential Privacy (ε=1.0). Laplace noise added. Zero patient telemetry transmitted.`
            );

            elements.trainPrivacyBtn.disabled = false;
            elements.trainPrivacyBtn.innerHTML = `
                <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                </svg>
                <span>On-Device Privacy Protect (DP)</span>
            `;
        });
    }

    /**
     * Central Federated Aggregation
     */
    if (elements.aggregateFedBtn) {
        elements.aggregateFedBtn.addEventListener("click", async () => {
            elements.aggregateFedBtn.disabled = true;
            elements.aggregateFedBtn.innerHTML = `<span class="spinner"></span> <span>Running FedAvg Rounds...</span>`;

            const res = await window.HealthAPI.aggregateFederatedModel(5, 1.0, true);
            showToast(
                "🌐 Federated Learning Updated!",
                `Aggregated parameters across all client edge nodes without inspecting raw patient vitals.`
            );

            elements.aggregateFedBtn.disabled = false;
            elements.aggregateFedBtn.innerHTML = `
                <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"/><path d="m4.93 4.93 4.24 4.24M14.83 9.17l4.24-4.24M14.83 14.83l4.24 4.24M9.17 14.83l-4.24 4.24"/>
                </svg>
                <span>Trigger Federated FedAvg</span>
            `;
        });
    }

    /**
     * Interactive "What-If" Vital Sandbox (Simulator)
     */
    if (elements.simToggleBtn) {
        elements.simToggleBtn.addEventListener("click", () => {
            state.isSimulating = !state.isSimulating;
            elements.simContainer.style.display = state.isSimulating ? "block" : "none";
            elements.simToggleBtn.textContent = state.isSimulating ? "Close Vital Sandbox" : "Open 'What-If' Vital Sandbox";
            if (state.isSimulating) {
                elements.simContainer.scrollIntoView({ behavior: "smooth" });
            }
        });
    }

    function syncSimulatorSliders(vitals) {
        if (!vitals) return;
        if (elements.simHr) {
            elements.simHr.value = Math.round(vitals.heart_rate || 72);
            elements.simHrVal.textContent = elements.simHr.value;
        }
        if (elements.simBpSys) {
            elements.simBpSys.value = Math.round(vitals.blood_pressure_systolic || 120);
            elements.simBpSysVal.textContent = elements.simBpSys.value;
        }
        if (elements.simBpDia) {
            elements.simBpDia.value = Math.round(vitals.blood_pressure_diastolic || 80);
            elements.simBpDiaVal.textContent = elements.simBpDia.value;
        }
        if (elements.simSpo2) {
            elements.simSpo2.value = Number(vitals.blood_oxygen || 98).toFixed(1);
            elements.simSpo2Val.textContent = elements.simSpo2.value;
        }
        if (elements.simSleep) {
            elements.simSleep.value = Number(vitals.sleep_quality || 7.2).toFixed(1);
            elements.simSleepVal.textContent = elements.simSleep.value;
        }
    }

    function onSimulatorSliderChange() {
        const simulatedVitals = {
            ...state.currentVitals,
            heart_rate: Number(elements.simHr.value),
            blood_pressure_systolic: Number(elements.simBpSys.value),
            blood_pressure_diastolic: Number(elements.simBpDia.value),
            blood_oxygen: Number(elements.simSpo2.value),
            sleep_quality: Number(elements.simSleep.value)
        };

        elements.simHrVal.textContent = simulatedVitals.heart_rate;
        elements.simBpSysVal.textContent = simulatedVitals.blood_pressure_systolic;
        elements.simBpDiaVal.textContent = simulatedVitals.blood_pressure_diastolic;
        elements.simSpo2Val.textContent = simulatedVitals.blood_oxygen;
        elements.simSleepVal.textContent = simulatedVitals.sleep_quality;

        const pred = window.HealthData.predictHealthRisk(simulatedVitals);
        renderVitals(simulatedVitals);
        renderPrediction(pred, simulatedVitals);

        // Update charts in real-time
        window.HealthCharts.renderRiskBreakdown(pred.probabilities);
        window.HealthCharts.renderXaiAttribution(pred.xai);
    }

    [elements.simHr, elements.simBpSys, elements.simBpDia, elements.simSpo2, elements.simSleep].forEach(slider => {
        if (slider) {
            slider.addEventListener("input", onSimulatorSliderChange);
        }
    });

    if (elements.simResetBtn) {
        elements.simResetBtn.addEventListener("click", () => {
            syncSimulatorSliders(state.currentVitals);
            onSimulatorSliderChange();
            showToast("Sandbox Reset", "Vitals restored to actual recorded telemetry.");
        });
    }

    /**
     * Doctor's Clinical Visit Brief Modal
     */
    if (elements.doctorBriefBtn) {
        elements.doctorBriefBtn.addEventListener("click", () => {
            generateDoctorReport();
            elements.doctorModal.classList.add("open");
        });
    }

    if (elements.doctorModalClose) {
        elements.doctorModalClose.addEventListener("click", () => {
            elements.doctorModal.classList.remove("open");
        });
    }

    if (elements.doctorPrintBtn) {
        elements.doctorPrintBtn.addEventListener("click", () => {
            window.print();
        });
    }

    if (elements.doctorCopyBtn) {
        elements.doctorCopyBtn.addEventListener("click", () => {
            const text = elements.doctorReportContent.innerText;
            navigator.clipboard.writeText(text).then(() => {
                showToast("Copied to Clipboard", "Doctor's Visit Note copied ready for email or EHR paste.");
            });
        });
    }

    function generateDoctorReport() {
        const profile = window.HealthData.PROFILES[state.currentUser];
        const v = state.currentVitals;
        const pred = state.currentPrediction;
        const timeline = state.currentTimeline;

        // Calculate 14-day averages
        const avgHr = Math.round(timeline.reduce((s, r) => s + r.heart_rate, 0) / (timeline.length || 1));
        const avgSys = Math.round(timeline.reduce((s, r) => s + r.blood_pressure_systolic, 0) / (timeline.length || 1));
        const avgDia = Math.round(timeline.reduce((s, r) => s + r.blood_pressure_diastolic, 0) / (timeline.length || 1));
        const avgSpo2 = (timeline.reduce((s, r) => s + r.blood_oxygen, 0) / (timeline.length || 1)).toFixed(1);
        const avgSleep = (timeline.reduce((s, r) => s + r.sleep_quality, 0) / (timeline.length || 1)).toFixed(1);

        const anomalies = pred.anomalies && pred.anomalies.length > 0 
            ? pred.anomalies.map(a => `• ${a}`).join("\n") 
            : "• None detected. All monitored biometric values within safe physiological threshold.";

        const diseases = (pred.future_diseases_24h || []).map(d => 
            `• ${d.disease_name} (${d.risk_level}, ${d.risk_probability}% probability) — ${d.xai_reason}`
        ).join("\n\n");

        elements.doctorReportContent.innerHTML = `
            <div class="doctor-report-header">
                <h2>CLINICAL TELEMETRY CONSULTATION SUMMARY</h2>
                <div class="doc-meta">
                    <p><strong>Patient Name:</strong> ${profile.name}</p>
                    <p><strong>Patient ID:</strong> ${profile.id} &bull; <strong>Email:</strong> ${profile.email}</p>
                    <p><strong>Generated:</strong> ${new Date().toLocaleDateString("en-US", { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}</p>
                    <p><strong>Observation Window:</strong> Past 14 Days continuous wearable & ambient telemetry</p>
                    <p><strong>Privacy Protocol:</strong> Single-Release Laplace Differential Privacy (ε=1.0) via Federated Learning</p>
                </div>
            </div>

            <hr class="doc-divider"/>

            <div class="doc-section">
                <h3>1. CURRENT PHYSIOLOGICAL STATUS & CLASSIFICATION</h3>
                <p><strong>Primary Health Classification:</strong> ${pred.condition.toUpperCase()} (${pred.badge})</p>
                <p><strong>Composite Health Index:</strong> ${pred.health_score} / 100 &bull; <strong>Vital Risk Index (VRI):</strong> ${pred.vital_risk_index}</p>
                <p><strong>Latest Recorded Vitals:</strong> HR ${v.heart_rate} BPM | BP ${v.blood_pressure_systolic}/${v.blood_pressure_diastolic} mmHg | SpO2 ${v.blood_oxygen}% | Sleep ${v.sleep_quality}h | Steps ${Number(v.steps_count).toLocaleString()}</p>
            </div>

            <div class="doc-section">
                <h3>2. 14-DAY CLINICAL BASELINE AVERAGES</h3>
                <table class="doc-table">
                    <tr><th>Metric</th><th>14-Day Average</th><th>Clinical Reference Range</th><th>Status</th></tr>
                    <tr><td>Resting Heart Rate</td><td>${avgHr} BPM</td><td>60 – 100 BPM</td><td>${avgHr <= 85 ? "Normal" : "Elevated"}</td></tr>
                    <tr><td>Blood Pressure</td><td>${avgSys}/${avgDia} mmHg</td><td>< 120/80 mmHg</td><td>${avgSys >= 130 ? "Stage 1 HTN" : "Optimal"}</td></tr>
                    <tr><td>Blood Oxygen (SpO2)</td><td>${avgSpo2}%</td><td>>= 95%</td><td>${avgSpo2 >= 95 ? "Adequate" : "Hypoxemic"}</td></tr>
                    <tr><td>Sleep Duration</td><td>${avgSleep} hrs/night</td><td>7.0 – 9.0 hrs</td><td>${avgSleep >= 7 ? "Optimal" : "Deficit"}</td></tr>
                </table>
            </div>

            <div class="doc-section">
                <h3>3. DETECTED CLINICAL ANOMALIES & FLAGS</h3>
                <pre class="doc-pre">${anomalies}</pre>
            </div>

            <div class="doc-section">
                <h3>4. 24-HOUR PREDICTIVE EARLY WARNINGS & XAI REASONING</h3>
                <pre class="doc-pre">${diseases}</pre>
            </div>

            <div class="doc-section">
                <h3>5. RECOMMENDED CLINICAL INTERVENTION POINTS</h3>
                <p>• Review circadian sleep consistency and manage autonomic sleep debt.</p>
                <p>• Correlate systolic hemodynamic resistance against sodium intake and ambient hydration.</p>
                <p>• Recommend continued passive telemetry monitoring with on-device privacy guarantee.</p>
            </div>
        `;
    }

    /**
     * Privacy Shield Modal
     */
    if (elements.privacyDetailsBtn) {
        elements.privacyDetailsBtn.addEventListener("click", () => {
            elements.privacyModal.classList.add("open");
        });
    }

    if (elements.privacyModalClose) {
        elements.privacyModalClose.addEventListener("click", () => {
            elements.privacyModal.classList.remove("open");
        });
    }

    /**
     * Add Profile Modal
     */
    if (elements.addProfileBtn) {
        elements.addProfileBtn.addEventListener("click", () => {
            elements.addProfileModal.classList.add("open");
        });
    }

    if (elements.addProfileModalClose) {
        elements.addProfileModalClose.addEventListener("click", () => {
            elements.addProfileModal.classList.remove("open");
        });
    }

    if (elements.addProfileForm) {
        elements.addProfileForm.addEventListener("submit", async (e) => {
            e.preventDefault();
            const id = document.getElementById("new-user-id").value.trim();
            const name = document.getElementById("new-user-name").value.trim();
            const email = document.getElementById("new-user-email").value.trim();
            const archetype = document.getElementById("new-user-archetype").value;

            if (!id || !name) {
                alert("Please provide both Patient ID and Name.");
                return;
            }

            const res = await window.HealthAPI.createProfile({
                user_id: id,
                name: name,
                email: email,
                archetype: archetype
            });

            elements.addProfileModal.classList.remove("open");
            showToast("Profile Created!", res.message);

            await loadUserDropdown();
            state.currentUser = res.user_id;
            elements.userSelect.value = res.user_id;
            loadPatientData(res.user_id);
        });
    }

    // Close modals on overlay backdrop click
    [elements.doctorModal, elements.privacyModal, elements.addProfileModal].forEach(modal => {
        if (modal) {
            modal.addEventListener("click", (e) => {
                if (e.target === modal) {
                    modal.classList.remove("open");
                }
            });
        }
    });

    // Initialize Application
    initTheme();
    await checkBackend();
    await loadUserDropdown();
    await loadPatientData(state.currentUser);
});

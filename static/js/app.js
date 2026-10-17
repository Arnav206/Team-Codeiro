/* ==========================================================================
   HealthSync AI — Client Application Logic
   Handles API communication, real-time telemetry rendering, and Chart.js
   ========================================================================== */

document.addEventListener("DOMContentLoaded", () => {
    // Tab Switching
    const tabs = document.querySelectorAll(".nav-tab");
    const panes = document.querySelectorAll(".tab-pane");

    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            tabs.forEach(t => t.classList.remove("active"));
            panes.forEach(p => p.classList.remove("active"));

            tab.classList.add("active");
            const targetPane = document.getElementById(tab.dataset.tab);
            if (targetPane) targetPane.classList.add("active");
        });
    });

    // Chart references
    let timelineChart = null;
    let riskProbChart = null;

    // Elements
    const userSelect = document.getElementById("user-select");
    const refreshFitBtn = document.getElementById("refresh-fit-btn");
    const trainOnDeviceBtn = document.getElementById("train-ondevice-btn");
    const triggerFedAvgBtn = document.getElementById("trigger-fedavg-btn");
    const clientTerminal = document.getElementById("client-terminal-logs");
    const serverTerminal = document.getElementById("server-terminal-logs");

    // Logging helpers
    function logClient(msg, type = "info") {
        const p = document.createElement("p");
        p.className = `term-line ${type}`;
        p.textContent = msg;
        clientTerminal.appendChild(p);
        clientTerminal.scrollTop = clientTerminal.scrollHeight;
    }

    function logServer(msg, type = "info") {
        const p = document.createElement("p");
        p.className = `term-line ${type}`;
        p.textContent = msg;
        serverTerminal.appendChild(p);
        serverTerminal.scrollTop = serverTerminal.scrollHeight;
    }

    // Initialize System Status
    async function loadSystemStatus() {
        try {
            const res = await fetch("/api/status");
            const data = await res.json();
            document.getElementById("header-client-count").textContent = `${data.active_client_count} Nodes Active`;
            document.getElementById("stat-active-nodes").textContent = data.active_client_count;
            document.getElementById("stat-global-acc").textContent = `${data.last_global_accuracy}%`;
            document.getElementById("stat-epsilon").textContent = data.privacy_budget_epsilon;
            document.getElementById("stat-records").textContent = data.total_records_processed;
        } catch (err) {
            console.error("Status load failed", err);
        }
    }

    // Initialize User Dropdown
    async function loadUserList() {
        try {
            const res = await fetch("/api/users");
            const data = await res.json();
            userSelect.innerHTML = "";
            data.users.forEach(u => {
                const opt = document.createElement("option");
                opt.value = u.user_id;
                opt.textContent = `${u.user_id.toUpperCase()} ${u.authenticated ? '(Verified Google Fit)' : '(Default)'}`;
                userSelect.appendChild(opt);
            });
            if (data.users.length > 0) {
                loadPatientData(userSelect.value);
            }
        } catch (err) {
            console.error("Users load failed", err);
        }
    }

    // Load Specific Patient Telemetry
    async function loadPatientData(userId) {
        try {
            const res = await fetch(`/api/user/${userId}/data`);
            const data = await res.json();

            const vitals = data.latest_vitals || {};
            const pred = data.prediction || {};

            // Update Vitals Cards
            document.getElementById("vital-hr").textContent = vitals.heart_rate ? Math.round(vitals.heart_rate) : "--";
            document.getElementById("vital-bp").textContent = (vitals.blood_pressure_systolic && vitals.blood_pressure_diastolic) 
                ? `${Math.round(vitals.blood_pressure_systolic)}/${Math.round(vitals.blood_pressure_diastolic)}` : "--/--";
            document.getElementById("vital-spo2").textContent = vitals.blood_oxygen ? Math.round(vitals.blood_oxygen) : "--";
            
            const steps = vitals.steps_count ? Math.round(vitals.steps_count) : 0;
            document.getElementById("vital-steps").textContent = steps.toLocaleString();
            document.getElementById("steps-bar").style.width = `${Math.min(100, (steps / 10000) * 100)}%`;

            document.getElementById("vital-sleep").textContent = vitals.sleep_quality ? (vitals.sleep_quality).toFixed(1) : "--";
            document.getElementById("vital-calories").textContent = vitals.calories_burned ? Math.round(vitals.calories_burned) : "--";

            // Update Condition Banner
            const banner = document.getElementById("condition-banner");
            const badge = document.getElementById("condition-badge");
            const title = document.getElementById("condition-title");
            const sub = document.getElementById("condition-subtitle");
            const vri = document.getElementById("vital-risk-num");

            banner.className = "condition-banner glass-panel";
            vri.textContent = pred.vital_risk_index !== undefined ? pred.vital_risk_index : "--";

            const cond = pred.condition || "Normal";
            if (cond === "Normal") {
                badge.textContent = "LOW RISK (NORMAL)";
                title.textContent = "Optimal Health Telemetry";
                sub.textContent = "All daily biometric streams within verified physiological ranges.";
            } else if (cond === "Mild Event") {
                banner.classList.add("mild");
                badge.textContent = "MILD RISK (MONITOR)";
                title.textContent = "Mild Health Event Detected";
                sub.textContent = "Elevated heart rate or sleep deficit observed. Continuous monitoring recommended.";
            } else if (cond === "Moderate Event") {
                banner.classList.add("moderate");
                badge.textContent = "MODERATE RISK (ATTENTION NEEDED)";
                title.textContent = "Moderate Vital Strain Warning";
                sub.textContent = "Pre-hypertension or SpO2 dip detected. Consult primary physician.";
            } else {
                banner.classList.add("severe");
                badge.textContent = "HIGH RISK (CRITICAL EVENT)";
                title.textContent = "Critical Physiological Alert";
                sub.textContent = "Severe vitals anomaly detected! Immediate clinical attention required.";
            }

            // Update Alerts
            const alertsBox = document.getElementById("alerts-container");
            alertsBox.innerHTML = "";
            if (pred.anomalies && pred.anomalies.length > 0) {
                alertsBox.style.display = "flex";
                pred.anomalies.forEach(a => {
                    const div = document.createElement("div");
                    div.className = "alert-item";
                    div.innerHTML = `<strong>⚠️ Alert:</strong> ${a}`;
                    alertsBox.appendChild(div);
                });
            } else {
                alertsBox.style.display = "none";
            }

            // Render Charts
            renderTimelineChart(data.timeline || []);
            renderRiskChart(pred.probabilities || {});

        } catch (err) {
            console.error("Patient data load failed", err);
        }
    }

    // Chart 1: Telemetry Timeline
    function renderTimelineChart(timeline) {
        const ctx = document.getElementById("timelineChart").getContext("2d");
        const labels = timeline.map((_, idx) => `Day ${idx + 1}`);
        const hrData = timeline.map(t => t.heart_rate);
        const stepsData = timeline.map(t => t.steps / 100); // scaled for visual harmony

        if (timelineChart) timelineChart.destroy();

        timelineChart = new Chart(ctx, {
            type: "line",
            data: {
                labels: labels,
                datasets: [
                    {
                        label: "Heart Rate (BPM)",
                        data: hrData,
                        borderColor: "#f43f5e",
                        backgroundColor: "rgba(244, 63, 94, 0.1)",
                        tension: 0.35,
                        fill: true,
                        yAxisID: "y"
                    },
                    {
                        label: "Steps (x100)",
                        data: stepsData,
                        borderColor: "#06b6d4",
                        backgroundColor: "transparent",
                        borderDash: [5, 5],
                        tension: 0.3,
                        yAxisID: "y"
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: "#94a3b8", font: { family: 'Inter', size: 11 } } }
                },
                scales: {
                    x: { grid: { color: "rgba(255,255,255,0.04)" }, ticks: { color: "#64748b" } },
                    y: { grid: { color: "rgba(255,255,255,0.04)" }, ticks: { color: "#64748b" } }
                }
            }
        });
    }

    // Chart 2: Disease Probability Bar
    function renderRiskChart(probs) {
        const ctx = document.getElementById("riskProbChart").getContext("2d");
        const labels = ["Normal", "Mild", "Moderate", "Severe"];
        const data = [
            (probs["Normal"] || 0) * 100,
            (probs["Mild Event"] || 0) * 100,
            (probs["Moderate Event"] || 0) * 100,
            (probs["Severe Event"] || 0) * 100
        ];

        if (riskProbChart) riskProbChart.destroy();

        riskProbChart = new Chart(ctx, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [{
                    label: "Probability (%)",
                    data: data,
                    backgroundColor: [
                        "rgba(16, 185, 129, 0.7)",
                        "rgba(6, 182, 212, 0.7)",
                        "rgba(245, 158, 11, 0.7)",
                        "rgba(244, 63, 94, 0.7)"
                    ],
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { grid: { display: false }, ticks: { color: "#94a3b8" } },
                    y: { max: 100, grid: { color: "rgba(255,255,255,0.04)" }, ticks: { color: "#64748b" } }
                }
            }
        });
    }

    // Event: User Select Change
    userSelect.addEventListener("change", () => {
        loadPatientData(userSelect.value);
    });

    // Event: Sync Google Fit
    refreshFitBtn.addEventListener("click", async () => {
        const u = userSelect.value;
        refreshFitBtn.disabled = true;
        refreshFitBtn.textContent = "Syncing...";
        logClient(`[SYNC] Querying Google Fitness REST API for user '${u}'...`, "cyan");

        try {
            const res = await fetch(`/api/user/${u}/fetch`, { method: "POST" });
            const data = await res.json();
            if (data.status === "success") {
                logClient(`[SUCCESS] ${data.message}`, "success");
                await loadPatientData(u);
            } else {
                logClient(`[ERROR] ${data.message}`, "warning");
            }
        } catch (err) {
            logClient(`[ERROR] Failed to connect: ${err}`, "warning");
        } finally {
            refreshFitBtn.disabled = false;
            refreshFitBtn.innerHTML = `
                <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/>
                </svg>
                <span>Sync Google Fit Data</span>`;
        }
    });

    // Event: Train On-Device & Send DP Weights
    trainOnDeviceBtn.addEventListener("click", async () => {
        const u = userSelect.value;
        trainOnDeviceBtn.disabled = true;
        trainOnDeviceBtn.textContent = "Training Local Model...";

        logClient(`\n------------------------------------------------------------`, "info");
        logClient(`[STEP 1] Starting on-device local SGD training for '${u}'...`, "cyan");
        logClient(`[STEP 2] Standardizing features with zero-leakage scalars...`, "info");

        try {
            const res = await fetch(`/api/user/${u}/train`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ epsilon: 1.0, epochs: 30 })
            });
            const data = await res.json();

            if (data.status === "success") {
                const dp = data.differential_privacy;
                logClient(`[STEP 3] Local SGD model trained on ${data.trained_samples} samples.`, "success");
                logClient(`[PRIVACY] Injecting Laplace Noise with privacy budget epsilon=${dp.epsilon}`, "cyan");
                logClient(`[PRIVACY] Sensitivity Clipping: R_clip=${dp.sensitivity_clip}`, "info");
                logClient(`[PRIVACY] Noisy Weights L2-Norm: ${dp.noisy_weights_l2_norm}`, "info");
                logClient(`[VERIFIED] Raw patient data leaked: ${dp.raw_patient_data_leaked ? 'YES' : 'NO (100% Secure)'}`, "success");
                logClient(`[STEP 4] Noisy DP weights submitted to Central Server!`, "success");
            } else {
                logClient(`[ERROR] ${data.message}`, "warning");
            }
        } catch (err) {
            logClient(`[ERROR] ${err}`, "warning");
        } finally {
            trainOnDeviceBtn.disabled = false;
            trainOnDeviceBtn.innerHTML = `
                <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
                    <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
                </svg>
                <span>Train & Push DP Weights</span>`;
        }
    });

    // Event: Trigger Federated Round (Server Tab)
    triggerFedAvgBtn.addEventListener("click", async () => {
        triggerFedAvgBtn.disabled = true;
        triggerFedAvgBtn.textContent = "Executing FedAvg...";

        logServer(`\n============================================================`, "info");
        logServer(`[FEDAVG] Initiating 5 Communication Rounds across connected client nodes...`, "cyan");

        try {
            const res = await fetch("/api/server/aggregate", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ rounds: 5, epsilon: 1.0, use_dp: true })
            });
            const data = await res.json();

            if (data.status === "success") {
                logServer(`[CONVERGENCE] ${data.message}`, "success");
                logServer(`[PRIVACY] Single-release Laplace DP active with epsilon=${data.epsilon}`, "cyan");
                logServer(`[UPDATE] Global model weights updated and broadcasted.`, "success");
                await loadSystemStatus();
            } else {
                logServer(`[ERROR] ${data.message}`, "warning");
            }
        } catch (err) {
            logServer(`[ERROR] ${err}`, "warning");
        } finally {
            triggerFedAvgBtn.disabled = false;
            triggerFedAvgBtn.innerHTML = `
                <svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/>
                </svg>
                <span>Trigger Federated Round (FedAvg)</span>`;
        }
    });

    // Simulator Sliders & Logic
    const simHr = document.getElementById("sim-hr");
    const simBpSys = document.getElementById("sim-bp-sys");
    const simBpDia = document.getElementById("sim-bp-dia");
    const simSpo2 = document.getElementById("sim-spo2");
    const simSleep = document.getElementById("sim-sleep");
    const simPredictBtn = document.getElementById("sim-predict-btn");

    function updateSliderLabels() {
        document.getElementById("sim-hr-val").textContent = `${simHr.value} BPM`;
        document.getElementById("sim-bp-sys-val").textContent = `${simBpSys.value} mmHg`;
        document.getElementById("sim-bp-dia-val").textContent = `${simBpDia.value} mmHg`;
        document.getElementById("sim-spo2-val").textContent = `${simSpo2.value} %`;
        document.getElementById("sim-sleep-val").textContent = `${simSleep.value} hrs`;
    }

    [simHr, simBpSys, simBpDia, simSpo2, simSleep].forEach(el => {
        el.addEventListener("input", updateSliderLabels);
    });

    async function runSimulatorPrediction() {
        const vitals = {
            heart_rate: parseFloat(simHr.value),
            blood_pressure_systolic: parseFloat(simBpSys.value),
            blood_pressure_diastolic: parseFloat(simBpDia.value),
            blood_oxygen: parseFloat(simSpo2.value),
            sleep_quality: parseFloat(simSleep.value)
        };

        try {
            const res = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(vitals)
            });
            const data = await res.json();
            const pred = data.prediction;

            const badge = document.getElementById("sim-result-badge");
            badge.textContent = `${pred.condition.toUpperCase()} (${pred.badge})`;

            document.getElementById("sim-vri").textContent = pred.vital_risk_index;
            const maxProb = Math.max(...Object.values(pred.probabilities));
            document.getElementById("sim-conf").textContent = `${Math.round(maxProb * 100)}%`;

            // Alerts
            const alertsBox = document.getElementById("sim-alerts-box");
            alertsBox.innerHTML = "";
            if (pred.anomalies && pred.anomalies.length > 0) {
                pred.anomalies.forEach(a => {
                    const p = document.createElement("p");
                    p.style.color = "#fda4af";
                    p.style.marginBottom = "4px";
                    p.textContent = `⚠️ ${a}`;
                    alertsBox.appendChild(p);
                });
            } else {
                alertsBox.innerHTML = `<p class="sim-safe-note">✓ All vitals within safe physiological thresholds.</p>`;
            }

            // Probability bars
            const probsContainer = document.getElementById("sim-probs-container");
            probsContainer.innerHTML = "";
            for (const [cname, p] of Object.entries(pred.probabilities)) {
                const pct = Math.round(p * 100);
                const row = document.createElement("div");
                row.className = "sim-prob-row";
                row.innerHTML = `
                    <div class="sim-prob-label">
                        <span>${cname}</span>
                        <span>${pct}%</span>
                    </div>
                    <div class="sim-prob-bar">
                        <div class="sim-prob-fill" style="width: ${pct}%"></div>
                    </div>`;
                probsContainer.appendChild(row);
            }
        } catch (err) {
            console.error("Simulator prediction failed", err);
        }
    }

    simPredictBtn.addEventListener("click", runSimulatorPrediction);

    // Initial Load
    loadSystemStatus();
    loadUserList();
    runSimulatorPrediction();
});

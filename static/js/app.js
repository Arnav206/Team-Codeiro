/* ==========================================================================
   HealthPulse — Clean Client Logic
   Manages user profiles, Google Fit sync, and clean clinical graphs
   ========================================================================== */

document.addEventListener("DOMContentLoaded", () => {
    const userSelect = document.getElementById("user-select");
    const userAvatar = document.getElementById("nav-user-avatar");
    const refreshFitBtn = document.getElementById("refresh-fit-btn");
    const trainOnDeviceBtn = document.getElementById("train-ondevice-btn");
    const connectNewBtn = document.getElementById("connect-new-btn");
    const privacyToast = document.getElementById("privacy-toast");
    const toastCloseBtn = document.getElementById("toast-close-btn");

    let timelineChart = null;
    let riskProbChart = null;

    // Toast dismissal
    toastCloseBtn.addEventListener("click", () => {
        privacyToast.style.display = "none";
    });

    function showToast(title, desc) {
        document.getElementById("toast-title").textContent = title;
        document.getElementById("toast-desc").textContent = desc;
        privacyToast.style.display = "flex";
        setTimeout(() => {
            privacyToast.style.display = "none";
        }, 8000);
    }

    // User Avatar Initials Map
    function getInitials(name) {
        if (!name) return "U";
        if (name.toLowerCase().includes("user1") || name.toLowerCase().includes("anurag")) return "AK";
        if (name.toLowerCase().includes("user2")) return "U2";
        if (name.toLowerCase().includes("user3")) return "U3";
        return name.slice(0, 2).toUpperCase();
    }

    // Load available users
    async function loadUserDropdown() {
        try {
            const res = await fetch("/api/users");
            const data = await res.json();
            userSelect.innerHTML = "";

            data.users.forEach(u => {
                const opt = document.createElement("option");
                opt.value = u.user_id;
                const displayName = u.user_id === "user1" ? "Anurag Kushwaha (User 1)" :
                                   u.user_id === "user2" ? "Family Member (User 2)" :
                                   `Patient (${u.user_id})`;
                opt.textContent = `${displayName} ${u.authenticated ? '✓' : ''}`;
                userSelect.appendChild(opt);
            });

            if (data.users.length > 0) {
                userAvatar.textContent = getInitials(userSelect.value);
                loadPatientVitals(userSelect.value);
            }
        } catch (err) {
            console.error("Could not load users", err);
        }
    }

    // Load Vitals & Predictions for Selected User
    async function loadPatientVitals(userId) {
        userAvatar.textContent = getInitials(userId);
        try {
            const res = await fetch(`/api/user/${userId}/data`);
            const data = await res.json();

            const vitals = data.latest_vitals || {};
            const pred = data.prediction || {};

            // 1. Update Vitals Grid
            document.getElementById("val-hr").textContent = vitals.heart_rate ? Math.round(vitals.heart_rate) : "--";
            document.getElementById("val-bp").textContent = (vitals.blood_pressure_systolic && vitals.blood_pressure_diastolic) 
                ? `${Math.round(vitals.blood_pressure_systolic)}/${Math.round(vitals.blood_pressure_diastolic)}` : "--/--";
            document.getElementById("val-spo2").textContent = vitals.blood_oxygen ? Math.round(vitals.blood_oxygen) : "--";

            const steps = vitals.steps_count ? Math.round(vitals.steps_count) : 0;
            document.getElementById("val-steps").textContent = steps.toLocaleString();
            document.getElementById("step-fill-bar").style.width = `${Math.min(100, (steps / 10000) * 100)}%`;

            document.getElementById("val-sleep").textContent = vitals.sleep_quality ? (vitals.sleep_quality).toFixed(1) : "--";
            document.getElementById("val-calories").textContent = vitals.calories_burned ? Math.round(vitals.calories_burned) : "--";

            // 2. Health Score out of 100
            // Convert vital risk index (range ~3 to 15) to a clean human score (100 is best)
            const vri = pred.vital_risk_index !== undefined ? pred.vital_risk_index : 5.0;
            const healthScore = Math.max(30, Math.min(98, Math.round(100 - (vri - 4.5) * 6)));
            document.getElementById("health-score-val").textContent = healthScore;

            // 3. Status Pill & Advice
            const pill = document.getElementById("hero-status-pill");
            const condTitle = document.getElementById("hero-condition-text");
            const advice = document.getElementById("hero-advice-text");
            const cond = pred.condition || "Normal";

            pill.className = "status-pill";
            if (cond === "Normal") {
                pill.textContent = "SAFE & HEALTHY";
                condTitle.textContent = "Optimal Health Telemetry";
                advice.textContent = "Your resting heart rate, sleep recovery, and activity levels are in great balance today.";
            } else if (cond === "Mild Event") {
                pill.classList.add("mild");
                pill.textContent = "MILD ATTENTION";
                condTitle.textContent = "Mild Physiological Event";
                advice.textContent = "Slightly elevated heart rate or minor sleep deficit recorded. Ensure adequate rest & hydration.";
            } else if (cond === "Moderate Event") {
                pill.classList.add("moderate");
                pill.textContent = "MODERATE STRAIN";
                condTitle.textContent = "Moderate Vital Strain Detected";
                advice.textContent = "Noticeable vital deviation observed (blood pressure or SpO2 dip). Consider checking in with a physician.";
            } else {
                pill.classList.add("severe");
                pill.textContent = "HIGH ALERT";
                condTitle.textContent = "Critical Physiological Alert";
                advice.textContent = "Acute vital anomaly detected! Please seek clinical medical evaluation.";
            }

            // 4. Clinical Warnings (if any)
            const warnBox = document.getElementById("clinical-warnings-box");
            warnBox.innerHTML = "";
            if (pred.anomalies && pred.anomalies.length > 0) {
                warnBox.style.display = "flex";
                pred.anomalies.forEach(a => {
                    const span = document.createElement("span");
                    span.className = "warn-item";
                    span.textContent = `⚠️ ${a}`;
                    warnBox.appendChild(span);
                });
            } else {
                warnBox.style.display = "none";
            }

            // 5. Render Clean Graphs
            renderRiskProbChart(pred.probabilities || {});
            renderTimelineChart(data.timeline || []);

        } catch (err) {
            console.error("Vitals load failed", err);
        }
    }

    // Graph 1: Disease Risk Probability Breakdown
    function renderRiskProbChart(probs) {
        const ctx = document.getElementById("riskProbChart").getContext("2d");
        const labels = ["Normal", "Mild", "Moderate", "Severe"];
        const dataVals = [
            Math.round((probs["Normal"] || 0) * 100),
            Math.round((probs["Mild Event"] || 0) * 100),
            Math.round((probs["Moderate Event"] || 0) * 100),
            Math.round((probs["Severe Event"] || 0) * 100)
        ];

        if (riskProbChart) riskProbChart.destroy();

        riskProbChart = new Chart(ctx, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [{
                    data: dataVals,
                    backgroundColor: [
                        "#10b981", // Normal
                        "#3b82f6", // Mild
                        "#f59e0b", // Moderate
                        "#ef4444"  // Severe
                    ],
                    borderRadius: 8,
                    barThickness: 36
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => ` Probability: ${ctx.raw}%`
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { color: "#94a3b8", font: { family: 'Inter', size: 12, weight: '500' } }
                    },
                    y: {
                        max: 100,
                        grid: { color: "rgba(255, 255, 255, 0.05)" },
                        ticks: { color: "#64748b", callback: v => `${v}%` }
                    }
                }
            }
        });
    }

    // Graph 2: 14-Day Vitals Trend
    function renderTimelineChart(timeline) {
        const ctx = document.getElementById("timelineChart").getContext("2d");
        const labels = timeline.map((_, i) => `D${i + 1}`);
        const hrData = timeline.map(t => t.heart_rate);
        const stepsData = timeline.map(t => t.steps);

        if (timelineChart) timelineChart.destroy();

        timelineChart = new Chart(ctx, {
            type: "line",
            data: {
                labels: labels,
                datasets: [
                    {
                        label: "Heart Rate (BPM)",
                        data: hrData,
                        borderColor: "#ef4444",
                        backgroundColor: "rgba(239, 68, 68, 0.08)",
                        tension: 0.35,
                        fill: true,
                        yAxisID: "y-hr",
                        pointRadius: 2
                    },
                    {
                        label: "Steps",
                        data: stepsData,
                        borderColor: "#3b82f6",
                        backgroundColor: "transparent",
                        borderDash: [4, 4],
                        tension: 0.3,
                        yAxisID: "y-steps",
                        pointRadius: 0
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: { color: "#94a3b8", font: { family: 'Inter', size: 11 } }
                    }
                },
                scales: {
                    x: {
                        grid: { color: "rgba(255,255,255,0.03)" },
                        ticks: { color: "#64748b" }
                    },
                    "y-hr": {
                        type: "linear",
                        position: "left",
                        grid: { color: "rgba(255,255,255,0.04)" },
                        ticks: { color: "#ef4444" }
                    },
                    "y-steps": {
                        type: "linear",
                        position: "right",
                        grid: { display: false },
                        ticks: { color: "#3b82f6" }
                    }
                }
            }
        });
    }

    // Switch User Event
    userSelect.addEventListener("change", () => {
        loadPatientVitals(userSelect.value);
    });

    // Refresh / Sync Google Fit Data
    refreshFitBtn.addEventListener("click", async () => {
        const u = userSelect.value;
        refreshFitBtn.disabled = true;
        refreshFitBtn.textContent = "Syncing...";

        try {
            const res = await fetch(`/api/user/${u}/fetch`, { method: "POST" });
            const data = await res.json();
            if (data.status === "success") {
                showToast("Google Fit Synced!", `Retrieved 30 days of real telemetry records for ${u}.`);
                await loadPatientVitals(u);
            } else {
                alert(`Sync Notice: ${data.message}`);
            }
        } catch (err) {
            alert(`Sync Error: ${err}`);
        } finally {
            refreshFitBtn.disabled = false;
            refreshFitBtn.innerHTML = `
                <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67"/>
                </svg>
                <span>Sync Now</span>`;
        }
    });

    // Save & Protect Privately (On-device Local DP Training)
    trainOnDeviceBtn.addEventListener("click", async () => {
        const u = userSelect.value;
        trainOnDeviceBtn.disabled = true;
        trainOnDeviceBtn.textContent = "Protecting & Syncing...";

        try {
            const res = await fetch(`/api/user/${u}/train`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ epsilon: 1.0, epochs: 30 })
            });
            const data = await res.json();
            if (data.status === "success") {
                showToast(
                    "Data Protected & Synchronized!",
                    "Trained locally on your device with Differential Privacy (Laplace noise). Only private weights were sent to the central server. Raw health data never left your device."
                );
            } else {
                alert(`Error: ${data.message}`);
            }
        } catch (err) {
            alert(`Connection Error: ${err}`);
        } finally {
            trainOnDeviceBtn.disabled = false;
            trainOnDeviceBtn.innerHTML = `
                <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                </svg>
                <span>Save & Protect Privately</span>`;
        }
    });

    // Add New User / Connect Google Fit
    connectNewBtn.addEventListener("click", () => {
        const newUserName = prompt("Enter new user ID (e.g. user3, dad, mom):", "user3");
        if (newUserName) {
            alert(`To connect '${newUserName}', run: \n\npython main_pipeline.py --auth ${newUserName}\n\nin terminal to link their Google account.`);
        }
    });

    // Initial Load
    loadUserDropdown();
});

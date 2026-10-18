/**
 * HealthShield — Interactive Clinical 
 * Implements Chart.js visualizers with theme awareness (Light & Dark),
 * smooth micro-animations, and clinical benchmark guides.
 */

class HealthCharts {
    constructor() {
        this.timelineChart = null;
        this.riskChart = null;
        this.xaiChart = null;
    }

    isDark() {
        return document.documentElement.getAttribute("data-theme") === "dark";
    }

    getColors() {
        const dark = this.isDark();
        return {
            text: dark ? "#cbd5e1" : "#475569",
            title: dark ? "#f8fafc" : "#0f172a",
            grid: dark ? "rgba(255, 255, 255, 0.06)" : "#f1f5f9",
            tooltipBg: dark ? "#1e293b" : "#0f172a",
            hrLine: "#f43f5e",
            hrFill: dark ? "rgba(244, 63, 94, 0.15)" : "rgba(244, 63, 94, 0.08)",
            stepsLine: "#0284c7",
            stepsFill: dark ? "rgba(2, 132, 199, 0.18)" : "rgba(2, 132, 199, 0.1)",
            normal: "#10b981",
            mild: "#0ea5e9",
            moderate: "#f59e0b",
            severe: "#ef4444"
        };
    }

    /**
     * 14-Day Vitals Trend (Dual Axis: Heart Rate & Steps, with SpO2 and Sleep)
     */
    renderTimeline(timelineData) {
        const canvas = document.getElementById("timelineChart");
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        const c = this.getColors();

        const labels = timelineData.map(d => d.date || "Day");
        const hrData = timelineData.map(d => d.heart_rate || 70);
        const stepsData = timelineData.map(d => d.steps_count || d.steps || 6000);
        const sleepData = timelineData.map(d => d.sleep_quality || d.sleep || 7.0);
        const spo2Data = timelineData.map(d => d.blood_oxygen || d.spo2 || 98.0);

        if (this.timelineChart) {
            this.timelineChart.destroy();
        }

        this.timelineChart = new Chart(ctx, {
            type: "line",
            data: {
                labels: labels,
                datasets: [
                    {
                        label: "Resting Heart Rate (BPM)",
                        data: hrData,
                        borderColor: c.hrLine,
                        backgroundColor: c.hrFill,
                        borderWidth: 2.5,
                        fill: true,
                        tension: 0.38,
                        yAxisID: "y-hr",
                        pointRadius: 3,
                        pointHoverRadius: 6,
                        pointBackgroundColor: c.hrLine
                    },
                    {
                        label: "Daily Steps",
                        data: stepsData,
                        borderColor: c.stepsLine,
                        backgroundColor: c.stepsFill,
                        borderWidth: 2,
                        tension: 0.3,
                        fill: false,
                        borderDash: [5, 4],
                        yAxisID: "y-steps",
                        pointRadius: 2,
                        pointHoverRadius: 5,
                        pointBackgroundColor: c.stepsLine
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: "index",
                    intersect: false
                },
                plugins: {
                    legend: {
                        position: "top",
                        align: "end",
                        labels: {
                            color: c.text,
                            boxWidth: 14,
                            boxHeight: 10,
                            borderRadius: 4,
                            usePointStyle: true,
                            font: { family: "'Plus Jakarta Sans', sans-serif", size: 12, weight: "600" }
                        }
                    },
                    tooltip: {
                        backgroundColor: c.tooltipBg,
                        titleColor: "#ffffff",
                        bodyColor: "#f1f5f9",
                        borderColor: "rgba(255,255,255,0.1)",
                        borderWidth: 1,
                        padding: 12,
                        boxPadding: 6,
                        callbacks: {
                            afterBody: (tooltipItems) => {
                                const idx = tooltipItems[0].dataIndex;
                                return [
                                    `Sleep: ${sleepData[idx]} hrs`,
                                    `Oxygen SpO2: ${spo2Data[idx]}%`
                                ];
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: c.grid, drawBorder: false },
                        ticks: { color: c.text, font: { family: "'Inter', sans-serif", size: 11 } }
                    },
                    "y-hr": {
                        type: "linear",
                        position: "left",
                        min: 45,
                        max: 130,
                        grid: { color: c.grid, drawBorder: false },
                        ticks: {
                            color: c.hrLine,
                            callback: v => `${v} bpm`,
                            font: { size: 11, weight: "500" }
                        }
                    },
                    "y-steps": {
                        type: "linear",
                        position: "right",
                        min: 0,
                        max: 16000,
                        grid: { display: false },
                        ticks: {
                            color: c.stepsLine,
                            callback: v => `${(v / 1000).toFixed(0)}k`,
                            font: { size: 11, weight: "500" }
                        }
                    }
                }
            }
        });
    }

    /**
     * Multi-Class Risk Probability Breakdown Bar
     */
    renderRiskBreakdown(probs) {
        const canvas = document.getElementById("riskProbChart");
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        const c = this.getColors();

        const labels = ["Normal", "Mild Attention", "Moderate Strain", "Severe Alert"];
        const values = [
            Math.round((probs["Normal"] || 0) * 100),
            Math.round((probs["Mild Event"] || 0) * 100),
            Math.round((probs["Moderate Event"] || 0) * 100),
            Math.round((probs["Severe Event"] || 0) * 100)
        ];

        if (this.riskChart) {
            this.riskChart.destroy();
        }

        this.riskChart = new Chart(ctx, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [{
                    data: values,
                    backgroundColor: [c.normal, c.mild, c.moderate, c.severe],
                    borderRadius: 8,
                    barThickness: 32,
                    borderSkipped: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: c.tooltipBg,
                        padding: 10,
                        callbacks: {
                            label: (ctx) => ` Confidence: ${ctx.raw}%`
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { color: c.text, font: { family: "'Plus Jakarta Sans', sans-serif", size: 11, weight: "600" } }
                    },
                    y: {
                        max: 100,
                        grid: { color: c.grid, drawBorder: false },
                        ticks: { color: c.text, callback: v => `${v}%` }
                    }
                }
            }
        });
    }

    /**
     * Explainable AI (XAI) Shapley Attribution Waterfall / Bar Chart
     */
    renderXaiAttribution(xaiData) {
        const canvas = document.getElementById("xaiChart");
        if (!canvas) return;
        const ctx = canvas.getContext("2d");
        const c = this.getColors();

        const topFeatures = xaiData.top_features || xaiData.all_features || [];
        const labels = topFeatures.map(f => f.name);
        const data = topFeatures.map(f => f.is_risk ? f.percentage : -f.percentage);
        const bgColors = topFeatures.map(f => f.is_risk ? c.severe : c.normal);

        if (this.xaiChart) {
            this.xaiChart.destroy();
        }

        this.xaiChart = new Chart(ctx, {
            type: "bar",
            data: {
                labels: labels,
                datasets: [{
                    data: data,
                    backgroundColor: bgColors,
                    borderRadius: 6,
                    barThickness: 20
                }]
            },
            options: {
                indexAxis: "y",
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: c.tooltipBg,
                        padding: 10,
                        callbacks: {
                            label: (ctx) => {
                                const val = ctx.raw;
                                return val > 0
                                    ? ` +${val}% Increased Cardiovascular Risk Burden`
                                    : ` ${val}% Protective Clinical Factor`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: c.grid, drawBorder: false },
                        ticks: {
                            color: c.text,
                            callback: v => `${v > 0 ? "+" : ""}${v}%`
                        }
                    },
                    y: {
                        grid: { display: false },
                        ticks: {
                            color: c.title,
                            font: { family: "'Plus Jakarta Sans', sans-serif", size: 11, weight: "500" }
                        }
                    }
                }
            }
        });
    }

    /**
     * Re-renders all active charts upon theme change
     */
    refreshTheme(timelineData, probs, xaiData) {
        if (timelineData) this.renderTimeline(timelineData);
        if (probs) this.renderRiskBreakdown(probs);
        if (xaiData) this.renderXaiAttribution(xaiData);
    }
}

window.HealthCharts = new HealthCharts();

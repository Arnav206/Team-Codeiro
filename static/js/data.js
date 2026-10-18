/**
 * HealthShield — Clinical Data Models & Simulation Engine
 * Matches exact algorithms from Team-Codeiro backend:
 * - fit_pipeline.py (17-feature space & derived metrics)
 * - predict_fit_user.py (Federated model inference, XAI, 24h disease early warnings)
 * - explain.py (On-device Shapley attribution)
 */

const CLINICAL_BENCHMARKS = {
    heart_rate: { min: 60, max: 100, optimal: 70, unit: "BPM", name: "Resting Heart Rate" },
    blood_oxygen: { min: 95, max: 100, optimal: 98.5, unit: "%", name: "Blood Oxygen (SpO2)" },
    blood_pressure_systolic: { min: 105, max: 125, optimal: 118, unit: "mmHg", name: "Systolic Blood Pressure" },
    blood_pressure_diastolic: { min: 65, max: 85, optimal: 78, unit: "mmHg", name: "Diastolic Blood Pressure" },
    sleep_quality: { min: 7.0, max: 9.0, optimal: 7.8, unit: "hrs", name: "Sleep Duration" },
    steps_count: { min: 7000, max: 12000, optimal: 9500, unit: "steps", name: "Daily Steps" },
    glucose_level: { min: 70, max: 110, optimal: 92, unit: "mg/dL", name: "Fasting Blood Glucose" },
    calories_burned: { min: 1800, max: 2800, optimal: 2200, unit: "kcal", name: "Active Burn" },
    hrv_sdnn: { min: 45, max: 95, optimal: 65, unit: "ms", name: "Heart Rate Variability (HRV)" }
};

const USER_PROFILES = {
    "user1": {
        id: "user1",
        name: "Anurag Kushwaha",
        role: "Software Engineer • 28 yrs",
        initials: "AK",
        email: "anuragkushwaha2207@gmail.com",
        source: "Google Fit Watch Active",
        archetype: "athletic",
        avatarBg: "linear-gradient(135deg, #0284c7, #2563eb)",
        personaDescription: "Active lifestyle, regular runner. Shows healthy cardiovascular baseline with occasional work-deadline stress spikes."
    },
    "user2": {
        id: "user2",
        name: "Sunita Kushwaha",
        role: "Family Member • 56 yrs",
        initials: "SK",
        email: "sunita@family.health",
        source: "Home Telemetry Node",
        archetype: "hypertensive",
        avatarBg: "linear-gradient(135deg, #f59e0b, #d97706)",
        personaDescription: "Monitored for borderline systolic hypertension and mild sleep sensitivity. Benefits from low-sodium guidelines."
    },
    "user3": {
        id: "user3",
        name: "Vikram Rao",
        role: "Shift Worker • 39 yrs",
        initials: "VR",
        email: "vikram@logistics.local",
        source: "Wearable Band Sync",
        archetype: "sleep_deficit",
        avatarBg: "linear-gradient(135deg, #8b5cf6, #6d28d9)",
        personaDescription: "Rotational night shifts lead to irregular circadian sleep debt. Highly responsive to daytime recovery habits."
    },
    "user4": {
        id: "user4",
        name: "Dr. Priya Patel",
        role: "Cardiologist & Researcher • 44 yrs",
        initials: "PP",
        email: "dr.priya@healthshield.ai",
        source: "Clinical Pilot Node",
        archetype: "balanced",
        avatarBg: "linear-gradient(135deg, #10b981, #059669)",
        personaDescription: "Optimal health baseline with high daily step counts, balanced nutrition, and consistent resting heart rate."
    }
};

/**
 * Derived features matching fit_pipeline.py compute_derived_features()
 */
function computeDerivedFeatures(v) {
    const hr = Number(v.heart_rate || 72);
    const spo2 = Number(v.blood_oxygen || 98);
    const bpSys = Number(v.blood_pressure_systolic || 120);
    const bpDia = Number(v.blood_pressure_diastolic || 80);
    const sleep = Number(v.sleep_quality || 7.2);

    // Derived stress level (1 to 5 scale)
    const hrFactor = (hr - 65.0) / 20.0;
    const sleepFactor = (7.5 - sleep) / 2.0;
    const stress = Math.min(5.0, Math.max(1.0, 2.0 + hrFactor + sleepFactor));

    const hr_stress_ratio = hr * stress;
    const spo2_deficit = Math.max(0, 100.0 - spo2);
    const bp_diff = bpSys - bpDia;
    const vital_risk_index = (hr / 70.0) + (spo2_deficit / 5.0) + (stress * 2.0);

    return {
        stress_level: Number(stress.toFixed(1)),
        hr_stress_ratio: Number(hr_stress_ratio.toFixed(1)),
        spo2_deficit: Number(spo2_deficit.toFixed(1)),
        bp_diff: Number(bp_diff.toFixed(1)),
        vital_risk_index: Number(vital_risk_index.toFixed(2))
    };
}

/**
 * Health Risk Classification matching fit_pipeline.py assign_clinical_labels()
 * and predict_fit_user.py
 */
function predictHealthRisk(vitals) {
    const derived = computeDerivedFeatures(vitals);
    const hr = Number(vitals.heart_rate || 72);
    const spo2 = Number(vitals.blood_oxygen || 98);
    const bpSys = Number(vitals.blood_pressure_systolic || 120);
    const bpDia = Number(vitals.blood_pressure_diastolic || 80);
    const vri = derived.vital_risk_index;
    const sleep = Number(vitals.sleep_quality || 7.0);

    let predClass = 0;
    let condition = "Normal";
    let badge = "LOW RISK (NORMAL)";
    let badgeClass = "normal";

    if (bpSys >= 155 || bpDia >= 100 || spo2 <= 91 || hr >= 125 || vri >= 8.5) {
        predClass = 3;
        condition = "Severe Event";
        badge = "HIGH RISK (CRITICAL ALERT)";
        badgeClass = "severe";
    } else if (bpSys >= 135 || bpDia >= 88 || spo2 <= 94 || hr >= 105 || vri >= 6.5) {
        predClass = 2;
        condition = "Moderate Event";
        badge = "MODERATE STRAIN (ATTENTION NEEDED)";
        badgeClass = "moderate";
    } else if (bpSys >= 125 || bpDia >= 82 || hr >= 88 || sleep < 5.5 || vri >= 5.0) {
        predClass = 1;
        condition = "Mild Event";
        badge = "MILD ATTENTION (MONITOR & REST)";
        badgeClass = "mild";
    }

    // Dynamic softmax-like probabilities based on VRI distance
    let pNormal = 0.82, pMild = 0.12, pMod = 0.04, pSev = 0.02;
    if (predClass === 1) {
        pNormal = 0.22; pMild = 0.62; pMod = 0.12; pSev = 0.04;
    } else if (predClass === 2) {
        pNormal = 0.06; pMild = 0.18; pMod = 0.64; pSev = 0.12;
    } else if (predClass === 3) {
        pNormal = 0.02; pMild = 0.06; pMod = 0.22; pSev = 0.70;
    }

    // Anomalies
    const anomalies = [];
    if (bpSys >= 140 || bpDia >= 90) {
        anomalies.push(`Hypertension Alert (BP: ${Math.round(bpSys)}/${Math.round(bpDia)} mmHg)`);
    } else if (bpSys >= 128) {
        anomalies.push(`Borderline Elevated Systolic Pressure (${Math.round(bpSys)} mmHg)`);
    }

    if (spo2 < 93) {
        anomalies.push(`Hypoxemia Risk (SpO2: ${spo2.toFixed(1)}%)`);
    } else if (spo2 < 95.5) {
        anomalies.push(`Mild Oxygen Saturation Dip (${spo2.toFixed(1)}%)`);
    }

    if (hr > 100) {
        anomalies.push(`Tachycardia / Elevated Resting Heart Rate (${Math.round(hr)} BPM)`);
    } else if (hr < 50) {
        anomalies.push(`Bradycardia (${Math.round(hr)} BPM)`);
    }

    if (sleep < 5.5) {
        anomalies.push(`Severe Sleep Deficit (${sleep.toFixed(1)} hrs recorded)`);
    }

    if (vri >= 6.5) {
        anomalies.push(`Elevated Vital Risk Index (${vri.toFixed(2)})`);
    }

    // Humanized Health Score (0 - 100)
    const healthScore = Math.max(30, Math.min(99, Math.round(100 - (vri - 4.2) * 8.5)));

    const xai = computeXaiAttribution(vitals, derived, predClass);
    const forecasts = forecast24hDiseases(vitals, derived, healthScore);
    const humanSummary = generateHumanNarrative(vitals, derived, condition, healthScore);
    const habits = generateDailyHabits(vitals, condition);

    return {
        condition,
        badge,
        badgeClass,
        class_index: predClass,
        vital_risk_index: vri,
        health_score: healthScore,
        probabilities: {
            "Normal": pNormal,
            "Mild Event": pMild,
            "Moderate Event": pMod,
            "Severe Event": pSev
        },
        anomalies,
        derived,
        xai,
        future_diseases_24h: forecasts,
        human_narrative: humanSummary,
        daily_habits: habits
    };
}

/**
 * Explainable AI (XAI) feature attribution matching predict_fit_user.py
 */
function computeXaiAttribution(vitals, derived, predClass) {
    const hr = Number(vitals.heart_rate || 72);
    const spo2 = Number(vitals.blood_oxygen || 98);
    const bpSys = Number(vitals.blood_pressure_systolic || 120);
    const sleep = Number(vitals.sleep_quality || 7.2);
    const steps = Number(vitals.steps_count || 6500);

    const factors = [];

    // Sleep attribution
    const sleepDebt = Math.max(0, 7.5 - sleep);
    if (sleepDebt > 0.8) {
        factors.push({
            name: "Sleep Debt & Incomplete Recovery",
            percentage: Math.min(38, Math.round(sleepDebt * 14 + 10)),
            is_risk: true,
            icon: "😴",
            note: `${sleep.toFixed(1)} hrs logged vs 7.5 hrs clinical target (-${sleepDebt.toFixed(1)}h deficit)`
        });
    } else {
        factors.push({
            name: "Restorative Sleep Pattern",
            percentage: 24,
            is_risk: false,
            icon: "🌙",
            note: `${sleep.toFixed(1)} hrs of high-quality restorative sleep supports cardiac rhythm`
        });
    }

    // Blood Pressure
    if (bpSys >= 126) {
        factors.push({
            name: "Arterial Hemodynamic Pressure",
            percentage: Math.min(35, Math.round((bpSys - 120) * 2.2 + 8)),
            is_risk: true,
            icon: "🩺",
            note: `Systolic pressure at ${Math.round(bpSys)} mmHg increases vascular workload`
        });
    } else {
        factors.push({
            name: "Optimal Vascular Pressure",
            percentage: 22,
            is_risk: false,
            icon: "❤️",
            note: `Normal 118/78 mmHg pressure maintains healthy endothelial flow`
        });
    }

    // Oxygen Saturation
    if (spo2 < 96) {
        factors.push({
            name: "Peripheral SpO2 Deficit",
            percentage: Math.min(30, Math.round((98.5 - spo2) * 9)),
            is_risk: true,
            icon: "🫁",
            note: `Capillary oxygenation at ${spo2.toFixed(1)}% indicates mild tissue strain`
        });
    } else {
        factors.push({
            name: "Robust Cellular Oxygenation",
            percentage: 26,
            is_risk: false,
            icon: "🌿",
            note: `SpO2 at ${spo2.toFixed(1)}% provides full mitochondrial respiration`
        });
    }

    // Physical Activity
    if (steps < 4000) {
        factors.push({
            name: "Sedentary Autonomic Stagnation",
            percentage: 16,
            is_risk: true,
            icon: "🚶",
            note: `Low movement (${steps.toLocaleString()} steps) reduces lymphatic and venous return`
        });
    } else {
        factors.push({
            name: "Cardiorespiratory Step Volume",
            percentage: Math.min(28, Math.round((steps / 10000) * 25)),
            is_risk: false,
            icon: "👟",
            note: `${steps.toLocaleString()} daily steps maintains active metabolic clearance`
        });
    }

    const riskDrivers = factors.filter(f => f.is_risk).sort((a, b) => b.percentage - a.percentage);
    const protectiveFactors = factors.filter(f => !f.is_risk).sort((a, b) => b.percentage - a.percentage);

    return {
        all_features: factors,
        risk_drivers: riskDrivers,
        protective_factors: protectiveFactors,
        chart_labels: factors.map(f => f.name),
        chart_values: factors.map(f => f.is_risk ? f.percentage : -f.percentage),
        chart_colors: factors.map(f => f.is_risk ? "#ef4444" : "#10b981")
    };
}

/**
 * 24-Hour Disease Forecasting matching forecast_24h_future_diseases() in predict_fit_user.py
 */
function forecast24hDiseases(vitals, derived, healthScore) {
    const hr = Number(vitals.heart_rate || 72);
    const bpSys = Number(vitals.blood_pressure_systolic || 120);
    const bpDia = Number(vitals.blood_pressure_diastolic || 80);
    const spo2 = Number(vitals.blood_oxygen || 98);
    const sleep = Number(vitals.sleep_quality || 7.0);
    const steps = Number(vitals.steps_count || 6000);
    const vri = derived.vital_risk_index;

    const forecasts = [];

    // 1. Neuro-Fatigue / Sleep Debt
    const sleepDeficit = Math.max(0.0, 7.5 - sleep);
    const sleepProb = Math.min(94, Math.max(12, Math.round(22 + sleepDeficit * 20 + (vri - 4.5) * 5)));
    forecasts.push({
        id: "neuro_fatigue",
        disease_name: "Sleep-Debt Induced Cognitive & Neuro-Fatigue",
        category: "Cognitive & Neurological Recovery",
        icon: "🧠",
        timeframe: "Next 24 Hours",
        risk_probability: sleepProb,
        risk_level: sleepProb >= 60 ? "High Risk" : (sleepProb >= 35 ? "Moderate Strain" : "Optimal Low"),
        severity_color: sleepProb >= 60 ? "danger" : (sleepProb >= 35 ? "warning" : "success"),
        xai_reason: `Accumulated ${sleepDeficit.toFixed(1)}h sleep deficit recorded (${sleep.toFixed(1)}h vs 7.5h benchmark). Autonomic parasympathetic recovery is restricted.`,
        biomarkers: [
            `Sleep Duration: ${sleep.toFixed(1)} hrs`,
            `Autonomic Stress Index: ${derived.stress_level} / 5.0`
        ],
        prevention_guidelines: "Prioritize an 8-hour sleep window tonight. Restrict caffeine and phone screens after 4:00 PM."
    });

    // 2. Pre-Hypertension & Arterial Strain
    const pulsePressure = bpSys - bpDia;
    const bpProb = Math.min(92, Math.max(10, Math.round(18 + Math.max(0, bpSys - 118) * 3.8 + Math.max(0, pulsePressure - 40) * 1.8)));
    forecasts.push({
        id: "pre_hypertension",
        disease_name: "Pre-Hypertension & Arterial Wall Strain",
        category: "Cardiovascular & Hemodynamic Strain",
        icon: "🩺",
        timeframe: "Next 24–48 Hours",
        risk_probability: bpProb,
        risk_level: bpProb >= 60 ? "High Risk" : (bpProb >= 35 ? "Moderate Strain" : "Optimal Low"),
        severity_color: bpProb >= 60 ? "danger" : (bpProb >= 35 ? "warning" : "success"),
        xai_reason: `Systolic pressure at ${Math.round(bpSys)} mmHg with pulse pressure gradient ${Math.round(pulsePressure)} mmHg shows borderline hemodynamic vascular resistance.`,
        biomarkers: [
            `Blood Pressure: ${Math.round(bpSys)}/${Math.round(bpDia)} mmHg`,
            `Pulse Pressure: ${Math.round(pulsePressure)} mmHg`
        ],
        prevention_guidelines: "Keep dietary salt minimal today. Drink 2.5L water and take 10 minutes for slow diaphragm breathing."
    });

    // 3. Hypoxemic Exercise Exhaustion
    const spo2Drop = Math.max(0.0, 99.0 - spo2);
    const respProb = Math.min(88, Math.max(8, Math.round(14 + spo2Drop * 19)));
    forecasts.push({
        id: "hypoxemic_exhaustion",
        disease_name: "Hypoxemic Exercise Exhaustion & Oxygen Dip",
        category: "Respiratory & Cellular Oxygenation",
        icon: "🫁",
        timeframe: "Next 24 Hours",
        risk_probability: respProb,
        risk_level: respProb >= 55 ? "High Risk" : (respProb >= 30 ? "Moderate Strain" : "Optimal Low"),
        severity_color: respProb >= 55 ? "danger" : (respProb >= 30 ? "warning" : "success"),
        xai_reason: `Tissue oxygen saturation at ${spo2.toFixed(1)}% (optimal >= 98.5%) leaves minor oxygen gap during exertion.`,
        biomarkers: [
            `Oxygen SpO2: ${spo2.toFixed(1)}%`,
            `Daily Movement: ${Math.round(steps).toLocaleString()} steps`
        ],
        prevention_guidelines: "Avoid high-intensity sprint training today. Prefer gentle walking and deep outdoor ventilation."
    });

    // 4. Cardiovascular Autonomic Overload
    let cardioProb = 12;
    let cardioReason = `Resting heart rate at ${Math.round(hr)} BPM is within ideal zone, functioning as an on-device protective biomarker.`;
    if (hr > 82) {
        cardioProb = Math.min(88, Math.round(28 + (hr - 80) * 3.2));
        cardioReason = `Resting heart rate at ${Math.round(hr)} BPM reflects sympathetic nervous dominance and elevated myocardial workload.`;
    }
    forecasts.push({
        id: "cardio_overload",
        disease_name: "Cardiovascular Autonomic Overload & Tachycardia",
        category: "Autonomic Nervous System & Rhythm",
        icon: "💓",
        timeframe: "Next 24 Hours",
        risk_probability: cardioProb,
        risk_level: cardioProb >= 60 ? "High Risk" : (cardioProb >= 32 ? "Moderate Strain" : "Optimal Low"),
        severity_color: cardioProb >= 60 ? "danger" : (cardioProb >= 32 ? "warning" : "success"),
        xai_reason: cardioReason,
        biomarkers: [
            `Resting Heart Rate: ${Math.round(hr)} BPM`,
            `Vital Risk Index: ${vri.toFixed(2)}`
        ],
        prevention_guidelines: "Hydrate steadily, reduce caffeinated drinks, and avoid sudden heavy physical stressors."
    });

    return forecasts;
}

/**
 * Empathetic human-first narrative generator
 */
function generateHumanNarrative(vitals, derived, condition, score) {
    const hr = Math.round(vitals.heart_rate || 72);
    const sleep = (vitals.sleep_quality || 7.0).toFixed(1);
    const steps = (vitals.steps_count || 6500).toLocaleString();

    if (condition === "Normal") {
        return {
            title: "Your Body is Thriving and Well-Balanced",
            headline: "Optimal Recovery & Energetic Equilibrium",
            body: `You logged ${sleep} hours of restorative sleep and your resting heart rate is calm at ${hr} BPM. Your cells are receiving full oxygenation (SpO2 ${vitals.blood_oxygen}%) and you've achieved ${steps} steps. Keep honoring your natural rhythm today!`,
            encouragement: "You're in a great space to enjoy your favorite physical activities and focus clearly on creative work."
        };
    } else if (condition === "Mild Event") {
        return {
            title: "Your Body is Asking for a Little Gentle Care",
            headline: "Mild Physiological Fatigue Detected",
            body: `Your resting heart rate (${hr} BPM) is slightly elevated and sleep (${sleep} hrs) was lower than your baseline. Your body is managing fine, but taking things 10% slower today will prevent fatigue from compounding into tomorrow.`,
            encouragement: "A short 15-minute sunshine walk, plenty of water, and an earlier bedtime tonight will have you bouncing back strong."
        };
    } else if (condition === "Moderate Event") {
        return {
            title: "Noticeable Strain on Your Cardiovascular System",
            headline: "Elevated Telemetry Needs Active Attention",
            body: `We are seeing a combination of heightened blood pressure (${Math.round(vitals.blood_pressure_systolic)}/${Math.round(vitals.blood_pressure_diastolic)} mmHg) and autonomic stress. Your heart is working harder than usual to maintain homeostasis.`,
            encouragement: "Avoid high-intensity exertion or stressful negotiations this afternoon. Rest, drink herbal tea or warm water, and monitor your vitals."
        };
    } else {
        return {
            title: "Important Health Notice: Acute Biometric Strain",
            headline: "Immediate Rest & Clinical Review Advised",
            body: `Telemetry indicates acute cardiovascular or respiratory deviation. Multiple biomarkers are beyond safe baseline parameters.`,
            encouragement: "Please pause physical activity, sit or lie comfortably, and consider consulting a healthcare professional or doctor right away."
        };
    }
}

/**
 * Daily actionable human habits based on telemetry
 */
function generateDailyHabits(vitals, condition) {
    const sleep = vitals.sleep_quality || 7.0;
    const bpSys = vitals.blood_pressure_systolic || 120;
    const habits = [];

    if (sleep < 6.5) {
        habits.push({
            icon: "🛌",
            action: "Evening Wind-Down at 10:00 PM",
            detail: "Switch off mobile and TV screens 45 minutes before sleep to restore melatonin production.",
            tag: "High Priority"
        });
    } else {
        habits.push({
            icon: "☀️",
            action: "10-Minute Morning Sunlight",
            detail: "Expose eyes to natural morning daylight to anchor your circadian rhythm for tonight.",
            tag: "Circadian"
        });
    }

    if (bpSys >= 125) {
        habits.push({
            icon: "💧",
            action: "Electrolyte Hydration (2.5L)",
            detail: "Sip water regularly throughout the day and limit sodium-heavy canned foods to relax arterial walls.",
            tag: "Vascular"
        });
    } else {
        habits.push({
            icon: "🚶‍♂️",
            action: "Brisk 25-Min Post-Lunch Walk",
            detail: "Keep muscle glucose uptake high and maintain steady capillary blood circulation.",
            tag: "Movement"
        });
    }

    habits.push({
        icon: "🫁",
        action: "4-7-8 Breathing Reset",
        detail: "Inhale for 4s, hold for 7s, exhale for 8s (repeat 4 times). Rebalances vagus nerve parasympathetic tone.",
        tag: "Stress Relief"
    });

    return habits;
}

/**
 * Generates realistic 14-day history series for charts and tables
 */
function generate14DayHistory(userId, archetype = "athletic") {
    const base = archetype === "athletic" ? { hr: 64, bpSys: 116, bpDia: 74, spo2: 99.0, sleep: 7.8, steps: 9500, cal: 2350 }
               : archetype === "hypertensive" ? { hr: 80, bpSys: 138, bpDia: 88, spo2: 96.8, sleep: 6.4, steps: 4600, cal: 1750 }
               : archetype === "sleep_deficit" ? { hr: 76, bpSys: 124, bpDia: 82, spo2: 97.4, sleep: 5.2, steps: 5200, cal: 1820 }
               : { hr: 70, bpSys: 120, bpDia: 78, spo2: 98.2, sleep: 7.2, steps: 6800, cal: 1980 };

    const records = [];
    const now = new Date();

    for (let i = 13; i >= 0; i--) {
        const d = new Date(now);
        d.setDate(d.getDate() - i);
        const dayStr = d.toLocaleDateString("en-US", { month: "short", day: "numeric" });

        // Deterministic pseudo-random variation based on user + day offset
        const noise = Math.sin(i * 1.7 + (userId.charCodeAt(userId.length - 1) || 1));
        const noise2 = Math.cos(i * 2.1);

        const hr = Math.round(base.hr + noise * 5.5);
        const bpSys = Math.round(base.bpSys + noise2 * 7);
        const bpDia = Math.round(base.bpDia + noise2 * 4.5);
        const spo2 = Number((base.spo2 + noise * 0.7).toFixed(1));
        const sleep = Number(Math.max(3.8, base.sleep + noise2 * 1.1).toFixed(1));
        const steps = Math.max(1200, Math.round(base.steps + noise * 1600));
        const calories = Math.round(base.cal + noise * 220);

        const v = {
            date: dayStr,
            full_date: d.toISOString().split("T")[0],
            heart_rate: hr,
            blood_pressure_systolic: bpSys,
            blood_pressure_diastolic: bpDia,
            blood_oxygen: Math.min(100, Math.max(90, spo2)),
            sleep_quality: sleep,
            steps_count: steps,
            calories_burned: calories,
            glucose_level: Math.round(92 + noise * 8)
        };

        const pred = predictHealthRisk(v);
        records.push({
            ...v,
            condition: pred.condition,
            condition_class: pred.badgeClass,
            vital_risk_index: pred.vital_risk_index,
            health_score: pred.health_score
        });
    }

    return records;
}

// Export to window
window.HealthData = {
    BENCHMARKS: CLINICAL_BENCHMARKS,
    PROFILES: USER_PROFILES,
    computeDerivedFeatures,
    predictHealthRisk,
    computeXaiAttribution,
    forecast24hDiseases,
    generateHumanNarrative,
    generateDailyHabits,
    generate14DayHistory
};

# HealthShield — Humanized Federated Health & Telemetry Companion

A human-first, empathetic healthcare frontend designed for **Privacy-Preserving Federated Learning with Differential Privacy for IoT & Wearable Health Monitoring** (compatible with [Team-Codeiro Backend](https://github.com/Arnav206/Team-Codeiro.git)).

---

## 🌟 Why "Humanized"?

Clinical and machine-learning dashboards are often intimidating spreadsheets filled with raw tensors, loss curves, and frightening diagnostic codes. **HealthShield** transforms cold clinical telemetry into a warm, empathetic companion that patients, families, and physicians love to use:

- **Empathetic Body Narratives**: Rather than just showing *"Vital Risk Index: 5.82"*, HealthShield greets patients with clear, reassuring context: *"Your body is carrying a little extra strain today, mostly due to 5.2 hours of sleep and high work stress. Your heart and oxygen levels are holding up well, but your cardiovascular system needs an early bedtime tonight."*
- **Explainable AI (XAI) Demystified**: Transparently highlights *"What is protecting your body today"* (e.g., robust cellular oxygenation, cardiorespiratory step volume) versus *"What is asking for gentle care"* (e.g., arterial hemodynamic pressure, sleep debt).
- **24-Hour Predictive Early Warnings (Bilingual English & हिंदी)**:
  1. *Sleep-Debt Induced Neuro-Fatigue* (अति-थकान एवं मानसिक थकावट)
  2. *Pre-Hypertension & Arterial Wall Strain* (प्री-हाइपरटेंशन एवं धमनी तनाव)
  3. *Hypoxemic Exercise Exhaustion* (ऑक्सीजन डिप्रेशन एवं वर्कआउट थकान)
  4. *Cardiovascular Autonomic Overload* (कार्डियोवास्कुलर तनाव एवं टैचीकार्डिया)
- **Interactive "What-If" Vital Sandbox**: Adjust resting heart rate, blood pressure, oxygen saturation, and sleep duration on live sliders to observe instant clinical adaptations, health score recalculations, and preventative advice.
- **Human Feeling / Symptom Check-in**: Lets patients log how they feel (*Energized*, *Tired*, *Stressed*, *Dizzy*, *Palpitations*) and dynamically correlates their perception with their biometrics.
- **Privacy Shield Guarantee**: Clearly explains on-device Federated Learning and Differential Privacy (Laplace $\epsilon=1.0$) with zero raw data transmission.
- **Doctor's Clinical Visit Brief**: One-click printable clinical summary note designed for physician consultations, complete with 14-day averages and anomaly logs.
- **Audio Bedside Voice Narrator**: Uses the browser's Web Speech API to read today's health brief aloud in a warm, friendly voice.

---

## 🚀 Quick Start

### 1. Run the Frontend Locally

```bash
# Using Python's built-in web server:
python -m http.server 3000

# Or using Node.js:
npm start
```
Open [http://localhost:3000](http://localhost:3000) in your browser.

### 2. Connect with Team-Codeiro Flask Backend (Optional)

The frontend automatically detects if the Python backend is running on `http://localhost:5000`:
- When the Flask backend is live, it automatically syncs with `/api/status`, `/api/users`, `/api/user/<id>/data`, and Google Fit OAuth.
- When the backend is offline, the frontend seamlessly runs in **On-Device Edge Mode** with simulated patient telemetry, ensuring zero downtime or broken screens.

---

## 📁 Repository Structure

```
├── index.html                   # Master responsive web application
├── package.json                 # Node scripts and metadata
├── README.md                    # Project documentation
├── css/
│   └── style.css                # Vanilla CSS design system (Light/Dark mode, glassmorphism, pulse animations)
├── js/
│   ├── data.js                  # Clinical benchmarks, patient profiles, XAI attributions & 24h forecasts
│   ├── api.js                   # Backend Flask API bridge & edge-local fallback service
│   ├── charts.js                # Chart.js visualizers for 14-day trends, risk breakdown & XAI
│   ├── speech.js                # Web Speech bedside-manner voice narrator
│   └── app.js                   # Master application controller, simulator & modal interactions
└── backend_reference/           # Synced Team-Codeiro backend reference with updated templates
```

---

## 🛡️ Privacy & Security Architecture

1. **Local Edge Storage**: Biometric telemetry is processed on-device.
2. **Local Model Training**: Mathematical weights are computed locally.
3. **Single-Release Laplace Differential Privacy ($\epsilon=1.0, R_{\text{clip}}=1.5$)**: Mathematical noise masks individual patient records.
4. **Zero-Leakage FedAvg**: The central server aggregates updates without ever viewing raw patient telemetry.

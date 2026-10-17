# -*- coding: utf-8 -*-
"""
Google Fit Disease & Health Event Prediction Engine
Uses the trained Federated Global Model (google_fit_global_model.pkl)
to predict patient health condition and clinical risk breakdown.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from scipy.special import softmax

# Enable utf-8 encoding for Windows terminal output
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


from fit_pipeline import MODEL_FEATURE_COLS, CLINICAL_DEFAULTS, compute_derived_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(BASE_DIR, "google_fit_global_model.pkl")

def load_global_model():
    """Loads the trained privacy-preserving federated model bundle."""
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"Global model not found at {MODEL_FILE}. "
            "Please run 'python federated_fit_server.py' first to train the model."
        )
    with open(MODEL_FILE, "rb") as f:
        return pickle.load(f)

def predict_health_risk(vitals_dict: dict) -> dict:
    """
    Predict health event risk and condition from a dictionary of raw vitals.
    Returns:
      - condition: str (Normal, Mild Event, Moderate Event, Severe Event)
      - probabilities: dict of class probabilities
      - risk_index: float
      - alert_level: str
      - key_anomalies: list of string descriptions of abnormal vitals
    """
    bundle = load_global_model()
    coef = bundle["coef"]          # shape: (4, n_features)
    intercept = bundle["intercept"] # shape: (4,)
    mean = bundle["mean"]
    scale = bundle["scale"]
    class_names = bundle["class_names"]

    # Fill defaults for any missing vitals
    row_data = {}
    for col in ['heart_rate', 'blood_oxygen', 'blood_pressure_systolic', 'blood_pressure_diastolic', 
                'glucose_level', 'body_temperature', 'respiratory_rate', 'activity_level', 
                'sleep_quality', 'stress_level', 'hrv_sdnn', 'steps_count', 'calories_burned']:
        row_data[col] = vitals_dict.get(col, CLINICAL_DEFAULTS.get(col, 0.0))

    # Calculate stress_level if not provided
    if "stress_level" not in vitals_dict:
        hr = row_data["heart_rate"]
        sleep = row_data["sleep_quality"]
        row_data["stress_level"] = round(float(np.clip(2.0 + (hr - 65)/20 + (7.5 - sleep)/2, 1.0, 5.0)), 1)

    if "hrv_sdnn" not in vitals_dict:
        row_data["hrv_sdnn"] = round(float(np.clip(100.0 - row_data["stress_level"] * 12.0, 20.0, 120.0)), 1)

    # Compute derived interaction features
    df = pd.DataFrame([row_data])
    df = compute_derived_features(df)
    feature_vector = df[MODEL_FEATURE_COLS].values[0]

    # Global standardization (Zero-Leakage Scaler)
    scaled_vector = (feature_vector - mean) / scale

    # Model inference
    raw_logits = np.dot(coef, scaled_vector) + intercept
    probs = softmax(raw_logits)
    pred_class_idx = int(np.argmax(probs))
    pred_condition = class_names[pred_class_idx]

    # Detect specific vital anomalies
    anomalies = []
    if row_data["blood_pressure_systolic"] >= 140 or row_data["blood_pressure_diastolic"] >= 90:
        anomalies.append(f"Hypertension Alert (BP: {row_data['blood_pressure_systolic']}/{row_data['blood_pressure_diastolic']} mmHg)")
    elif row_data["blood_pressure_systolic"] >= 130:
        anomalies.append(f"Elevated Blood Pressure ({row_data['blood_pressure_systolic']} mmHg)")

    if row_data["blood_oxygen"] < 92.0:
        anomalies.append(f"Critical Hypoxemia (SpO2: {row_data['blood_oxygen']}%)")
    elif row_data["blood_oxygen"] < 95.0:
        anomalies.append(f"Low Oxygen Saturation (SpO2: {row_data['blood_oxygen']}%)")

    if row_data["heart_rate"] > 100:
        anomalies.append(f"Tachycardia / Elevated Resting Heart Rate ({row_data['heart_rate']} bpm)")
    elif row_data["heart_rate"] < 50:
        anomalies.append(f"Bradycardia ({row_data['heart_rate']} bpm)")

    if row_data["sleep_quality"] < 5.0:
        anomalies.append(f"Severe Sleep Deprivation ({row_data['sleep_quality']} hrs)")

    if row_data["glucose_level"] > 140:
        anomalies.append(f"Hyperglycemia Alert (Glucose: {row_data['glucose_level']} mg/dL)")

    if df['vital_risk_index'].values[0] >= 6.5:
        anomalies.append(f"High Vital Risk Index ({df['vital_risk_index'].values[0]:.2f})")

    # Risk badge
    risk_color_map = {
        "Normal": "LOW RISK (NORMAL)",
        "Mild Event": "MILD RISK (MONITOR)",
        "Moderate Event": "MODERATE RISK (ATTENTION NEEDED)",
        "Severe Event": "HIGH RISK (CRITICAL EVENT ALERT)"
    }

    return {
        "condition": pred_condition,
        "badge": risk_color_map.get(pred_condition, "UNKNOWN"),
        "class_index": pred_class_idx,
        "vital_risk_index": round(float(df['vital_risk_index'].values[0]), 2),
        "probabilities": {class_names[i]: round(float(probs[i]), 4) for i in range(len(class_names))},
        "anomalies": anomalies,
        "vitals_input": row_data
    }

def print_patient_report(report: dict, patient_name: str = "Google Fit Patient"):
    print("\n" + "=" * 60)
    print(f" HEALTH CONDITION PREDICTION REPORT: {patient_name.upper()}")
    print("=" * 60)
    print(f" Predicted Status : {report['condition'].upper()}")
    print(f" Status Badge     : {report['badge']}")
    print(f" Vital Risk Index : {report['vital_risk_index']}")
    print("-" * 60)
    print(" Class Probabilities:")
    for cname, p in report['probabilities'].items():
        bar = "#" * int(p * 30)
        print(f"   {cname:<16}: {p*100:>6.2f}% | {bar}")
    print("-" * 60)
    if report["anomalies"]:
        print(" Clinical Alerts / Anomalies Detected:")
        for a in report["anomalies"]:
            print(f"   [!] {a}")
    else:
        print(" Clinical Alerts: None (All monitored vitals within safe physiological ranges)")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict Health Event from Google Fit Telemetry")
    parser.add_argument("--hr", type=float, default=74.0, help="Heart Rate (bpm)")
    parser.add_argument("--spo2", type=float, default=98.0, help="Blood Oxygen SpO2 (%)")
    parser.add_argument("--bp-sys", type=float, default=120.0, help="Systolic Blood Pressure (mmHg)")
    parser.add_argument("--bp-dia", type=float, default=80.0, help="Diastolic Blood Pressure (mmHg)")
    parser.add_argument("--glucose", type=float, default=95.0, help="Blood Glucose (mg/dL)")
    parser.add_argument("--sleep", type=float, default=7.5, help="Sleep duration (hours)")
    parser.add_argument("--steps", type=float, default=7000.0, help="Daily Steps")
    parser.add_argument("--patient", type=str, default="User Test", help="Patient / User Name")
    args = parser.parse_args()

    vitals = {
        "heart_rate": args.hr,
        "blood_oxygen": args.spo2,
        "blood_pressure_systolic": args.bp_sys,
        "blood_pressure_diastolic": args.bp_dia,
        "glucose_level": args.glucose,
        "sleep_quality": args.sleep,
        "steps_count": args.steps,
    }

    report = predict_health_risk(vitals)
    print_patient_report(report, patient_name=args.patient)

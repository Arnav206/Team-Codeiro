# -*- coding: utf-8 -*-
"""
Master Orchestrator: Google Fit Federated Learning & Differential Privacy Pipeline
Provides a unified interface to authenticate, extract data, train federated models,
and run disease risk predictions.
"""

import sys
import os
import argparse

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from google_fit_auth import authenticate_user, list_authenticated_users
from google_fit_fetcher import fetch_user_fit_data
from fit_pipeline import process_fit_pipeline, save_client_dataset
from federated_fit_server import run_federated_fit_training
from predict_fit_user import predict_health_risk, print_patient_report

def banner():
    print("""
========================================================================
   FEDERATED HEALTH AI - GOOGLE FIT & DIFFERENTIAL PRIVACY PIPELINE   
   Decentralized Clinical Risk Assessment with Zero Raw Data Leakage   
========================================================================
""")

def run_demo():
    banner()
    print("[PHASE 1] Initializing Edge Client Nodes (Google Fit Users)...")
    print("  Client 1: User 1 (Active athlete, high steps, normal vitals)")
    print("  Client 2: User 2 (Sedentary, border BP, elevated stress)")
    print("  Client 3: User 3 (Shift worker, sleep deficit, cardiac strain)")
    
    print("\n[PHASE 2] Executing Privacy-Preserving Federated Training...")
    run_federated_fit_training(
        user_ids=["user1", "user2", "user3"],
        rounds=5,
        local_epochs=25,
        epsilon=1.0,
        use_dp=True
    )

    print("\n[PHASE 3] Live Patient Inference Demonstration...")
    
    sample_patients = [
        {
            "name": "Arnav (Normal Profile)",
            "vitals": {"heart_rate": 68.0, "blood_oxygen": 98.5, "blood_pressure_systolic": 118.0, "blood_pressure_diastolic": 78.0, "sleep_quality": 8.0, "steps_count": 9200}
        },
        {
            "name": "Patient B (Hypertensive & Low Sleep)",
            "vitals": {"heart_rate": 96.0, "blood_oxygen": 95.0, "blood_pressure_systolic": 142.0, "blood_pressure_diastolic": 92.0, "sleep_quality": 4.5, "steps_count": 3100}
        },
        {
            "name": "Patient C (Critical Cardiac & SpO2 Warning)",
            "vitals": {"heart_rate": 128.0, "blood_oxygen": 89.0, "blood_pressure_systolic": 165.0, "blood_pressure_diastolic": 105.0, "sleep_quality": 3.0, "steps_count": 1200}
        }
    ]

    for p in sample_patients:
        res = predict_health_risk(p["vitals"])
        print_patient_report(res, patient_name=p["name"])

def main():
    parser = argparse.ArgumentParser(description="Master Google Fit Federated Learning Orchestrator")
    parser.add_argument("--demo", action="store_true", help="Run full end-to-end demonstration")
    parser.add_argument("--auth", type=str, metavar="USER_ID", help="Authenticate a Google Fit user (e.g. --auth user1)")
    parser.add_argument("--fetch", type=str, metavar="USER_ID", help="Fetch Google Fit data for an authenticated user")
    parser.add_argument("--days", type=int, default=30, help="Days of telemetry to fetch (default: 30)")
    parser.add_argument("--train", action="store_true", help="Run Federated Learning with Differential Privacy")
    parser.add_argument("-e", "--epsilon", type=float, default=1.0, help="Privacy Budget Epsilon for training (default: 1.0)")
    parser.add_argument("--rounds", type=int, default=5, help="Communication rounds (default: 5)")
    parser.add_argument("--predict", action="store_true", help="Run health condition prediction")
    parser.add_argument("--hr", type=float, default=72.0, help="Heart rate (bpm)")
    parser.add_argument("--spo2", type=float, default=98.0, help="Blood Oxygen (%)")
    parser.add_argument("--bp-sys", type=float, default=120.0, help="Systolic BP (mmHg)")
    parser.add_argument("--bp-dia", type=float, default=80.0, help="Diastolic BP (mmHg)")
    parser.add_argument("--sleep", type=float, default=7.5, help="Sleep duration (hours)")
    parser.add_argument("--patient", type=str, default="Sample Patient", help="Patient name for report")
    args = parser.parse_args()

    if args.demo:
        run_demo()
        return

    if args.auth:
        authenticate_user(user_id=args.auth)
        return

    if args.fetch:
        raw_df = fetch_user_fit_data(user_id=args.fetch, days=args.days)
        processed_df = process_fit_pipeline(raw_df)
        save_path = os.path.join(os.path.dirname(__file__), f"fit_data_{args.fetch}.json")
        save_client_dataset(processed_df, save_path)
        print(f"Data pipeline complete for '{args.fetch}'. Saved to {save_path}")
        return

    if args.train:
        run_federated_fit_training(rounds=args.rounds, epsilon=args.epsilon)
        return

    if args.predict:
        vitals = {
            "heart_rate": args.hr,
            "blood_oxygen": args.spo2,
            "blood_pressure_systolic": args.bp_sys,
            "blood_pressure_diastolic": args.bp_dia,
            "sleep_quality": args.sleep
        }
        res = predict_health_risk(vitals)
        print_patient_report(res, patient_name=args.patient)
        return

    # Default if no arguments provided
    banner()
    parser.print_help()

if __name__ == "__main__":
    main()

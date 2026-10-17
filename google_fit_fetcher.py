# -*- coding: utf-8 -*-
"""
Google Fit Data Extraction Engine
Fetches daily aggregated telemetry (steps, HR, calories, sleep, vitals)
for an authenticated user over a specified time window.
"""

import os
import sys
import time
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np
from googleapiclient.discovery import build
from google_fit_auth import authenticate_user

# Data Types to Aggregate
AGGREGATE_METRICS = [
    {"dataTypeName": "com.google.step_count.delta"},
    {"dataTypeName": "com.google.heart_rate.bpm"},
    {"dataTypeName": "com.google.calories.expended"},
    {"dataTypeName": "com.google.active_minutes"},
    {"dataTypeName": "com.google.oxygen_saturation"},
    {"dataTypeName": "com.google.blood_pressure"},
    {"dataTypeName": "com.google.blood_glucose"},
    {"dataTypeName": "com.google.body.temperature"},
]

def fetch_user_fit_data(user_id: str = "user1", days: int = 30) -> pd.DataFrame:
    """
    Fetch daily aggregated telemetry from Google Fit REST API for a specific user.
    Returns a DataFrame with raw daily metrics.
    """
    creds = authenticate_user(user_id=user_id)
    service = build("fitness", "v1", credentials=creds, cache_discovery=False)

    now = datetime.now(timezone.utc)
    start_date = now - timedelta(days=days)
    
    # Align to midnight UTC for clean daily buckets
    start_midnight = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)
    start_time_ms = int(start_midnight.timestamp() * 1000)
    end_time_ms = int(now.timestamp() * 1000)

    request_body = {
        "aggregateBy": AGGREGATE_METRICS,
        "bucketByTime": {"durationMillis": 86400000},  # 24 hours in ms
        "startTimeMillis": start_time_ms,
        "endTimeMillis": end_time_ms,
    }

    print(f"[FIT FETCHER] Requesting past {days} days of telemetry for '{user_id}'...")
    response = service.users().dataset().aggregate(userId="me", body=request_body).execute()

    buckets = response.get("bucket", [])
    records = []

    for bucket in buckets:
        start_ms = int(bucket.get("startTimeMillis", 0))
        date_str = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")

        row = {
            "date": date_str,
            "steps_count": None,
            "heart_rate": None,
            "calories_burned": None,
            "activity_level": None,
            "blood_oxygen": None,
            "blood_pressure_systolic": None,
            "blood_pressure_diastolic": None,
            "glucose_level": None,
            "body_temperature": None,
            "sleep_quality": None,
            "stress_level": None,
            "hrv_sdnn": None,
            "respiratory_rate": None,
        }

        dataset = bucket.get("dataset", [])
        for ds in dataset:
            data_source_id = ds.get("dataSourceId", "")
            points = ds.get("point", [])
            if not points:
                continue

            for pt in points:
                values = pt.get("value", [])
                if not values:
                    continue

                # Step Count Delta
                if "step_count" in data_source_id:
                    row["steps_count"] = sum(v.get("intVal", 0) for v in values)
                # Heart Rate BPM (average / summary)
                elif "heart_rate" in data_source_id:
                    # Summary dataset usually gives [avg, max, min] in fpVal
                    hr_vals = [v.get("fpVal", 0.0) for v in values if v.get("fpVal") is not None]
                    if hr_vals:
                        row["heart_rate"] = float(hr_vals[0])
                # Calories Expended
                elif "calories" in data_source_id:
                    row["calories_burned"] = sum(v.get("fpVal", 0.0) for v in values)
                # Active Minutes
                elif "active_minutes" in data_source_id:
                    row["activity_level"] = sum(v.get("intVal", 0) for v in values)
                # Oxygen Saturation (SpO2)
                elif "oxygen_saturation" in data_source_id:
                    row["blood_oxygen"] = values[0].get("fpVal")
                # Blood Pressure [systolic, diastolic]
                elif "blood_pressure" in data_source_id:
                    if len(values) >= 2:
                        row["blood_pressure_systolic"] = values[0].get("fpVal")
                        row["blood_pressure_diastolic"] = values[1].get("fpVal")
                # Blood Glucose
                elif "blood_glucose" in data_source_id:
                    row["glucose_level"] = values[0].get("fpVal")
                # Body Temperature
                elif "body.temperature" in data_source_id:
                    row["body_temperature"] = values[0].get("fpVal")

        records.append(row)

    # Fetch Sleep Sessions via sessions API
    try:
        sleep_response = service.users().sessions().list(
            startTime=(start_midnight.isoformat()),
            endTime=now.isoformat(),
            activityType=72  # Activity type 72 is Sleep in Google Fit
        ).execute()

        sleep_sessions = sleep_response.get("session", [])
        # Aggregate sleep duration (hours) per date
        sleep_by_date = {}
        for session in sleep_sessions:
            s_start = int(session.get("startTimeMillis", 0))
            s_end = int(session.get("endTimeMillis", 0))
            s_date = datetime.fromtimestamp(s_start / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
            duration_hrs = (s_end - s_start) / (1000 * 3600)
            sleep_by_date[s_date] = sleep_by_date.get(s_date, 0.0) + duration_hrs

        for r in records:
            if r["date"] in sleep_by_date:
                r["sleep_quality"] = sleep_by_date[r["date"]]
    except Exception as e:
        print(f"[FIT FETCHER] Sleep sessions fetch notice: {e}")

    df = pd.DataFrame(records)
    print(f"[FIT FETCHER] Retrieved {len(df)} daily telemetry records for '{user_id}'.")
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fetch Google Fit Telemetry")
    parser.add_argument("--user", type=str, default="user1", help="User ID to fetch data for")
    parser.add_argument("--days", type=int, default=14, help="Number of past days to fetch (default: 14)")
    args = parser.parse_args()

    data = fetch_user_fit_data(user_id=args.user, days=args.days)
    print("\nTelemetry Preview:")
    print(data.head())

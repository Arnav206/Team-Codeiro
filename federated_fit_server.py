# -*- coding: utf-8 -*-
"""
Federated Learning Aggregator for Google Fit Telemetry
Coordinates privacy-preserving FedAvg across multiple Google Fit client nodes
with Single-Release Laplace Output Differential Privacy.
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from fit_client_node import FitClientNode, ALPHA, R_CLIP
from fit_pipeline import MODEL_FEATURE_COLS
from google_fit_auth import list_authenticated_users

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "google_fit_global_model.pkl")

CLASS_NAMES = ["Normal", "Mild Event", "Moderate Event", "Severe Event"]

def run_federated_fit_training(
    user_ids=None,
    rounds=5,
    local_epochs=30,
    epsilon=1.0,
    use_dp=True
):
    print("\n" + "=" * 65)
    print(" Federated Learning Aggregator for Google Fit Telemetry")
    print("=" * 65)

    if not user_ids:
        # Check authenticated tokens or use standard 3 client nodes
        auth_users = list_authenticated_users()
        if len(auth_users) >= 2:
            user_ids = auth_users
            print(f"Discovered {len(user_ids)} authenticated Google Fit users: {user_ids}")
        else:
            # Default to 3 representative client nodes (user1, user2, user3)
            user_ids = ["user1", "user2", "user3"]
            print(f"Running multi-client federated training on: {user_ids}")

    print(f"Privacy Configuration:")
    print(f"  Differential Privacy : {'Enabled (Laplace Output DP)' if use_dp else 'Disabled'}")
    if use_dp:
        print(f"  Privacy Budget (eps) : {epsilon}")
        print(f"  Sensitivity Clip (R) : {R_CLIP}")
        print(f"  Regularization alpha : {ALPHA}")
    print(f"  Communication Rounds : {rounds}")
    print(f"  Local Epochs / Round : {local_epochs}")
    print("-" * 65)

    # 1. Initialize Client Nodes
    clients = []
    for uid in user_ids:
        c = FitClientNode(user_id=uid, epsilon=epsilon if use_dp else 0.0)
        clients.append(c)

    # 2. Privacy-Preserving Federated Feature Statistics Aggregation (Zero Data Leakage)
    print("\n[STEP 1] Aggregating Zero-Leakage Global Feature Statistics...")
    total_samples = 0
    sum_mean = np.zeros(len(MODEL_FEATURE_COLS))
    for c in clients:
        n_i, mean_i, _ = c.get_feature_stats()
        total_samples += n_i
        sum_mean += n_i * mean_i
    
    global_mean = sum_mean / max(total_samples, 1)

    sum_var = np.zeros(len(MODEL_FEATURE_COLS))
    for c in clients:
        n_i, mean_i, var_i = c.get_feature_stats()
        sum_var += n_i * var_i + n_i * ((mean_i - global_mean) ** 2)
    
    global_var = sum_var / max(total_samples, 1)
    global_scale = np.sqrt(np.maximum(global_var, 1e-8))

    # Broadcast global scaler to all edge clients
    for c in clients:
        c.set_federated_scaler(global_mean, global_scale)
    print(f"  Global standardization complete across {total_samples} total telemetry records.")

    # 3. Federated Averaging Rounds
    print(f"\n[STEP 2] Starting {rounds} Federated Communication Rounds (FedAvg)...")
    
    # Initial weights
    sample_coef, sample_int = clients[0].get_weights()
    global_coef = np.zeros_like(sample_coef)
    global_intercept = np.zeros_like(sample_int)

    for r in range(1, rounds + 1):
        is_final_round = (r == rounds)
        print(f"\n--- Communication Round {r}/{rounds} ---")

        round_coefs = []
        round_intercepts = []
        round_weights = []

        for c in clients:
            # Set latest global weights before local training
            if r > 1:
                c.set_weights(global_coef, global_intercept)

            # Local training on edge client
            c.local_train(epochs=local_epochs)

            # Check if DP noise should be injected (Single-Release at final round)
            if is_final_round and use_dp:
                noisy_coef, noisy_int = c.get_dp_weights(custom_epsilon=epsilon)
                round_coefs.append(noisy_coef)
                round_intercepts.append(noisy_int)
                print(f"  [{c.user_id}] Injected Laplace noise (epsilon={epsilon})")
            else:
                clean_coef, clean_int = c.get_weights()
                round_coefs.append(clean_coef)
                round_intercepts.append(clean_int)

            n_samples = c.get_train_size()
            round_weights.append(n_samples)

        # FedAvg Aggregation on Server
        total_w = sum(round_weights)
        new_coef = np.zeros_like(global_coef)
        new_int = np.zeros_like(global_intercept)

        for w_idx in range(len(clients)):
            weight_factor = round_weights[w_idx] / total_w
            new_coef += weight_factor * round_coefs[w_idx]
            new_int += weight_factor * round_intercepts[w_idx]

        global_coef = new_coef
        global_intercept = new_int

    print(f"\n[STEP 3] Global Model Convergence Complete.")

    # 4. Evaluate Global Model on Combined Client Test Sets
    all_X_test = []
    all_y_test = []
    client_test_accuracies = {}

    for c in clients:
        X_te, y_te = c.get_test_data()
        all_X_test.append(X_te)
        all_y_test.append(y_te)

        # Set global weights on client model to evaluate
        c.set_weights(global_coef, global_intercept)
        preds = c.model.predict(X_te)
        acc = accuracy_score(y_te, preds)
        client_test_accuracies[c.user_id] = acc

    combined_X_test = np.vstack(all_X_test)
    combined_y_test = np.concatenate(all_y_test)

    eval_model = clients[0].model
    eval_model.coef_ = global_coef.copy()
    eval_model.intercept_ = global_intercept.copy()
    global_preds = eval_model.predict(combined_X_test)
    global_acc = accuracy_score(combined_y_test, global_preds)

    print("\n" + "=" * 65)
    print(" FEDERATED EVALUATION BENCHMARK")
    print("=" * 65)
    print(f"Global Test Accuracy : {global_acc * 100:.2f}%")
    for uid, acc in client_test_accuracies.items():
        print(f"  {uid:<15} Test Accuracy : {acc * 100:.2f}%")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(combined_y_test, global_preds, labels=[0, 1, 2, 3])
    header = f"{'Actual / Pred':<15}" + "".join([f"{c:>12}" for c in CLASS_NAMES])
    print(header)
    print("-" * len(header))
    for idx, cname in enumerate(CLASS_NAMES):
        row = f"{cname:<15}" + "".join([f"{cm[idx][j]:>12}" for j in range(4)])
        print(row)

    # 5. Save Global Model & Scaler Artifacts
    model_bundle = {
        "coef": global_coef,
        "intercept": global_intercept,
        "mean": global_mean,
        "scale": global_scale,
        "feature_names": MODEL_FEATURE_COLS,
        "class_names": CLASS_NAMES,
        "epsilon": epsilon if use_dp else 0.0,
        "use_dp": use_dp
    }

    with open(MODEL_SAVE_PATH, "wb") as f:
        pickle.dump(model_bundle, f)
    print(f"\n[SAVE] Trained Federated Global Model saved to: {MODEL_SAVE_PATH}")

    return model_bundle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated Server for Google Fit Telemetry")
    parser.add_argument("--no-dp", action="store_true", help="Run without Differential Privacy")
    parser.add_argument("-e", "--epsilon", type=float, default=1.0, help="Privacy Budget (Epsilon) [default: 1.0]")
    parser.add_argument("--rounds", type=int, default=5, help="Number of FL communication rounds [default: 5]")
    parser.add_argument("--epochs", type=int, default=30, help="Local epochs per round [default: 30]")
    parser.add_argument("--users", type=str, default="", help="Comma-separated user IDs (e.g. user1,user2,user3)")
    args = parser.parse_args()

    u_list = [u.strip() for u in args.users.split(",") if u.strip()] if args.users else None
    run_federated_fit_training(
        user_ids=u_list,
        rounds=args.rounds,
        local_epochs=args.epochs,
        epsilon=args.epsilon,
        use_dp=not args.no_dp
    )

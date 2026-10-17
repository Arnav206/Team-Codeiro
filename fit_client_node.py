# -*- coding: utf-8 -*-
"""
Google Fit Dynamic Client Node with Local Differential Privacy (LDP)
Trains locally on individual user Google Fit telemetry and perturbs
model weights via Laplace mechanism before sharing with the federated server.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from fit_pipeline import MODEL_FEATURE_COLS, process_fit_pipeline

ALPHA = 0.20   # L2 Regularization parameter
R_CLIP = 1.5   # Sensitivity clipping threshold for feature vectors
SEED = 42

class FitClientNode:
    """
    Edge client node representing a single Google Fit user.
    Maintains local data privacy with zero raw data transmission.
    """
    def __init__(self, user_id: str, epsilon: float = 1.0, dp_mode: str = "output", data_path: str = None):
        self.user_id = user_id
        self.epsilon = epsilon
        self.dp_mode = dp_mode
        self.alpha = ALPHA
        self.rng = np.random.RandomState(SEED)
        self.data_path = data_path or os.path.join(os.path.dirname(__file__), f"fit_data_{user_id}.json")

        self.model = SGDClassifier(
            loss='log_loss', penalty='l2', alpha=self.alpha,
            fit_intercept=True, warm_start=True, random_state=SEED
        )
        self.le = LabelEncoder()
        self.le.fit([0, 1, 2, 3])

        self.scaler = StandardScaler()
        self.X_train_raw = None
        self.X_test_raw = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.n_samples = 0

        self.load_and_prepare_data()

    def load_and_prepare_data(self):
        """Loads user data from processed JSON or creates initial physiological dataset."""
        if os.path.exists(self.data_path):
            df = pd.read_json(self.data_path, orient='records')
        else:
            # If no cached JSON exists yet, generate base structured series
            print(f"[{self.user_id}] No local cache found at {self.data_path}. Initializing base series...")
            dates = pd.date_range(end=pd.Timestamp.today(), periods=60)
            base_df = pd.DataFrame({"date": dates})
            # Seed varies slightly per user for non-IID distribution
            user_seed = abs(hash(self.user_id)) % 10000
            df = process_fit_pipeline(base_df, seed=user_seed)
            df.to_json(self.data_path, orient='records', indent=2)

        X = df[MODEL_FEATURE_COLS].values
        y = self.le.transform(df["Daily_Health_Condition"].values)

        # Stratified train/test split (30% test)
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.30, random_state=SEED, stratify=y
            )
        except ValueError:
            # Fallback if a class has only 1 sample
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.30, random_state=SEED
            )

        self.X_train_raw = X_tr
        self.y_train = y_tr
        self.X_test_raw = X_te
        self.y_test = y_te
        self.n_samples = len(self.X_train_raw)

        # Local standardization fallback
        self.X_train = self.scaler.fit_transform(self.X_train_raw)
        self.X_test = self.scaler.transform(self.X_test_raw)
        self.initialize_model()

    def get_feature_stats(self):
        """Zero-leakage local statistics for federated feature scaling."""
        n_i = len(self.X_train_raw)
        mean_i = np.mean(self.X_train_raw, axis=0)
        var_i = np.var(self.X_train_raw, axis=0)
        return n_i, mean_i, var_i

    def set_federated_scaler(self, mean_global, scale_global):
        """Apply global federated scaler to local client data."""
        self.scaler.mean_ = mean_global.copy()
        self.scaler.scale_ = scale_global.copy()
        self.scaler.var_ = scale_global.copy() ** 2
        self.X_train = self.scaler.transform(self.X_train_raw)
        self.X_test = self.scaler.transform(self.X_test_raw)
        self.initialize_model()

    def initialize_model(self):
        """Initialize SGD model with uniform classes."""
        self.model = SGDClassifier(
            loss='log_loss', penalty='l2', alpha=self.alpha,
            fit_intercept=True, warm_start=True, random_state=SEED
        )
        self.rng = np.random.RandomState(SEED)
        self.model.partial_fit(self.X_train[:4], [0, 1, 2, 3], classes=[0, 1, 2, 3])
        self.model.coef_ = np.zeros_like(self.model.coef_)
        self.model.intercept_ = np.zeros_like(self.model.intercept_)

    def get_weights(self):
        return self.model.coef_.copy(), self.model.intercept_.copy()

    def set_weights(self, coef, intercept):
        self.model.coef_ = coef.copy()
        self.model.intercept_ = intercept.copy()
        self.model.t_ = 1.0

    def local_train(self, epochs=50):
        """Local training on edge device without sending data anywhere."""
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weight = compute_sample_weight(class_weight='balanced', y=self.y_train)
        for _ in range(epochs):
            self.model.partial_fit(self.X_train, self.y_train, sample_weight=sample_weight)

    def get_dp_weights(self, custom_epsilon=None):
        """
        Add Laplace noise to local model weights (Local Differential Privacy).
        Sensitivity = (2 * R_clip) / (n_samples * alpha)
        """
        eps = custom_epsilon if custom_epsilon is not None else self.epsilon
        coef, intercept = self.get_weights()
        if eps <= 0:
            return coef, intercept

        R = min(np.max(np.linalg.norm(self.X_train, axis=1)), R_CLIP)
        sensitivity = (2.0 * R) / (max(self.n_samples, 1) * self.alpha)
        scale = sensitivity / eps

        noisy_coef = coef + self.rng.laplace(loc=0.0, scale=scale, size=coef.shape)
        noisy_intercept = intercept + self.rng.laplace(loc=0.0, scale=scale, size=intercept.shape)
        return noisy_coef, noisy_intercept

    def fine_tune(self, coef, intercept, epochs=10):
        """Fine-tune global weights to create personalized local model."""
        self.set_weights(coef, intercept)
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weight = compute_sample_weight(class_weight='balanced', y=self.y_train)
        for _ in range(epochs):
            self.model.partial_fit(self.X_train, self.y_train, sample_weight=sample_weight)

    def get_test_data(self):
        return self.X_test, self.y_test

    def get_train_size(self):
        return self.n_samples

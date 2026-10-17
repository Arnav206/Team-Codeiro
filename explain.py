# -*- coding: utf-8 -*-
"""
Per-Patient Local Explanation Module (On-Device XAI)

PRIVACY BOUNDARY GUARANTEE:
This module executes strictly on-device at the client node.
It computes local SHAP explanations using ONLY the local model and local background data.
No raw patient data (X_patient), training data (X_train_local), or raw SHAP matrices are 
transmitted to the central server. Only lightweight text summaries (e.g. "heart_rate increased risk")
are combined into alerts locally.
"""

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

import numpy as np

def explain_prediction(local_model, X_train_local, X_patient, feature_names):
    """
    On-device SHAP explanation for a single patient prediction.
    
    Privacy note:
    This runs ENTIRELY locally, using only this client's own model and own training data 
    as SHAP background. Nothing here is transmitted to the server — same privacy boundary as local_train().
    """
    X_p2d = X_patient.reshape(1, -1)
    pred_class = int(local_model.predict(X_p2d)[0])
    
    contributions = None
    if HAS_SHAP:
        try:
            explainer = shap.LinearExplainer(local_model, X_train_local)
            shap_values = explainer.shap_values(X_p2d)
            if isinstance(shap_values, list):
                contributions = shap_values[pred_class][0]
            elif isinstance(shap_values, np.ndarray):
                if shap_values.ndim == 3:
                    if shap_values.shape[0] == 1 and shap_values.shape[2] > pred_class:
                        contributions = shap_values[0, :, pred_class]
                    else:
                        contributions = shap_values[0, pred_class, :]
                elif shap_values.ndim == 2:
                    contributions = shap_values[0]
                else:
                    contributions = shap_values.ravel()
            else:
                contributions = np.array(shap_values).ravel()
        except Exception:
            contributions = None

    if contributions is None:
        # Exact linear Shapley attribution fallback: phi_j = w_j * (x_j - E[X_j])
        mean_bg = np.mean(X_train_local, axis=0) if X_train_local is not None and len(X_train_local) > 0 else 0.0
        coef = local_model.coef_
        c_idx = pred_class if coef.ndim == 2 and coef.shape[0] > pred_class else 0
        contributions = coef[c_idx] * (X_patient - mean_bg)

    # Rank top-3 contributing features by absolute magnitude
    ranked = sorted(
        zip(feature_names, contributions),
        key=lambda x: -abs(x[1])
    )[:3]
    
    explanation_parts = []
    for fname, val in ranked:
        direction = "increased" if val > 0 else "decreased"
        explanation_parts.append(f"{fname} {direction} risk")
        
    return pred_class, explanation_parts


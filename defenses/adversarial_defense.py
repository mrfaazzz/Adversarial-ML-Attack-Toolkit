"""
defenses/adversarial_defense.py
---------------------------------
Four defenses against adversarial attacks:

  1. Adversarial Training  — retrain on a mix of clean + adversarial samples
  2. Feature Squeezing     — reduce feature precision to kill small perturbations
  3. Gaussian Smoothing    — average multiple noisy copies to cancel adversarial signal
  4. Ensemble Defense      — three diverse models vote; hard to fool all at once

Run compare_defenses() to see all results side by side.
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb

SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "saved")
os.makedirs(SAVE_DIR, exist_ok=True)


# ── Defense 1: Adversarial Training ──────────────────────────────────────────
def adversarial_training(base_model, X_train_clean: np.ndarray, y_train: np.ndarray,
                          X_train_adv: np.ndarray, augment_ratio: float = 0.5):
    """
    Retrain the model on a mix of clean and adversarial examples.
    augment_ratio controls what fraction of adversarial samples to add.
    """
    from models.train_model import TorchMLP

    n_adv = int(len(X_train_adv) * augment_ratio)
    idx   = np.random.choice(len(X_train_adv), n_adv, replace=False)
    X_aug = np.vstack([X_train_clean, X_train_adv[idx]])
    y_aug = np.concatenate([y_train, y_train[idx]])

    perm  = np.random.permutation(len(X_aug))
    X_aug, y_aug = X_aug[perm], y_aug[perm]

    print(f"[Defense] Adversarial training: {len(X_aug):,} samples "
          f"({len(X_train_clean):,} clean + {n_adv:,} adversarial)")

    if isinstance(base_model, TorchMLP):
        hardened = TorchMLP(input_dim=base_model.input_dim, epochs=30)
        hardened.fit(X_aug, y_aug)
    elif isinstance(base_model, RandomForestClassifier):
        hardened = RandomForestClassifier(
            n_estimators=base_model.n_estimators,
            max_depth=base_model.max_depth,
            random_state=42, n_jobs=-1, class_weight="balanced",
        )
        hardened.fit(X_aug, y_aug)
    else:
        hardened = xgb.XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            eval_metric="logloss", random_state=42, n_jobs=-1,
        )
        hardened.fit(X_aug, y_aug)

    print("[Defense] Hardened model trained.")
    return hardened


# ── Defense 2: Feature Squeezing ─────────────────────────────────────────────
def feature_squeezing(X: np.ndarray, bit_depth: int = 4) -> np.ndarray:
    """
    Quantize features to a fixed bit-depth.
    Tiny adversarial perturbations fall below the quantization threshold and get removed.
    """
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    span  = x_max - x_min
    span[span == 0] = 1.0  # avoid divide-by-zero

    levels = (2 ** bit_depth) - 1
    return (np.round((X - x_min) / span * levels) / levels * span + x_min).astype(np.float32)


# ── Defense 3: Gaussian Smoothing ────────────────────────────────────────────
def gaussian_smoothing(X: np.ndarray, sigma: float = 0.05) -> np.ndarray:
    """
    Average 5 noisy copies of each sample.
    The structured adversarial signal cancels out; the real signal survives.
    """
    copies = [X + np.random.normal(0, sigma, X.shape) for _ in range(5)]
    return np.mean(copies, axis=0).astype(np.float32)


# ── Defense 4: Ensemble Defense ───────────────────────────────────────────────
def build_ensemble_defense(X_train: np.ndarray, y_train: np.ndarray):
    """
    Train 3 diverse models (RF x2 + XGBoost) with soft-voting.
    Hard to fool all three simultaneously.
    """
    rf1  = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=1, n_jobs=-1)
    rf2  = RandomForestClassifier(n_estimators=150, max_depth=14, random_state=2, n_jobs=-1)
    xgb1 = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        eval_metric="logloss", random_state=3, n_jobs=-1,
    )
    ensemble = VotingClassifier(
        estimators=[("rf1", rf1), ("rf2", rf2), ("xgb", xgb1)],
        voting="soft",
    )
    print("[Defense] Training ensemble (RF x2 + XGBoost)...")
    ensemble.fit(X_train, y_train)
    print("[Defense] Ensemble ready.")
    return ensemble


# ── Compare all defenses side by side ────────────────────────────────────────
def compare_defenses(original_model, hardened_model, X_clean: np.ndarray,
                     X_adv: np.ndarray, y_true: np.ndarray,
                     squeezed_X_adv: np.ndarray = None,
                     smoothed_X_adv: np.ndarray = None) -> dict:
    """
    Print an accuracy comparison table for all defense configurations.
    Returns a dict of {label: accuracy}.
    """
    configs = [
        ("Original — clean input", original_model, X_clean),
        ("Original — adv input",   original_model, X_adv),
        ("Hardened — adv input",   hardened_model, X_adv),
    ]
    if squeezed_X_adv is not None:
        configs.append(("Original + squeezing — adv", original_model, squeezed_X_adv))
    if smoothed_X_adv is not None:
        configs.append(("Original + smoothing — adv", original_model, smoothed_X_adv))

    results = {}
    print(f"\n  {'─'*60}")
    print(f"  {'Configuration':<42} {'Accuracy':>8}")
    print(f"  {'─'*60}")
    for label, model, X in configs:
        acc = accuracy_score(y_true, model.predict(X))
        results[label] = acc
        print(f"  {label:<42} {acc:>8.4f}")
    print(f"  {'─'*60}")
    return results


# ── Save hardened model ───────────────────────────────────────────────────────
def save_hardened_model(model, name: str = "hardened_model") -> str:
    path = os.path.join(SAVE_DIR, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"[Defense] Saved hardened model → {path}")
    return path
